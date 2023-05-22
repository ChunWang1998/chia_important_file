import dataclasses
import logging
from typing import Any, ClassVar, TypeAlias, cast

import numpy as np
from blspy import G2Element
from chia.rpc.full_node_rpc_client import FullNodeRpcClient
from chia.types.blockchain_format.coin import Coin
from chia.types.blockchain_format.program import INFINITE_COST, Program
from chia.types.blockchain_format.serialized_program import SerializedProgram
from chia.types.blockchain_format.sized_bytes import bytes32
from chia.types.coin_record import CoinRecord
from chia.types.coin_spend import CoinSpend, compute_additions
from chia.types.condition_opcodes import ConditionOpcode
from chia.types.condition_with_args import ConditionWithArgs
from chia.types.spend_bundle import SpendBundle
from chia.util.condition_tools import (
    conditions_dict_for_solution,
    conditions_for_solution,
    created_outputs_for_conditions_dict,
)
from chia.util.ints import uint64, uint128
from chia.wallet.cat_wallet.cat_utils import (
    SpendableCAT,
    construct_cat_puzzle,
    unsigned_spend_bundle_for_spendable_cats,
)
from chia.wallet.coin_selection import select_coins
from chia.wallet.lineage_proof import LineageProof
from chia.wallet.outer_puzzles import (
    AssetType,
    driver_lookup,
    get_inner_puzzle,
    match_puzzle,
)
from chia.wallet.payment import Payment
from chia.wallet.puzzles.cat_loader import CAT_MOD, CAT_MOD_HASH
from chia.wallet.puzzles.singleton_top_layer_v1_1 import (
    SINGLETON_LAUNCHER_HASH,
    SINGLETON_MOD_HASH,
    SINGLETON_MOD,
    launch_conditions_and_coinsol,
    lineage_proof_for_coinsol,
    solution_for_singleton,
)
from chia.wallet.trading.offer import OFFER_MOD_HASH, NotarizedPayment, Offer
from chia.wallet.uncurried_puzzle import UncurriedPuzzle, uncurry_puzzle
from clvm import SExp
from clvm_tools.binutils import disassemble

from pyke.v2.puzzle_drivers import PuzzleInfo, Solver, encode_info_value
from pyke.v2.puzzles import (
    ACS_MOD,
    ACS_MOD_HASH,
    LIQUIDITY_TAIL_MOD,
    P2_SINGLETON_V2_MOD,
    P2_SINGLETON_V2_MOD_HASH,
    POOL_MOD,
    POOL_MOD_HASH,
)

RequestedPayments: TypeAlias = dict[bytes32 | None, list[NotarizedPayment]]


@dataclasses.dataclass
class InvalidOfferError(Exception):
    residue: float
    message: str

    def __str__(self) -> str:
        return self.message


@dataclasses.dataclass(frozen=True)
class PoolPuzzleInfo(PuzzleInfo):
    TYPE: ClassVar[str] = "HashgreenSwap AMM"

    def __init__(self, info: dict[str, Any]):
        info = info.copy()
        info.update({"type": self.TYPE})

        super().__init__(info=info)

    @classmethod
    def create(cls, info: dict[str, Any]) -> "PoolPuzzleInfo":
        if "fee" not in info:
            info["fee"] = 300

        if "amount" not in info:
            # 下一代的singleton 的amount 會根據這個
            info["amount"] = 1

        info.update(
            {
                "asset_amount_v": 0,
                "asset_amount_a": 0,
                "asset_amount_b": 0,
            }
        )

        return cls(info=info)


@dataclasses.dataclass
class PoolPuzzleDriver(object):
    @staticmethod
    def verify_asset_belief(
        asset_belief: Program,
        singleton_mod: Program | None = None,
        cat_mod_hash: Program | None = None,
        offer_mod_hash: Program | None = None,
        p2_singleton_mod_hash: Program | None = None,
        asset_id: Program | None = None,
        trusted: bool = True,
    ) -> int:
        if not trusted:
            logging.warning(
                "Parameter `trusted` is not implemented and is ignored for now."
            )

        delta: int = 0

        incoming_payments: Program = asset_belief.at("f")
        for incoming_payment in incoming_payments.as_iter():
            delta += Program(incoming_payment).at("rrf").as_int()

        outgoing_payments: Program = asset_belief.at("r")
        for outgoing_payment in outgoing_payments.as_iter():
            delta -= Program(outgoing_payment).at("rrf").as_int()

        return delta

    @staticmethod
    def verify_liquidity_belief(
        liquidity_belief: Program,
        singleton_mod: Program | None = None,
        trusted: bool = True,
    ) -> int:
        if not trusted:
            logging.warning(
                "Parameter `trusted` is not implemented and is ignored for now."
            )

        delta: int = 0

        liquidity_deltas: Program = liquidity_belief
        for liquidity_delta in liquidity_deltas.as_iter():
            delta += Program(liquidity_delta).at("r").as_int()

        return delta

    def match(
        self, uncurried_puzzle: UncurriedPuzzle, solution: Program, trusted: bool = True
    ) -> PuzzleInfo | None:
        if uncurried_puzzle.mod != POOL_MOD:
            return None

        (
            pool_mod,
            singleton_mod,
            cat_mod_hash,
            offer_mod_hash,
            p2_singleton_mod_hash,
            asset_id_amount_v,
            asset_id_amount_a,
            asset_id_amount_b,
        ) = uncurried_puzzle.args.as_iter()

        (
            liquidity_belief,
            asset_belief_a,
            asset_belief_b,
        ) = solution.as_iter()

        delta_v: int = self.verify_liquidity_belief(
            liquidity_belief=Program(liquidity_belief), trusted=trusted
        )
        delta_a: int = self.verify_asset_belief(
            asset_belief=Program(asset_belief_a), trusted=trusted
        )
        delta_b: int = self.verify_asset_belief(
            asset_belief=Program(asset_belief_b), trusted=trusted
        )

        fee: int = Program(pool_mod).at("rf").as_int()
        asset_amount_v: int = Program(asset_id_amount_v).at("r").as_int()
        asset_amount_a: int = Program(asset_id_amount_a).at("r").as_int()
        asset_amount_b: int = Program(asset_id_amount_b).at("r").as_int()

        do_verify: bool
        if trusted:
            do_verify = False
        else:
            is_eve: bool = all(
                [asset_amount_a == 0, asset_amount_b == 0, asset_amount_v == 0]
            )

            do_verify = not is_eve

        if do_verify:
            # AMM equation verification
            fee_float: float = fee / 10_000
            a: float = 1 + delta_a / asset_amount_a
            b: float = 1 + delta_b / asset_amount_b
            v: float = 1 + delta_v / asset_amount_v

            residue: bool = (
                (np.log(a) + np.log(b))
                - 0.5 * fee_float * np.abs(np.log(a) - np.log(b))
            ) - 2 * np.log(v)

            if residue < 0:
                details: dict[str, Any] = {
                    "fee": fee_float,
                    "a": a,
                    "b": b,
                    "v": v,
                    "residue": residue,
                }

                raise InvalidOfferError(
                    residue=residue,
                    message=(
                        f"The offer is offering insufficient assets and/or requesting too much. "
                        f"Details: {details}."
                    ),
                )

        return PoolPuzzleInfo(
            {
                "fee": fee,
                "amount": Program(pool_mod).at("rr").as_int(),
                "singleton_launcher_id": Program(singleton_mod).at("rf"),
                "asset_id_v": Program(asset_id_amount_v).at("f"),
                "asset_amount_v": asset_amount_v + delta_v,
                "asset_id_a": Program(asset_id_amount_a).at("f"),
                "asset_amount_a": asset_amount_a + delta_a,
                "asset_id_b": Program(asset_id_amount_b).at("f"),
                "asset_amount_b": asset_amount_b + delta_b,
            }
        )

    def construct(self, puzzle_info: PuzzleInfo) -> Program:
        return POOL_MOD.curry(
            (POOL_MOD_HASH, (puzzle_info.fee, puzzle_info.amount)),
            (
                SINGLETON_MOD_HASH,
                (puzzle_info.singleton_launcher_id, SINGLETON_LAUNCHER_HASH),
            ),
            CAT_MOD_HASH,
            OFFER_MOD_HASH,
            P2_SINGLETON_V2_MOD_HASH,
            (puzzle_info.asset_id_v, puzzle_info.asset_amount_v),
            (puzzle_info.asset_id_a, puzzle_info.asset_amount_a),
            (puzzle_info.asset_id_b, puzzle_info.asset_amount_b),
        )


@dataclasses.dataclass
class Singleton(object):
    full_node_client: FullNodeRpcClient

    puzzle_info: PuzzleInfo
    puzzle_driver: Any = driver_lookup[AssetType.SINGLETON]

    _genesis_coin: Coin | None = None
    # 為了知道目前的latest coin 的state(要知道他爸的coinspend才知道現在的state)
    # 在pool sync 時會被更新到鏈上最新的pool 資訊 
    _latest_parent_coin_spend: CoinSpend | None = None
    _latest_coin_record: CoinRecord | None = None

    @property
    def _latest_coin(self) -> Coin:
        if self._latest_coin_record is None:
            raise ValueError("Please sync pool first!")

        return self._latest_coin_record.coin

    @staticmethod
    def get_payment_puzzle(launcher_id: bytes32) -> Program:
        return P2_SINGLETON_V2_MOD.curry(
            (SINGLETON_MOD_HASH, (launcher_id, SINGLETON_LAUNCHER_HASH))
        )

    @property
    def id(self) -> bytes32:
        return self.puzzle_info.launcher_id

    @property
    def is_eve(self) -> bool:
        assert self._latest_parent_coin_spend is not None

        return (
            self._latest_parent_coin_spend.puzzle_reveal.get_tree_hash()
            == self.puzzle_info.launcher_ph
        )

    @property
    def inner_puzzle(self) -> Program:
        pass

    @property
    def inner_puzzle_hash(self) -> bytes32:
        pass

    @property
    def payment_puzzle(self) -> Program:
        return self.get_payment_puzzle(launcher_id=self.id)

    @property
    def payment_puzzle_hash(self) -> bytes32:
        return self.payment_puzzle.get_tree_hash()

    async def _select_coins(
        self, coin_records: list[CoinRecord], amount: int
    ) -> list[CoinRecord]:
        coin_record_by_id: dict[bytes32, CoinRecord] = {
            coin_record.coin.name(): coin_record for coin_record in coin_records
        }
        amounts: list[int] = [coin_record.coin.amount for coin_record in coin_records]
        spendable_amount: int = sum(amounts)
        max_coin_amount: int = max(amounts)

        coins: set[Coin] = await select_coins(
            spendable_amount=uint128(spendable_amount),
            max_coin_amount=uint64(max_coin_amount),
            spendable_coins=coin_records,  # type: ignore
            unconfirmed_removals={},
            log=logging.getLogger("chia"),
            amount=uint128(amount),
        )

        return [coin_record_by_id[coin.name()] for coin in coins]
    
    async def _pay_to_conditions(
        self,
        coins: list[Coin],
        conditions: list[SExp],
        asset_id: bytes32 | None,
        lineage_proofs: list[LineageProof] | None = None,
    ) -> SpendBundle:
        coin_spends: list[CoinSpend] = []
        spendable_cats: list[SpendableCAT] = []

        if asset_id is not None:
            if lineage_proofs is None:
                logging.warning("LineageProof is empty, maybe you forgot?")
                lineage_proofs = [LineageProof() for _ in coins]

        for num, coin in enumerate(coins):
            coin_conditions: list = []
            if coin == coins[0]:
                coin_conditions = conditions
            # create announcement 給資產assert
            solution = Program.to(
                [self.inner_puzzle_hash, coin.name(), coin_conditions]
            )
            solution = Program(solution)

            if asset_id is None:
                # 被授權的asset 的coin spend, 需要去assert singleton 的announcement
                coin_spend = CoinSpend(
                    coin=coin,
                    puzzle_reveal=SerializedProgram.from_program(self.payment_puzzle),
                    solution=SerializedProgram.from_program(solution),
                )
                coin_spends.append(coin_spend)

            else:
                assert lineage_proofs is not None

                spendable_cat = SpendableCAT(
                    coin=coin,
                    limitations_program_hash=asset_id,
                    inner_puzzle=self.payment_puzzle,
                    inner_solution=solution,
                    lineage_proof=lineage_proofs[num],
                    extra_delta=0,
                )
                spendable_cats.append(spendable_cat)

        spend_bundle: SpendBundle
        if asset_id is None:
            spend_bundle = SpendBundle(
                coin_spends=coin_spends, aggregated_signature=G2Element()
            )
        else:
            spend_bundle = unsigned_spend_bundle_for_spendable_cats(
                CAT_MOD, spendable_cat_list=spendable_cats
            )

        return spend_bundle

    def _mint_melt_liquidity(
        self,
        coin: Coin,
        asset_id: bytes32,
        tail_program: Program,
        tail_solution: Program,
        payment: Payment | None = None,
        lineage_proof: LineageProof | None = None,
    ) -> SpendBundle:
        # - if payment is not `None`, mint to `puzzle_hash`
        # - if payment is `None`, melt the coin

        tail_program_hash: bytes32 = tail_program.get_tree_hash()
        if tail_program_hash != asset_id:
            raise ValueError("Provided TAIL program does not match asset id!")

        if payment is not None:
            if coin.amount != payment.amount:
                raise ValueError("Payment should consume the whole ephemeral CAT coin!")

        if lineage_proof is None:
            lineage_proof = LineageProof()

        amount: int = coin.amount

        # `conditions` are for eve CAT coin

        conditions: list[SExp] = [
            Program.to(
                [
                    ConditionOpcode.CREATE_COIN,
                    None,
                    -113,
                    tail_program,
                    tail_solution,
                ]
            )
        ]
        extra_delta: int = -amount
        if payment is not None:
            conditions.append(
                Program.to([ConditionOpcode.CREATE_COIN, payment.puzzle_hash, amount])
            )
            extra_delta += amount

        spendable_cat = SpendableCAT(
            coin=coin,
            limitations_program_hash=tail_program_hash,
            inner_puzzle=ACS_MOD,
            inner_solution=Program.to(conditions),  # type: ignore
            limitations_program_reveal=tail_program,
            limitations_solution=tail_solution,
            lineage_proof=lineage_proof,
            extra_delta=extra_delta,
        )
        spend_bundle = unsigned_spend_bundle_for_spendable_cats(
            CAT_MOD, spendable_cat_list=[spendable_cat]
        )

        return spend_bundle


@dataclasses.dataclass
class Pool(Singleton):
    inner_puzzle_driver: Any = dataclasses.field(default_factory=PoolPuzzleDriver)
    inner_puzzle_info: PuzzleInfo = dataclasses.field(kw_only=True)

    @staticmethod
    def get_liquidity_tail_program(launcher_id: bytes32):
        return LIQUIDITY_TAIL_MOD.curry(
            (
                SINGLETON_MOD_HASH,
                (launcher_id, SINGLETON_LAUNCHER_HASH),
            )
        )

    @staticmethod
    def get_cat_puzzle(puzzle: Program, asset_id: bytes32 | None):
        full_puzzle: Program = puzzle
        if asset_id is not None:
            full_puzzle = construct_cat_puzzle(
                CAT_MOD,
                limitations_program_hash=asset_id,
                inner_puzzle=full_puzzle,
            )
        return full_puzzle

    @classmethod
    def create(
        cls,
        inner_puzzle_info: PuzzleInfo,
        full_node_client: FullNodeRpcClient,
        genesis_coin: Coin | None = None,
        launcher_id: bytes32 | None = None,
    ) -> "Pool":
        if genesis_coin is not None:
            assert launcher_id is None

            launcher_coin_spend: CoinSpend

            (_, launcher_coin_spend) = launch_conditions_and_coinsol(
                coin=genesis_coin,
                inner_puzzle=Program.to([]),  # type: ignore
                comment=[],
                amount=inner_puzzle_info.amount,
            )
            launcher_id = launcher_coin_spend.coin.name()

        if launcher_id is None:
            raise ValueError("Please specify `genesis_coin` or `launcher_id`!")

        liquidity_tail_program: Program = cls.get_liquidity_tail_program(
            launcher_id=launcher_id
        )
        liquidity_tail_program_hash: bytes32 = liquidity_tail_program.get_tree_hash()

        return cls(
            puzzle_info=PuzzleInfo(
                {
                    "type": "singleton",
                    "launcher_id": launcher_id,
                    "launcher_ph": SINGLETON_LAUNCHER_HASH,
                },
            ),
            inner_puzzle_info=PoolPuzzleInfo(
                {
                    "singleton_launcher_id": launcher_id,
                    "asset_id_v": liquidity_tail_program_hash,
                    **inner_puzzle_info.info,
                }
            ),
            full_node_client=full_node_client,
            _genesis_coin=genesis_coin,
        )

    # there is no launcher_id in the pool created from this func 
    @classmethod
    def create_from_offer(
        cls,
        inner_puzzle_info: PuzzleInfo,
        full_node_client: FullNodeRpcClient,
        offer: Offer | None = None,
        genesis_coin: Coin | None = None,
        launcher_id: bytes32 | None = None,
    ) -> "Pool":
        if genesis_coin is not None:
            assert launcher_id is None

            launcher_coin_spend: CoinSpend

            (_, launcher_coin_spend) = launch_conditions_and_coinsol(
                coin=genesis_coin,
                inner_puzzle=Program.to([]),  # type: ignore
                comment=[],
                amount=inner_puzzle_info.amount,
            )
            launcher_id = launcher_coin_spend.coin.name()

        if launcher_id is None:
            raise ValueError("Please specify `genesis_coin` or `launcher_id`!")

        liquidity_tail_program: Program = cls.get_liquidity_tail_program(
            launcher_id=launcher_id
        )
        liquidity_tail_program_hash: bytes32 = liquidity_tail_program.get_tree_hash()

        return cls(
            puzzle_info=PuzzleInfo(
                {
                    "type": "singleton",
                    "launcher_id": launcher_id,
                    "launcher_ph": SINGLETON_LAUNCHER_HASH,
                },
            ),
            inner_puzzle_info=PoolPuzzleInfo(
                {
                    "singleton_launcher_id": launcher_id,
                    "asset_id_v": liquidity_tail_program_hash,
                    **inner_puzzle_info.info,
                }
            ),
            full_node_client=full_node_client,
            _genesis_coin=genesis_coin,
        )

    @property
    def liquidity_tail_program(self) -> Program:
        return self.get_liquidity_tail_program(launcher_id=self.id)

    @property
    def liquidity_tail_program_hash(self) -> bytes32:
        return self.liquidity_tail_program.get_tree_hash()

    @property
    def driver_dict(self) -> dict[bytes32, PuzzleInfo]:
        driver_dict: dict[bytes32, PuzzleInfo] = {}

        asset_id: bytes32 | None
        for asset_id in [
            self.inner_puzzle_info.asset_id_v,
            self.inner_puzzle_info.asset_id_a,
            self.inner_puzzle_info.asset_id_b,
        ]:
            if asset_id is None:
                continue

            driver_dict[asset_id] = PuzzleInfo({"type": "CAT", "tail": asset_id})

        return driver_dict

    @property
    def inner_puzzle(self) -> Program:
        if self.inner_puzzle_info is None:
            raise ValueError(
                "`inner_puzzle_info` not supplied! Please create or sync pool first."
            )

        return self.inner_puzzle_driver.construct(self.inner_puzzle_info)

    @property
    def inner_puzzle_hash(self) -> bytes32:
        return self.inner_puzzle.get_tree_hash()

    @property
    def puzzle(self) -> Program:
        return self.puzzle_driver.construct(
            self.puzzle_info, inner_puzzle=self.inner_puzzle
        )

    @property
    def puzzle_hash(self) -> bytes32:
        return self.puzzle.get_tree_hash()

    def format_summary(self):
        info: PuzzleInfo = self.inner_puzzle_info

        def human_amount(amount: int, asset_id: bytes32 | None) -> float:
            if asset_id is None:
                return amount / 10**12
            else:
                return amount / 10**3

        def amount_str(amount: int, asset_id: bytes32 | None) -> str:
            if asset_id is None:
                return f"{amount / 10 ** 12:.12f}"
            else:
                return f"{amount / 10 ** 3:.3f}"

        s: str = (
            f"\n"
            f"{info.type_}\n"
            f"- Pool ID: {encode_info_value(info.singleton_launcher_id)}\n"
            f"- Fee: {info.fee / 10_000:.2%}\n"
            f"- Asset A:\n"
            f"  - Asset ID: {encode_info_value(info.asset_id_a)}\n"
            f"  - Amount: {amount_str(info.asset_amount_a, info.asset_id_a)}\n"
            f"- Asset B:\n"
            f"  - Asset ID: {encode_info_value(info.asset_id_b)}\n"
            f"  - Amount: {amount_str(info.asset_amount_b, info.asset_id_b)}\n"
            f"- Liquidity Token:\n"
            f"  - Asset ID: {encode_info_value(info.asset_id_v)}\n"
            f"  - Amount: {amount_str(info.asset_amount_v, info.asset_id_v)}\n"
        )

        if (info.asset_amount_a == 0) or (info.asset_amount_b == 0):
            return s

        price_a_to_b: float = human_amount(
            info.asset_amount_a, info.asset_id_a
        ) / human_amount(info.asset_amount_b, info.asset_id_b)

        s += (
            f"- Stats:\n"
            f"  - Price (B per A): {price_a_to_b:.6f}\n"
            f"  - Price (A per B): {1 / price_a_to_b:.6f}\n"
        )
        return s

    def print_summary(self):
        logging.info(self.format_summary())

    def _get_launch_conditions_and_coin_spend(
        self,
    ) -> tuple[list[Program], CoinSpend]:
        if self._genesis_coin is None:
            raise ValueError("Pool not in a launching setup!")

        # pool singleton

        singleton_conditions: list[Program]
        launcher_coin_spend: CoinSpend

        (singleton_conditions, launcher_coin_spend) = launch_conditions_and_coinsol(
            coin=self._genesis_coin,
            inner_puzzle=self.inner_puzzle,
            comment=[],
            amount=self.inner_puzzle_info.amount,
        )

        return (singleton_conditions, launcher_coin_spend)

    def get_launch_conditions(self) -> list[Program]:
        conditions: list[Program]
        (conditions, _) = self._get_launch_conditions_and_coin_spend()

        return conditions

    async def get_launch_spend_bundle(self, spend_bundle: SpendBundle) -> SpendBundle:
        conditions: list[Program]
        launcher_coin_spend: CoinSpend
        (conditions, launcher_coin_spend) = self._get_launch_conditions_and_coin_spend()

        # check conditions has been made

        payer_conditions: list[SExp] = []
        for coin_spend in spend_bundle.coin_spends:
            (_, payer_coin_conditions, _) = conditions_for_solution(
                puzzle_reveal=coin_spend.puzzle_reveal,
                solution=coin_spend.solution,
                max_cost=INFINITE_COST,
            )

            assert payer_coin_conditions is not None

            payer_coin_condition: ConditionWithArgs
            for payer_coin_condition in payer_coin_conditions:
                payer_conditions.append(
                    Program.to((payer_coin_condition.opcode, payer_coin_condition.vars))
                )

        for condition in conditions:
            if condition not in payer_conditions:
                raise ValueError(
                    f"Condition '{disassemble(condition)}' not found in payer spend bundle!"
                )

        # construct final spend bundle

        launcher_spend_bundle = SpendBundle(
            coin_spends=[launcher_coin_spend], aggregated_signature=G2Element()
        )
        spend_bundle = SpendBundle.aggregate([spend_bundle, launcher_spend_bundle])

        return spend_bundle

    def _update_state(
        self, latest_parent_coin_spend: CoinSpend, latest_coin_record: CoinRecord
    ) -> None:
        coin_spend: CoinSpend = latest_parent_coin_spend
        coin_record: CoinRecord = latest_coin_record

        self._latest_parent_coin_spend = coin_spend
        self._latest_coin_record = latest_coin_record

        coin: Coin = coin_record.coin

        logging.info(f"Updating latest singleton to coin '0x{coin.name().hex()}'.")

        if self.is_eve:
            logging.info(
                "Latest pool spend not found. It is likely we have an eve singleton and "
                "we will rely on supplied `inner_puzzle_info` about the pool."
            )

            if self.inner_puzzle_info is None:
                raise ValueError("No `inner_puzzle_info` supplied!")

            if self.puzzle.get_tree_hash() != coin.puzzle_hash:
                raise ValueError(
                    "Supplied `inner_puzzle_info` invalid when checked against the launcher!"
                )

        else:
            puzzle: UncurriedPuzzle = uncurry_puzzle(
                coin_spend.puzzle_reveal.to_program()
            )
            solution: Program = coin_spend.solution.to_program()

            inner_puzzle: Program = self.puzzle_driver.get_inner_puzzle(
                self.puzzle_info, puzzle_reveal=puzzle
            )
            inner_solution: Program = self.puzzle_driver.get_inner_solution(
                self.puzzle_info, solution=solution
            )

            inner_puzzle_info: PuzzleInfo = self.inner_puzzle_driver.match(
                uncurry_puzzle(inner_puzzle), inner_solution
            )

            self.inner_puzzle_info = inner_puzzle_info

            logging.info(f"Updating `inner_puzzle_info` to {inner_puzzle_info.info}.")

    async def sync(
        self,
        coin_id: bytes32 | None = None,
        allow_fallback: bool = True,
        max_steps: int | None = None,
    ) -> bool:
        """Synchorizes the information about the pool.

        - If `self.params` is `None`, then we rely on the spends from the chain
        to reconstruct.

        - If `self.params` is not `None`, then we check the latest spend against it.
        """

        coin_record: CoinRecord | None = None

        # user has provided a desired point of sync
        if coin_id is not None:
            coin_record = await self.full_node_client.get_coin_record_by_name(coin_id)
            if coin_record is None:
                logging.warning(
                    "Search failed! If you are specifying `coin_id`, please make sure it is a valid pool singleton"
                )

        # user has not provided point of sync, or the point of sync is problematic
        if coin_record is None:
            if not allow_fallback:
                raise ValueError(
                    "Please provide a valid point of sync when `allow_fallback` is False."
                )
            # refer to cached latest point of sync
            if self._latest_coin_record is not None:
                coin_id = self._latest_coin_record.coin.name()
                logging.info(
                    f"Coin id not provided, now using latest coin '0x{coin_id.hex()}'."
                )

            # last resort: use genesis singleton
            else:
                coin_id = self.id
                logging.info(
                    f"Coin id not provided, now using launcher coin '0x{coin_id.hex()}'."
                )

            coin_record = await self.full_node_client.get_coin_record_by_name(coin_id)

        # all resorts failed!
        if coin_record is None:
            raise ValueError("Search failed! Maybe your pool info is misconfigured!")

        assert coin_id is not None

        coin_spend: CoinSpend | None = None
        created_coin: Coin | None = None

        current_steps: int = 0

        # we are the latest singleton coin
        if not coin_record.spent:
            if (self._latest_parent_coin_spend is not None) and (
                self._latest_parent_coin_spend.coin.name()
                == coin_record.coin.parent_coin_info
            ):
                logging.info("We are already synced.")
                return True

            else:
                logging.info(
                    f"Coin '0x{coin_id.hex()}' is already the latest pool singleton. "
                    f"Rewinding a generation to obtain pool params."
                )
                coin_id = coin_record.coin.parent_coin_info
                coin_record = await self.full_node_client.get_coin_record_by_name(
                    coin_id
                )
                current_steps -= 1

        assert coin_record is not None

        while True:
            # two breaking conditions: either we've reached max sync steps or we are at the top of chain
            if ((max_steps is not None) and (current_steps >= max_steps)) or (
                not coin_record.spent
            ):
                break

            logging.info(
                f"Start searching for singleton from coin id '0x{coin_id.hex()}'."
            )
            # 抓已經上鏈的資料
            coin_spend = await self.full_node_client.get_puzzle_and_solution(
                coin_id, coin_record.spent_block_index
            )
            assert coin_spend is not None

            (_, conditions_dict, _) = conditions_dict_for_solution(
                coin_spend.puzzle_reveal, coin_spend.solution, INFINITE_COST
            )
            assert conditions_dict is not None

            created_coins: list[Coin] = created_outputs_for_conditions_dict(
                conditions_dict=conditions_dict,
                input_coin_name=coin_record.coin.name(),
            )
            created_coins = list(filter(lambda coin: coin.amount % 2, created_coins))

            if len(created_coins) != 1:
                raise ValueError(
                    f"Coin '0x{coin_id.hex()}' creating more than one odd coins!"
                )

            created_coin = created_coins[0]
            coin_id = created_coin.name()
            current_steps += 1

            coin_record = await self.full_node_client.get_coin_record_by_name(coin_id)
            if coin_record is None:
                raise ValueError(f"Coin '0x{coin_id.hex()}' not found.")

        assert coin_spend is not None
        assert coin_record is not None

        self._update_state(
            latest_parent_coin_spend=coin_spend, latest_coin_record=coin_record
        )

        return False

    async def get_coin_records(
        self, amount: int, asset_id: bytes32 | None, consolidate_coins: bool
    ) -> list[CoinRecord]:
        payment_puzzle: Program = self.get_cat_puzzle(
            puzzle=self.payment_puzzle, asset_id=asset_id
        )
        payment_puzzle_hash: bytes32 = payment_puzzle.get_tree_hash()

        coin_records: list[
            CoinRecord
        ] = await self.full_node_client.get_coin_records_by_puzzle_hash(
            puzzle_hash=payment_puzzle_hash, include_spent_coins=False
        )
        if len(coin_records) == 0:
            raise ValueError(
                f"Insufficient coin count for singleton {encode_info_value(self.id)} "
                f"and asset {encode_info_value(asset_id)}!"
            )
        # 預設盡量合併 coin
        if len(coin_records) > 1 and (not consolidate_coins):
            logging.warning(
                f"Singleton {encode_info_value(self.id)} has more than one coins for "
                f"asset {encode_info_value(asset_id)}. Consider consolidating them."
            )
        if consolidate_coins:
            # spend all coins, aiming to consolidate all into one
            pass
        else:
            coin_records = await self._select_coins(coin_records, amount=amount)

        return coin_records

    async def get_cat_lineage_proof(self, coin_record: CoinRecord) -> LineageProof:
        coin: Coin = coin_record.coin
        parent_spend: CoinSpend | None = (
            await self.full_node_client.get_puzzle_and_solution(
                coin_id=coin.parent_coin_info,
                height=coin_record.confirmed_block_index,
            )
        )
        assert parent_spend is not None
        parent_coin: Coin = parent_spend.coin
        parent_puzzle: Program = parent_spend.puzzle_reveal.to_program()

        parent_uncurried_puzzle: UncurriedPuzzle = uncurry_puzzle(parent_puzzle)
        puzzle_info: PuzzleInfo | None = cast(
            PuzzleInfo | None, match_puzzle(puzzle=parent_uncurried_puzzle)
        )

        assert puzzle_info is not None

        inner_puzzle: Program | None = get_inner_puzzle(
            constructor=puzzle_info,
            puzzle_reveal=parent_uncurried_puzzle,
        )

        assert inner_puzzle is not None

        lineage_proof = LineageProof(
            parent_name=parent_coin.parent_coin_info,
            inner_puzzle_hash=inner_puzzle.get_tree_hash(),
            amount=uint64(parent_coin.amount),
        )
        return lineage_proof

    async def get_outgoing_asset_payments(
        self, amount: int, asset_id: bytes32 | None, consolidate_coins: bool
    ) -> tuple[SpendBundle, RequestedPayments]:
        # 你帳戶有的 coin 數量
        coin_records: list[CoinRecord] = await self.get_coin_records(
            amount=amount, asset_id=asset_id, consolidate_coins=consolidate_coins
        )
        coins: list[Coin] = [coin_record.coin for coin_record in coin_records]
        total_amount: int = sum([coin.amount for coin in coins])

        conditions: list[SExp] = [
            Program.to([ConditionOpcode.CREATE_COIN, OFFER_MOD_HASH, total_amount])
        ]

        lineage_proofs: list[LineageProof] | None = None
        if asset_id is not None:
            lineage_proofs = []
            for coin_record in coin_records:
                lineage_proof = await self.get_cat_lineage_proof(coin_record)
                lineage_proofs.append(lineage_proof)

        spend_bundle: SpendBundle = await self._pay_to_conditions(
            coins=coins,
            conditions=conditions,
            asset_id=asset_id,
            lineage_proofs=lineage_proofs,
        )

        # now that we are paying more than we should, we need to request for the change coin to come back
        nonce: bytes32 = Program.to([coin.name() for coin in coins]).get_tree_hash()  # type: ignore
        change_amount: int = total_amount - amount

        notarized_payment = NotarizedPayment(
            self.payment_puzzle_hash, uint64(change_amount), [], nonce
        )

        return (spend_bundle, {asset_id: [notarized_payment]})

    async def get_outgoing_lp_payments(
        self, amount: int, asset_id: bytes32, coin_for_mint: Coin
    ) -> tuple[SpendBundle, RequestedPayments]:
        cat_puzzle: Program = self.get_cat_puzzle(puzzle=ACS_MOD, asset_id=asset_id)
        cat_puzzle_hash: bytes32 = cat_puzzle.get_tree_hash()

        cat_coin = Coin(
            parent_coin_info=coin_for_mint.name(),
            puzzle_hash=cat_puzzle_hash,
            amount=amount,
        )

        payment = Payment(OFFER_MOD_HASH, uint64(amount), [])
        spend_bundle: SpendBundle = self._mint_melt_liquidity(
            coin=cat_coin,
            payment=payment,
            asset_id=asset_id,
            tail_program=self.liquidity_tail_program,
            tail_solution=Program.to([self.inner_puzzle_hash]),  # type: ignore
        )

        nonce: bytes32 = cat_coin.name()
        notarized_payment = NotarizedPayment(cat_puzzle_hash, uint64(amount), [], nonce)

        return (spend_bundle, {None: [notarized_payment]}) 

    async def get_launcher_payments(
        self, amount: int, launcher_coin: Coin
    ) -> tuple[SpendBundle, RequestedPayments]:
        # refer: test.py (mint pool singleton)
        # conditions = self.get_launch_conditions()

        # refer: wallet.py=> checking AMM equations / launch_conditions_and_coinsol()
        curried_singleton: Program = SINGLETON_MOD.curry(
            (SINGLETON_MOD_HASH, (launcher_coin.name(), SINGLETON_LAUNCHER_HASH)),
            self.inner_puzzle,
        )
        launcher_solution = Program.to(
            [
                curried_singleton.get_tree_hash(),
                amount,
                [],
            ]
        )
        coin_spend = CoinSpend(
            coin=launcher_coin,
            puzzle_reveal=SINGLETON_LAUNCHER_HASH,
            solution=launcher_solution,
        )
        spend_bundle: SpendBundle = SpendBundle(
            coin_spends=[coin_spend], aggregated_signature=G2Element()
        )

        requested_payment = {None: [Payment(SINGLETON_LAUNCHER_HASH, amount, [])]}
        notarized_payment = Offer.notarize_payments(
            requested_payment, coins=[launcher_coin]
        )

        return (spend_bundle, notarized_payment)

    async def get_incoming_asset_payments(
        self, amount: int, asset_id: bytes32 | None
    ) -> tuple[SpendBundle, RequestedPayments]:
        assert self._latest_coin is not None

        nonce: bytes32 = self._latest_coin.name()
        notarized_payment = NotarizedPayment(
            self.payment_puzzle_hash, uint64(amount), [], nonce
        )
        return (SpendBundle.aggregate([]), {asset_id: [notarized_payment]})

    async def get_incoming_lp_payments(
        self, amount: int, asset_id: bytes32, coin_for_melt: Coin
    ) -> tuple[SpendBundle, RequestedPayments]:
        assert coin_for_melt.amount == amount

        cat_puzzle: Program = self.get_cat_puzzle(puzzle=ACS_MOD, asset_id=asset_id)
        cat_puzzle_hash: bytes32 = cat_puzzle.get_tree_hash()

        cat_coin = Coin(
            parent_coin_info=coin_for_melt.name(),
            puzzle_hash=cat_puzzle_hash,
            amount=amount,
        )

        lineage_proof = LineageProof(
            parent_name=coin_for_melt.parent_coin_info,
            inner_puzzle_hash=OFFER_MOD_HASH,
            amount=uint64(amount),
        )
        spend_bundle: SpendBundle = self._mint_melt_liquidity(
            coin=cat_coin,
            asset_id=asset_id,
            tail_program=self.liquidity_tail_program,
            tail_solution=Program.to([self.inner_puzzle_hash]),  # type: ignore
            lineage_proof=lineage_proof,
        )

        nonce: bytes32 = cat_coin.name()
        # it is find to spend to `ACS` as the payment from pool requires liquidity token melting
        notarized_payment = NotarizedPayment(ACS_MOD_HASH, uint64(amount), [], nonce)

        return (spend_bundle, {asset_id: [notarized_payment]})

    async def get_inner_solution(
        self,
        spend_bundle: SpendBundle,
        requested_payments: RequestedPayments,
    ) -> Program:
        asset_id_a: bytes32 | None = self.inner_puzzle_info.asset_id_a
        asset_id_b: bytes32 = self.inner_puzzle_info.asset_id_b
        asset_id_v: bytes32 = self.inner_puzzle_info.asset_id_v

        # asset_belief

        asset_id_to_belief: dict[bytes32 | None, SExp] = {}
        for asset_id in [asset_id_a, asset_id_b]:
            # incoming payments

            incoming_payments: list[SExp] = []

            notarized_payments: list[NotarizedPayment] = requested_payments[asset_id]
            notarized_payment: NotarizedPayment
            for notarized_payment in notarized_payments:
                if notarized_payment.puzzle_hash != self.payment_puzzle_hash:
                    continue

                incoming_payments.append(
                    Program.to(
                        (notarized_payment.nonce, notarized_payment.as_condition_args())
                    )
                )

            # outgoing payments

            payment_puzzle: Program = self.get_cat_puzzle(
                puzzle=self.payment_puzzle, asset_id=asset_id
            )
            payment_puzzle_hash: bytes32 = payment_puzzle.get_tree_hash()

            outgoing_payments: list[SExp] = []

            coin_spend: CoinSpend
            for coin_spend in spend_bundle.coin_spends:
                if coin_spend.coin.puzzle_hash != payment_puzzle_hash:
                    continue

                coin: Coin = coin_spend.coin
                outgoing_payments.append(
                    Program.to([coin.parent_coin_info, coin.puzzle_hash, coin.amount])
                )

            asset_id_to_belief[asset_id] = Program.to(
                (incoming_payments, outgoing_payments)
            )

        # liquidity_belief

        acs_cat_puzzle: Program = self.get_cat_puzzle(
            puzzle=ACS_MOD, asset_id=asset_id_v
        )
        acs_cat_puzzle_hash: bytes32 = acs_cat_puzzle.get_tree_hash()

        liquidity_deltas: list[SExp] = []
        for coin_spend in spend_bundle.coin_spends:
            if coin_spend.coin.puzzle_hash != acs_cat_puzzle_hash:
                continue

            coin = coin_spend.coin
            additions: list[Coin] = compute_additions(coin_spend)
            amount: int = coin.amount if additions else -coin.amount

            liquidity_deltas.append(
                Program.to(
                    ([coin.parent_coin_info, coin.puzzle_hash, coin.amount], amount)
                )
            )

        liquidity_belief = Program.to(liquidity_deltas)

        # inner_solution

        inner_solution = Program.to(
            [
                liquidity_belief,
                asset_id_to_belief[asset_id_a],
                asset_id_to_belief[asset_id_b],
            ]
        )

        return Program(inner_solution)

    async def respond_to_offer(
        self,
        offer: Offer | None = None,
        consolidate_coins: bool = False,
    ) -> SpendBundle:
        if offer is None:
            offer = Offer({}, SpendBundle.aggregate([]), driver_dict={})

        logging.info(f"User offering assets: {offer.get_offered_amounts()}")
        logging.info(f"User requesting assets: {offer.get_requested_amounts()}")

        await self.sync()

        assert self._latest_coin is not None
        assert self._latest_parent_coin_spend is not None

        asset_id_v = self.inner_puzzle_info.asset_id_v
        asset_id_a = self.inner_puzzle_info.asset_id_a
        asset_id_b = self.inner_puzzle_info.asset_id_b

        pool_spend_bundles: list[SpendBundle] = []
        pool_requested_payments: dict[bytes32 | None, list[NotarizedPayment]] = {
            asset_id_v: [],
            asset_id_a: [],
            asset_id_b: [],
        }

        lineage_proof: LineageProof
        spend_bundle: SpendBundle
        requested_payments: RequestedPayments

        # outgoing payments

        requested_amounts: dict[bytes32 | None, int] = offer.get_requested_amounts()
        for asset_id, amount in requested_amounts.items():
            if asset_id in [asset_id_a, asset_id_b]:
                (
                    spend_bundle,
                    requested_payments,
                ) = await self.get_outgoing_asset_payments(
                    amount=amount,
                    asset_id=asset_id,
                    consolidate_coins=consolidate_coins,
                )

            elif asset_id == asset_id_v:
                assert asset_id is not None

                xch_coins: list[Coin] | None = offer.get_offered_coins().get(None)
                if xch_coins is None:
                    raise ValueError(
                        f"Since we are minting liquidity tokens here, please provide "
                        f"{amount} mojos as XCH in the offer!"
                    )

                genesis_coin = xch_coins[0]

                (
                    spend_bundle,
                    requested_payments,
                ) = await self.get_outgoing_lp_payments(
                    amount=amount, asset_id=asset_id, coin_for_mint=genesis_coin
                )

            else:
                raise ValueError(
                    f"Pool {self} cannot offer asset {encode_info_value(asset_id)}!"
                )

            pool_spend_bundles.append(spend_bundle)
            for asset_id, notarized_payments in requested_payments.items():
                pool_requested_payments[asset_id].extend(notarized_payments)

        # incoming payments

        offered_amounts: dict[bytes32 | None, int] = offer.get_offered_amounts()
        for asset_id, amount in offered_amounts.items():
            if amount == 0:
                raise ValueError(
                    f"Offering zero amount of asset {encode_info_value(asset_id)}! "
                    f"This will likely fail catastrophically so we might as well end your misery here."
                )

            # adjust XCH for liquidity token minting mojos
            if asset_id is None:
                notarized_payments = pool_requested_payments[asset_id]

                if notarized_payments:
                    used_amount: int = sum(
                        [payment.amount for payment in notarized_payments]
                    )
                    amount -= used_amount

                # okay if amount is zero it is acceptable, just skip requesting
                if amount == 0:
                    continue

            if asset_id in [asset_id_a, asset_id_b]:
                (
                    spend_bundle,
                    requested_payments,
                ) = await self.get_incoming_asset_payments(
                    amount=amount, asset_id=asset_id
                )

            elif asset_id == asset_id_v:
                assert asset_id is not None

                liq_coins: list[Coin] | None = offer.get_offered_coins().get(asset_id)

                if liq_coins is None:
                    raise ValueError("No coin offered for liquidity token!")

                if len(liq_coins) != 1:
                    raise ValueError("Expecting only singular liquidity token payment!")

                coin_for_melt = liq_coins[0]

                (
                    spend_bundle,
                    requested_payments,
                ) = await self.get_incoming_lp_payments(
                    amount=amount, asset_id=asset_id, coin_for_melt=coin_for_melt
                )

            else:
                raise ValueError(
                    f"Pool {self} cannot request asset {encode_info_value(asset_id)}!"
                )

            pool_spend_bundles.append(spend_bundle)
            for asset_id, notarized_payments in requested_payments.items():
                if notarized_payments:
                    pool_requested_payments[asset_id].extend(notarized_payments)

        # pool singleton

        inner_solution: Program = await self.get_inner_solution(
            SpendBundle.aggregate(pool_spend_bundles),
            pool_requested_payments,
        )

        solution: Program
        if self.is_eve:
            # SingletonOuterPuzzle does not deal with eve spends

            lineage_proof = lineage_proof_for_coinsol(self._latest_parent_coin_spend)
            solution = solution_for_singleton(
                lineage_proof=lineage_proof,
                amount=self.inner_puzzle_info.amount,
                inner_solution=inner_solution,
            )

        else:
            solution = self.puzzle_driver.solve(
                self.puzzle_info,
                solver=Solver(
                    {
                        "coin": bytes(self._latest_coin),
                        "parent_spend": bytes(self._latest_parent_coin_spend),
                    }
                ),
                inner_puzzle=self.inner_puzzle,
                inner_solution=inner_solution,
            )

        # checking AMM equations

        _ = self.inner_puzzle_driver.match(
            uncurried_puzzle=uncurry_puzzle(self.inner_puzzle),
            solution=inner_solution,
            trusted=False,
        )

        coin_spend = CoinSpend(
            coin=self._latest_coin,
            puzzle_reveal=SerializedProgram.from_program(self.puzzle),
            solution=SerializedProgram.from_program(solution),
        )
        spend_bundle = SpendBundle(
            coin_spends=[coin_spend], aggregated_signature=G2Element()
        )
        pool_spend_bundles.append(spend_bundle)

        # finalize

        # to ensure `driver_dict`s match
        offer = Offer(
            offer.requested_payments, offer._bundle, driver_dict=self.driver_dict
        )

        pool_spend_bundle = SpendBundle.aggregate(pool_spend_bundles)

        # requested_payments cannot be empty lists
        pool_requested_payments = {
            asset_id: notarized_payments
            for asset_id, notarized_payments in pool_requested_payments.items()
            if notarized_payments
        }
        logging.info(f"pool_requested_payments:{pool_requested_payments}")
        logging.info(f"pool_spend_bundle:{pool_spend_bundle}")
        pool_offer = Offer(
            pool_requested_payments, pool_spend_bundle, driver_dict=self.driver_dict
        )
        #如果總 offer >= 總 request 就是可上鏈
        agg_offer = Offer.aggregate([offer, pool_offer])
        return agg_offer.to_valid_spend()

    async def respond_to_offerV2(
        self,
        offer: Offer | None = None,
        consolidate_coins: bool = False,
    ) -> SpendBundle:
        if offer is None:
            offer = Offer({}, SpendBundle.aggregate([]), driver_dict={})

        logging.info(f"User offering assets: {offer.get_offered_amounts()}")
        logging.info(f"User requesting assets: {offer.get_requested_amounts()}")

        await self.sync()

        assert self._latest_coin is not None
        assert self._latest_parent_coin_spend is not None

        asset_id_v = self.inner_puzzle_info.asset_id_v
        asset_id_a = self.inner_puzzle_info.asset_id_a
        asset_id_b = self.inner_puzzle_info.asset_id_b

        pool_spend_bundles: list[SpendBundle] = []
        pool_requested_payments: dict[bytes32 | None, list[NotarizedPayment]] = {
            asset_id_v: [],
            asset_id_a: [],
            asset_id_b: [],
        }

        lineage_proof: LineageProof
        spend_bundle: SpendBundle
        requested_payments: RequestedPayments

        # ask 1 mojo from xch_coins for launcher to OFFER_MOD_HASH
        xch_coins: list[Coin] | None = offer.get_offered_coins().get(None)
        if xch_coins is None:
            raise ValueError(
                f"Since we are providing 1 mojo to launcher here, please provide "
                f"it in the offer!"
            )

        launcher_genesis_coin = xch_coins[0]
        (
            spend_bundle,
            requested_payments,
        ) = await self.get_launcher_payments(
            amount=self.inner_puzzle_info.amount, launcher_coin=launcher_genesis_coin
        )
        pool_spend_bundles.append(spend_bundle)
        for asset_id, notarized_payments in requested_payments.items():
            pool_requested_payments[asset_id].extend(notarized_payments)

        # outgoing payments

        requested_amounts: dict[bytes32 | None, int] = offer.get_requested_amounts()
        for asset_id, amount in requested_amounts.items():
            # for add liq, amount is liq_amount
            if asset_id in [asset_id_a, asset_id_b]:
                (
                    spend_bundle,
                    requested_payments,
                ) = await self.get_outgoing_asset_payments(
                    amount=amount,
                    asset_id=asset_id,
                    consolidate_coins=consolidate_coins,
                )

            elif asset_id == asset_id_v:
                assert asset_id is not None
                # xch_amount
                xch_coins: list[Coin] | None = offer.get_offered_coins().get(None)
                # logging.info(f"xch_coins:{xch_coins}")
                if xch_coins is None:
                    raise ValueError(
                        f"Since we are minting liquidity tokens here, please provide "
                        f"{amount} mojos as XCH in the offer!"
                    )

                genesis_coin = xch_coins[0]

                (
                    spend_bundle,
                    requested_payments,
                ) = await self.get_outgoing_lp_payments(
                    amount=amount, asset_id=asset_id, coin_for_mint=genesis_coin
                )
                # logging.info(f"normal spend_bundle here:{spend_bundle}")
            else:
                raise ValueError(
                    f"Pool {self} cannot offer asset {encode_info_value(asset_id)}!"
                )

            pool_spend_bundles.append(spend_bundle)
            for asset_id, notarized_payments in requested_payments.items():
                pool_requested_payments[asset_id].extend(notarized_payments)

        # incoming payments

        offered_amounts: dict[bytes32 | None, int] = offer.get_offered_amounts()
        logging.info(f"offered_amountssss:{offered_amounts}")
        for asset_id, amount in offered_amounts.items():
            if amount == 0:
                raise ValueError(
                    f"Offering zero amount of asset {encode_info_value(asset_id)}! "
                    f"This will likely fail catastrophically so we might as well end your misery here."
                )

            # adjust XCH for liquidity token minting mojos
            if asset_id is None:
                notarized_payments = pool_requested_payments[asset_id]

                if notarized_payments:
                    used_amount: int = sum(
                        [payment.amount for payment in notarized_payments]
                    )
                    amount -= used_amount

                # okay if amount is zero it is acceptable, just skip requesting
                if amount == 0:
                    continue

            if asset_id in [asset_id_a, asset_id_b]:
                # for asset_id_a, amount = xch_amount - liq_amount
                # for asset_id_b, amount = cat_amount
                (
                    spend_bundle,
                    requested_payments,
                ) = await self.get_incoming_asset_payments(
                    amount=amount, asset_id=asset_id
                )

            elif asset_id == asset_id_v:
                assert asset_id is not None

                liq_coins: list[Coin] | None = offer.get_offered_coins().get(asset_id)

                if liq_coins is None:
                    raise ValueError("No coin offered for liquidity token!")

                if len(liq_coins) != 1:
                    raise ValueError("Expecting only singular liquidity token payment!")

                coin_for_melt = liq_coins[0]

                (
                    spend_bundle,
                    requested_payments,
                ) = await self.get_incoming_lp_payments(
                    amount=amount, asset_id=asset_id, coin_for_melt=coin_for_melt
                )

            else:
                raise ValueError(
                    f"Pool {self} cannot request asset {encode_info_value(asset_id)}!"
                )

            pool_spend_bundles.append(spend_bundle)
            for asset_id, notarized_payments in requested_payments.items():
                if notarized_payments:
                    pool_requested_payments[asset_id].extend(notarized_payments)

        # pool singleton

        inner_solution: Program = await self.get_inner_solution(
            SpendBundle.aggregate(pool_spend_bundles),
            pool_requested_payments,
        )

        solution: Program
        if self.is_eve:
            # SingletonOuterPuzzle does not deal with eve spends

            lineage_proof = lineage_proof_for_coinsol(self._latest_parent_coin_spend)
            solution = solution_for_singleton(
                lineage_proof=lineage_proof,
                amount=self.inner_puzzle_info.amount,
                inner_solution=inner_solution,
            )

        else:
            solution = self.puzzle_driver.solve(
                self.puzzle_info,
                solver=Solver(
                    {
                        "coin": bytes(self._latest_coin),
                        "parent_spend": bytes(self._latest_parent_coin_spend),
                    }
                ),
                inner_puzzle=self.inner_puzzle,
                inner_solution=inner_solution,
            )

        # checking AMM equations

        _ = self.inner_puzzle_driver.match(
            uncurried_puzzle=uncurry_puzzle(self.inner_puzzle),
            solution=inner_solution,
            trusted=False,
        )

        coin_spend = CoinSpend(
            coin=self._latest_coin,
            puzzle_reveal=SerializedProgram.from_program(self.puzzle),
            solution=SerializedProgram.from_program(solution),
        )
        spend_bundle = SpendBundle(
            coin_spends=[coin_spend], aggregated_signature=G2Element()
        )
        pool_spend_bundles.append(spend_bundle)

        # finalize

        # to ensure `driver_dict`s match
        offer = Offer(
            offer.requested_payments, offer._bundle, driver_dict=self.driver_dict
        )

        pool_spend_bundle = SpendBundle.aggregate(pool_spend_bundles)

        # requested_payments cannot be empty lists
        pool_requested_payments = {
            asset_id: notarized_payments
            for asset_id, notarized_payments in pool_requested_payments.items()
            if notarized_payments
        }

        pool_offer = Offer(
            pool_requested_payments, pool_spend_bundle, driver_dict=self.driver_dict
        )
        logging.info(f"offer:{offer.arbitrage()}")
        logging.info(f"pool_offer:{pool_offer.arbitrage()}")
        agg_offer = Offer.aggregate([offer, pool_offer])
        logging.info(f"agg_offer:{agg_offer.arbitrage()}")
        return agg_offer.to_valid_spend()
