# refer: amm test 流程 .mov
class TestPykeV2(object):
    # _pay_to_conditions 是 p2singleton (也就是 pool 管理的 asset)
    # 如果管理了acs 錢包,  用acs 作為puzzle 的coin都可以管理, 透過這個func 來花掉
    def acs_pay_to_conditions(
        self,
        coin: Coin,
        conditions: list,
        tail_program: Program | None = None,
        tail_program_hash: bytes32 | None = None,
        tail_solution: Program | None = None,
        lineage_proof: LineageProof | None = None,
        extra_delta: int = 0,
    ) -> SpendBundle:
        puzzle = ACS_MOD
        solution = Program.to(conditions)

        if any(
            [
                (tail_program is not None) and (tail_solution is not None),
                tail_program_hash is not None,
            ]
        ):
            if tail_program_hash is None:
                assert tail_program is not None

                tail_program_hash = tail_program.get_tree_hash()

            if lineage_proof is None:
                lineage_proof = LineageProof()

            spendable_cat = SpendableCAT(
                coin=coin,
                limitations_program_hash=tail_program_hash,
                inner_puzzle=puzzle,
                inner_solution=Program.to(conditions),
                limitations_program_reveal=tail_program,
                limitations_solution=tail_solution,
                lineage_proof=lineage_proof,
                extra_delta=extra_delta,
            )
            spend_bundle = unsigned_spend_bundle_for_spendable_cats(
                CAT_MOD, spendable_cat_list=[spendable_cat]
            )

        else:
            coin_spend = CoinSpend(coin=coin, puzzle_reveal=puzzle, solution=solution)
            spend_bundle = SpendBundle(
                coin_spends=[coin_spend], aggregated_signature=G2Element()
            )

        return spend_bundle

    @pytest.mark.asyncio
    async def test_pool_simulator(self):
        try:
            sim = await SpendSim.create()
            sim_client = SimClient(sim)

            # ACS can be wrapped by CAT, 讓CAT變成大家都可以花
            funding_puzzle = ACS_MOD  # xch

            funding_puzzle_hash = funding_puzzle.get_tree_hash()

            tail_program = ACM_MOD
            tail_program_hash = tail_program.get_tree_hash()
            tail_solution = Program.to([])
            # cat 的inner puzzle 是ACS
            # 如果這個cat 的inner puzzle 是funding_puzzle, 這個是funding_puzzle 的user 就可以對這個cat 進行任意控制
            funding_cat_puzzle = construct_cat_puzzle(
                CAT_MOD,
                limitations_program_hash=tail_program_hash,
                inner_puzzle=funding_puzzle,
            )
            funding_cat_puzzle_hash = funding_cat_puzzle.get_tree_hash()

            ### genesis block

            await sim.farm_block()
            await sim.farm_block()

            ### prepare CAT

            cat_total_amount: int = 1_000_000 * 10**3

            await sim.farm_block(funding_puzzle_hash)

            # 第一代的cat 的parent (xch)
            funding_coin = (
                await sim_client.get_coin_records_by_puzzle_hash(
                    funding_puzzle_hash, include_spent_coins=False
                )
            )[0].coin
            funding_conditions = [
                Program.to(
                    [
                        ConditionOpcode.CREATE_COIN,
                        funding_cat_puzzle_hash,
                        cat_total_amount,
                    ]
                ),
            ]
            funding_spend_bundle = self.acs_pay_to_conditions(
                coin=funding_coin, conditions=funding_conditions
            )
            # funding_cat_coin:第一代的cat (不能自由花費的cat), 會去create 可以spend 的CAT
            funding_cat_coin = [
                coin
                for coin in funding_spend_bundle.additions()
                if coin.puzzle_hash == funding_cat_puzzle_hash
            ][0]
            funding_cat_conditions = [
                Program.to(
                    [
                        ConditionOpcode.CREATE_COIN,
                        funding_puzzle_hash,
                        cat_total_amount,
                    ]
                ),
                Program.to(
                    [
                        ConditionOpcode.CREATE_COIN,
                        None,
                        -113,
                        tail_program,
                        tail_solution,
                    ]
                ),
            ]
            funding_cat_spend_bundle = self.acs_pay_to_conditions(
                coin=funding_cat_coin,
                conditions=funding_cat_conditions,
                tail_program=tail_program,
                tail_solution=tail_solution,
            )

            spend_bundle = SpendBundle.aggregate(
                [funding_spend_bundle, funding_cat_spend_bundle]
            )

            await self.push_tx_and_farm(spend_bundle, sim_client, sim)
#----------------------------------------------------------------
# 先create好 要拿來create launcher 的用戶xch, 用這個xch 來create pool, 拿到這個pool 的condition(也就是"要創造launcher 需要達到什麼"),
# 拿這個condition 和要花掉的用戶xch 來create SB. 拿這個“花掉xch 創造launcher SB”和"花掉launcher 創造singleton 的SB" aggregated, 推上鏈

            ### mint pool (mint pool singleton)
            # launch singleton

            inner_puzzle_info = PoolPuzzleInfo.create(
                {
                    "asset_id_a": None,  # XCH
                    "asset_id_b": tail_program_hash,
                }
            )

            await sim.farm_block(funding_puzzle_hash)
#拿來create launcher coin 的用戶的xch
            funding_coin = (
                await sim_client.get_coin_records_by_puzzle_hash(
                    funding_puzzle_hash, include_spent_coins=False
                )
            )[0].coin
#不會看到1 mojo 的過程, 因為會大手筆的直接把xch coin 花掉
#從funding_coin做launcher coin 
            pool = Pool.create(
                genesis_coin=funding_coin,#製造launch coin的coin
                inner_puzzle_info=inner_puzzle_info,
                full_node_client=sim_client,
            )
            # 這裡的 conditions 指定要產生一個 condition 叫做 (CREATE_COIN LAUNCHER_PUZZLE_HASH LAUNCHER_AMOUNT)
            # 所以 pay_to_conditions 只要照做便可以產生 launcher
            conditions = pool.get_launch_conditions()#-> 要創造這個pool 的launcher 的條件是什麼呢？

            # 用戶試著pay to condition(create launcher puzzle hash的condition)
            # 本來需要用戶簽名, 但這邊是acs所以不用
            # 要創造launcher coin的條件(condition), 就是花掉funding_coin
            # 用戶透過acs_pay_to_conditions花掉以ACS為puzzle 的funding_coin
            funding_spend_bundle = self.acs_pay_to_conditions(
                coin=funding_coin, conditions=conditions
            )

            # 把launcher coin 花掉
            # 用戶已經對剛剛的condition 做支付, funding_spend_bundle 製造launcher, spend launcher 不需簽名
            # 把 "要花掉xch 來create launcher coin“ 和“要花掉launcher coin 來create singleton” 兩個SB 合併
            # 透過這個func, 只要有 launcher 的SB 就可以直接拿到singleton 的SB
            spend_bundle = await pool.get_launch_spend_bundle(funding_spend_bundle)

            #funding coin 被真的spend 掉了, 可以用coin reccord 發現多了spent_block_index: 
            # coin_record = await sim_client.get_coin_record_by_name(funding_coin.name())
            await self.push_tx_and_farm(spend_bundle, sim_client, sim)

            ### deposit first liquidity
            # 比例隨意設計
            xch_amount: int = int(0.1 * 10**12)  # 0.1 XCH
            cat_amount: int = int(100 * 10**3)  # 100 CAT
            liq_amount: int = int(100 * 10**3)

#-- prepare coin -----
            funding_xch_coin = (
                await sim_client.get_coin_records_by_puzzle_hash(
                    funding_puzzle_hash, include_spent_coins=False
                )
            )[0].coin
            funding_cat_coin = (
                await sim_client.get_coin_records_by_puzzle_hash(
                    funding_cat_puzzle_hash, include_spent_coins=False
                )
            )[0].coin
            funding_cat_parent_coin = (
                await sim_client.get_coin_record_by_name(
                    funding_cat_coin.parent_coin_info
                )
            ).coin
#---------------------

            # 用戶要求的payments
            # :請你付liq_amount到ACS_PH
            # 假設有三個coin 被用在offer 中, requested_payments:((N . ((PH1 AMT1 ...) (PH2 AMT2 ...) (PH3 AMT3 ...))) ...)
            requested_payments = {
                pool.liquidity_tail_program_hash: [Payment(ACS_PH, liq_amount, [])],
            }

            # requested_payments說: 送liq_amount到ACS_PH
            # 換取notarized_payments request 的資產
            # nonce: 用 被提供到offer 的coinId list的treehash 做的
            # notarized_payments: 支付證明,你支付的同時要把支付地址,額度,nonce 一起 hash 起來去做 
            # maker 要求的coin 會產生notarized coin payment, 像(PH1 AMT1 ...)
            # 把這些notarized coin payment 和nonce 合起來變notarized_payments
            # coins=[...]: 只是拿來做nonce
            notarized_payments = Offer.notarize_payments(
                requested_payments, coins=[funding_xch_coin, funding_cat_coin]
            )

            # 要被所有wallet create 的SB assert
            # The offer driver calculates the announcements that need to be asserted in order to get paid.
            announcements: list[Announcement] = Offer.calculate_announcements(
                notarized_payments, driver_dict=pool.driver_dict
            )

            # lp token 支付完後產生announcements
            announcement_conditions: list = [
                [ConditionOpcode.ASSERT_PUZZLE_ANNOUNCEMENT, announcement.name()]
                for announcement in announcements
            ]

            # OFFER_MOD_HASH = OFFER_MOD.get_tree_hash()，而 OFFER_MOD 就是 settlement_payments.clsp，第三方的 puzzle
            funding_xch_conditions = [
                [ConditionOpcode.CREATE_COIN, OFFER_MOD_HASH, xch_amount],
            ]
            # 要"+ announcement_conditions": 代表lp token 收到才能送出xch
            # 因為要花掉的是用戶的以ACS 當puzzle 的xch, 所以用acs_pay_to_conditions
            # 花掉user 的xch, 但要create 有xch_amount的xch,以及確保(assert) lp coin 有被傳到ACS_PH(參考requested_payments)
            funding_xch_spend_bundle = self.acs_pay_to_conditions(
                coin=funding_xch_coin,
                conditions=funding_xch_conditions + announcement_conditions,
            )

            #create兩個coin, 一個是要新增給pool的cat , 一個是退回給用戶的找錢
            funding_cat_conditions = [
                Program.to([ConditionOpcode.CREATE_COIN, OFFER_MOD_HASH, cat_amount]),
                Program.to(
                    [
                        ConditionOpcode.CREATE_COIN,
                        funding_puzzle_hash,
                        funding_cat_coin.amount - cat_amount,
                    ]
                ),
            ]
            funding_cat_spend_bundle = self.acs_pay_to_conditions(
                coin=funding_cat_coin,
                conditions=funding_cat_conditions + announcement_conditions,
                tail_program_hash=tail_program_hash,
                lineage_proof=LineageProof(
                    funding_cat_parent_coin.parent_coin_info,
                    funding_puzzle_hash,
                    funding_cat_parent_coin.amount,
                ),
            )

            # 願意支付XCH 和CAT 換取lp token. 願意支付funding_xch_spend_bundle + funding_cat_spend_bundle 換取notarized_payments
            # 這只是user(offer maker)的sb, 要找taker 去create這 
            funding_spend_bundle = SpendBundle.aggregate(
                [funding_xch_spend_bundle, funding_cat_spend_bundle]
            )

            # 包著taker SB的offer 
            funding_offer = Offer(
                notarized_payments, funding_spend_bundle, driver_dict=pool.driver_dict
            )
            # funding_spend_bundle vs spend_bundle

            # offer coin 被spend 後,會create 可以被pool 控制的asset coin
            # 所謂的"可以被pool 控制", 就是pool 會提供一個可以被pool 控制的p2 singleton
            # user 和pool 沒有支付上的誰先誰後順序,都是 "對方有支付才支付"
            # funding_offer: request lp ; pool_offer: request xch / cat
            spend_bundle = await pool.respond_to_offer(offer=funding_offer)

            await self.push_tx_and_farm(spend_bundle, sim_client, sim)
            await pool.sync()

            pool.print_summary()
#----------------------------------------------------------------
            ### do nothing

            spend_bundle = await pool.respond_to_offer()

            await self.push_tx_and_farm(spend_bundle, sim_client, sim)

            ### free XCH(捐xch)

            amount: int = int(0.001 * 10**12)  # 0.001 XCH

            await sim.farm_block(funding_puzzle_hash)

            funding_coin = (
                await sim_client.get_coin_records_by_puzzle_hash(
                    funding_puzzle_hash, include_spent_coins=False
                )
            )[0].coin

            funding_conditions = [
                Program.to([ConditionOpcode.CREATE_COIN, OFFER_MOD_HASH, amount]),
            ]
            funding_spend_bundle = self.acs_pay_to_conditions(
                coin=funding_coin, conditions=funding_conditions
            )
            funding_offer = Offer({}, funding_spend_bundle, driver_dict={})

            spend_bundle = await pool.respond_to_offer(offer=funding_offer)

            await self.push_tx_and_farm(spend_bundle, sim_client, sim)
            await pool.sync()

            pool.print_summary()

            ### free CAT

            amount: int = 1_000 * 10**3

            funding_cat_coin = (
                await sim_client.get_coin_records_by_puzzle_hash(
                    funding_cat_puzzle_hash, include_spent_coins=False
                )
            )[0].coin
            funding_cat_parent_coin = (
                await sim_client.get_coin_record_by_name(
                    funding_cat_coin.parent_coin_info
                )
            ).coin

            funding_conditions = [
                Program.to([ConditionOpcode.CREATE_COIN, OFFER_MOD_HASH, amount]),
                Program.to(
                    [
                        ConditionOpcode.CREATE_COIN,
                        funding_puzzle_hash,
                        funding_cat_coin.amount - amount,
                    ]
                ),
            ]
            funding_spend_bundle = self.acs_pay_to_conditions(
                coin=funding_cat_coin,
                conditions=funding_conditions,
                tail_program_hash=tail_program_hash,
                lineage_proof=LineageProof(
                    funding_cat_parent_coin.parent_coin_info,
                    funding_puzzle_hash,
                    funding_cat_parent_coin.amount,
                ),
            )
            funding_offer = Offer({}, funding_spend_bundle, driver_dict={})

            spend_bundle = await pool.respond_to_offer(offer=funding_offer)

            await self.push_tx_and_farm(spend_bundle, sim_client, sim)
            await pool.sync()

            pool.print_summary()

            ### swap token

            xch_amount: int = int(0.05 * 10**12)
            cat_amount: int = int(350 * 10**3)

            funding_coin = (
                await sim_client.get_coin_records_by_puzzle_hash(
                    funding_puzzle_hash, include_spent_coins=False
                )
            )[0].coin

            requested_payments = {
                tail_program_hash: [Payment(ACS_PH, cat_amount, [])],
            }
            notarized_payments = Offer.notarize_payments(
                requested_payments, coins=[funding_coin]
            )
            announcements: list[Announcement] = Offer.calculate_announcements(
                notarized_payments, driver_dict=pool.driver_dict
            )
            announcement_conditions: list = [
                [ConditionOpcode.ASSERT_PUZZLE_ANNOUNCEMENT, announcement.name()]
                for announcement in announcements
            ]

            funding_conditions = [
                [ConditionOpcode.CREATE_COIN, OFFER_MOD_HASH, xch_amount],
            ]
            funding_spend_bundle = self.acs_pay_to_conditions(
                coin=funding_coin,
                conditions=funding_conditions + announcement_conditions,
            )
            funding_offer = Offer(
                notarized_payments, funding_spend_bundle, driver_dict=pool.driver_dict
            )

            spend_bundle = await pool.respond_to_offer(offer=funding_offer)

            await self.push_tx_and_farm(spend_bundle, sim_client, sim)
            await pool.sync()

            pool.print_summary()

            ### withdraw liquidity

            xch_amount: int = int(0.015 * 10**12)
            cat_amount: int = int(75 * 10**3)
            liq_amount: int = int(10 * 10**3)

            funding_liq_puzzle = construct_cat_puzzle(
                CAT_MOD,
                limitations_program_hash=pool.liquidity_tail_program_hash,
                inner_puzzle=funding_puzzle,
            )
            funding_liq_puzzle_hash = funding_liq_puzzle.get_tree_hash()

            funding_liq_coin = (
                await sim_client.get_coin_records_by_puzzle_hash(
                    funding_liq_puzzle_hash, include_spent_coins=False
                )
            )[0].coin
            funding_liq_parent_coin = (
                await sim_client.get_coin_record_by_name(
                    funding_liq_coin.parent_coin_info
                )
            ).coin

            requested_payments = {
                None: [Payment(ACS_PH, xch_amount, [])],
                tail_program_hash: [Payment(ACS_PH, cat_amount, [])],
            }
            #???
            notarized_payments = Offer.notarize_payments(
                requested_payments, coins=funding_spend_bundle.removals()
            )
            announcements: list[Announcement] = Offer.calculate_announcements(
                notarized_payments, driver_dict=pool.driver_dict
            )
            announcement_conditions: list = [
                Program.to(
                    [ConditionOpcode.ASSERT_PUZZLE_ANNOUNCEMENT, announcement.name()]
                )
                for announcement in announcements
            ]

            funding_liq_conditions = [
                Program.to([ConditionOpcode.CREATE_COIN, OFFER_MOD_HASH, liq_amount]),
                Program.to(
                    [
                        ConditionOpcode.CREATE_COIN,
                        funding_puzzle_hash,
                        funding_liq_coin.amount - liq_amount,
                    ]
                ),
            ]
            funding_liq_spend_bundle = self.acs_pay_to_conditions(
                coin=funding_liq_coin,
                conditions=funding_liq_conditions + announcement_conditions,
                tail_program_hash=pool.liquidity_tail_program_hash,
                lineage_proof=LineageProof(
                    funding_liq_parent_coin.parent_coin_info,
                    OFFER_MOD_HASH,
                    funding_liq_parent_coin.amount,
                ),
            )

            funding_offer = Offer(
                notarized_payments,
                funding_liq_spend_bundle,
                driver_dict=pool.driver_dict,
            )
            spend_bundle = await pool.respond_to_offer(offer=funding_offer)

            await self.push_tx_and_farm(spend_bundle, sim_client, sim)
            await pool.sync()

            pool.print_summary()

        finally:
            await sim.close()
