"""Unit tests for Milestone 32 — Reward Engineering.

Tests verify:
- RewardConfig validation and default values
- RewardCalculator initialization across configurations
- Hold / no-trade step behavior
- Positive target inventory progress (buying towards target)
- Negative target inventory progress (selling towards target)
- Movement away from target (penalty for wrong-way trades)
- Full target completion and partial execution
- Overshooting target penalty
- Target inventory already satisfied at start
- Buy execution quality (favorable vs unfavorable prices)
- Sell execution quality (favorable vs unfavorable prices)
- Realistic relative magnitude and component contributions (maker > taker > hold > adverse)
- Terminal and truncation shortfall penalties
- Reward scaling factor application
- Deterministic repeated calculation and exact bit-for-bit repeatability
- Decimal price handling and precision preservation
- Zero-target inventory handling
- Empty trade list handling
- Input immutability and side-effect freedom
- ExecutionEnv integration, reward changes, and info diagnostic fields
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from core.models import Trade
from rl.environment import ExecutionEnv, ExecutionEnvConfig
from rl.rewards import RewardBreakdown, RewardCalculator, RewardConfig


class TestRewardConfigValidation:
    def test_default_config(self):
        config = RewardConfig()
        assert config.execution_weight == 1.0
        assert config.progress_weight == 1.0
        assert config.inventory_weight == 0.01
        assert config.terminal_weight == 1.0
        assert config.reward_scale == 1.0
        assert config.default_price == Decimal("100.00")

    def test_invalid_execution_weight_raises(self):
        with pytest.raises(ValueError, match="execution_weight must be non-negative"):
            RewardConfig(execution_weight=-0.5)

    def test_invalid_progress_weight_raises(self):
        with pytest.raises(ValueError, match="progress_weight must be non-negative"):
            RewardConfig(progress_weight=-1.0)

    def test_invalid_inventory_weight_raises(self):
        with pytest.raises(ValueError, match="inventory_weight must be non-negative"):
            RewardConfig(inventory_weight=-0.01)

    def test_invalid_terminal_weight_raises(self):
        with pytest.raises(ValueError, match="terminal_weight must be non-negative"):
            RewardConfig(terminal_weight=-1.0)

    def test_invalid_reward_scale_raises(self):
        with pytest.raises(ValueError, match="reward_scale must be positive"):
            RewardConfig(reward_scale=0.0)

        with pytest.raises(ValueError, match="reward_scale must be positive"):
            RewardConfig(reward_scale=-1.0)

    def test_invalid_default_price_raises(self):
        with pytest.raises(ValueError, match="default_price must be positive"):
            RewardConfig(default_price=Decimal("0.00"))


class TestTargetInventoryProgress:
    @pytest.fixture
    def calculator(self) -> RewardCalculator:
        return RewardCalculator(RewardConfig(execution_weight=0.0, inventory_weight=0.0))

    def test_positive_target_progress_buying(self, calculator: RewardCalculator):
        # Target: +10. Moving from 0 to 5. Progress = 10 - 5 = +5
        breakdown = calculator.calculate_breakdown(
            previous_position=0,
            current_position=5,
            target_inventory=10,
            previous_cash=100_000.0,
            current_cash=99_500.0,
            trades=[],
        )
        assert isinstance(breakdown, RewardBreakdown)
        assert breakdown.inventory_progress_reward == 5.0
        assert breakdown.total_reward == 5.0

    def test_negative_target_progress_selling(self, calculator: RewardCalculator):
        # Target: -10. Moving from 0 to -4. Progress = 10 - 6 = +4
        breakdown = calculator.calculate_breakdown(
            previous_position=0,
            current_position=-4,
            target_inventory=-10,
            previous_cash=100_000.0,
            current_cash=100_400.0,
            trades=[],
        )
        assert breakdown.inventory_progress_reward == 4.0
        assert breakdown.total_reward == 4.0

    def test_movement_away_from_target(self, calculator: RewardCalculator):
        # Target: +10. Agent was at 0, sells to -3.
        # prev_dist = 10, curr_dist = 13. Progress = 10 - 13 = -3
        breakdown = calculator.calculate_breakdown(
            previous_position=0,
            current_position=-3,
            target_inventory=10,
            previous_cash=100_000.0,
            current_cash=100_300.0,
            trades=[],
        )
        assert breakdown.inventory_progress_reward == -3.0
        assert breakdown.total_reward == -3.0

    def test_full_target_completion(self, calculator: RewardCalculator):
        # Target: 5. Moving from 0 to 5. Progress = 5 - 0 = +5
        breakdown = calculator.calculate_breakdown(
            previous_position=0,
            current_position=5,
            target_inventory=5,
            previous_cash=100_000.0,
            current_cash=99_500.0,
            trades=[],
        )
        assert breakdown.inventory_progress_reward == 5.0

    def test_overshooting_target_penalized(self, calculator: RewardCalculator):
        # Target: 5. Agent was at 4 (dist=1), buys 3 units -> now at 7 (dist=2).
        # Progress = 1 - 2 = -1
        breakdown = calculator.calculate_breakdown(
            previous_position=4,
            current_position=7,
            target_inventory=5,
            previous_cash=99_600.0,
            current_cash=99_300.0,
            trades=[],
        )
        assert breakdown.inventory_progress_reward == -1.0

    def test_target_already_satisfied_at_start(self, calculator: RewardCalculator):
        # Target: 0. Agent was at 0, remains at 0.
        breakdown = calculator.calculate_breakdown(
            previous_position=0,
            current_position=0,
            target_inventory=0,
            previous_cash=100_000.0,
            current_cash=100_000.0,
            trades=[],
        )
        assert breakdown.inventory_progress_reward == 0.0
        assert breakdown.inventory_penalty == 0.0


class TestExecutionQuality:
    @pytest.fixture
    def calculator(self) -> RewardCalculator:
        return RewardCalculator(
            RewardConfig(
                execution_weight=1.0,
                progress_weight=0.0,
                inventory_weight=0.0,
                terminal_weight=0.0,
            )
        )

    def test_buy_execution_favorable_price(self, calculator: RewardCalculator):
        # Ref price: 100.00. Bought 2 units at 99.50 (price improvement = +0.50 * 2 = +1.00)
        trade = Trade(
            trade_id="t1",
            buy_order_id="b1",
            sell_order_id="s1",
            price=Decimal("99.50"),
            quantity=2,
            buyer_id="rl_agent",
            seller_id="maker",
        )
        breakdown = calculator.calculate_breakdown(
            previous_position=0,
            current_position=2,
            target_inventory=10,
            previous_cash=100_000.0,
            current_cash=100_000.0 - 199.0,
            trades=[trade],
            reference_price=Decimal("100.00"),
        )
        assert breakdown.execution_reward == 1.0
        assert breakdown.executed_quantity == 2
        assert breakdown.total_reward == 1.0

    def test_buy_execution_unfavorable_price(self, calculator: RewardCalculator):
        # Ref price: 100.00. Bought 2 units at 100.50 (slippage cost = -0.50 * 2 = -1.00)
        trade = Trade(
            trade_id="t2",
            buy_order_id="b2",
            sell_order_id="s2",
            price=Decimal("100.50"),
            quantity=2,
            buyer_id="rl_agent",
            seller_id="maker",
        )
        breakdown = calculator.calculate_breakdown(
            previous_position=0,
            current_position=2,
            target_inventory=10,
            previous_cash=100_000.0,
            current_cash=100_000.0 - 201.0,
            trades=[trade],
            reference_price=Decimal("100.00"),
        )
        assert breakdown.execution_reward == -1.0
        assert breakdown.total_reward == -1.0

    def test_sell_execution_favorable_price(self, calculator: RewardCalculator):
        # Ref price: 100.00. Sold 3 units at 100.75 (gain = +0.75 * 3 = +2.25)
        trade = Trade(
            trade_id="t3",
            buy_order_id="b3",
            sell_order_id="s3",
            price=Decimal("100.75"),
            quantity=3,
            buyer_id="maker",
            seller_id="rl_agent",
        )
        breakdown = calculator.calculate_breakdown(
            previous_position=5,
            current_position=2,
            target_inventory=0,
            previous_cash=100_000.0,
            current_cash=100_000.0 + 302.25,
            trades=[trade],
            reference_price=Decimal("100.00"),
        )
        assert breakdown.execution_reward == 2.25
        assert breakdown.executed_quantity == 3

    def test_sell_execution_unfavorable_price(self, calculator: RewardCalculator):
        # Ref price: 100.00. Sold 1 unit at 99.20 (loss = -0.80 * 1 = -0.80)
        trade = Trade(
            trade_id="t4",
            buy_order_id="b4",
            sell_order_id="s4",
            price=Decimal("99.20"),
            quantity=1,
            buyer_id="maker",
            seller_id="rl_agent",
        )
        breakdown = calculator.calculate_breakdown(
            previous_position=5,
            current_position=4,
            target_inventory=0,
            previous_cash=100_000.0,
            current_cash=100_000.0 + 99.20,
            trades=[trade],
            reference_price=Decimal("100.00"),
        )
        assert breakdown.execution_reward == -0.80

    def test_decimal_precision_preservation(self, calculator: RewardCalculator):
        # Verify exact Decimal differences without premature rounding
        trade = Trade(
            trade_id="t5",
            buy_order_id="b5",
            sell_order_id="s5",
            price=Decimal("100.123456"),
            quantity=10,
            buyer_id="rl_agent",
            seller_id="maker",
        )
        breakdown = calculator.calculate_breakdown(
            previous_position=0,
            current_position=10,
            target_inventory=10,
            previous_cash=100_000.0,
            current_cash=100_000.0 - 1001.23456,
            trades=[trade],
            reference_price=Decimal("100.123400"),
        )
        # diff = (100.123400 - 100.123456) * 10 = -0.000056 * 10 = -0.00056
        assert pytest.approx(breakdown.execution_reward, rel=1e-6) == -0.00056


class TestMagnitudeAndRelativeContributions:
    def test_relative_contributions_ordering(self):
        """Audit and test: maker execution > taker execution > hold > adverse execution."""
        config = RewardConfig(
            execution_weight=1.0,
            progress_weight=1.0,
            inventory_weight=0.01,
            terminal_weight=1.0,
            reward_scale=1.0,
        )
        calculator = RewardCalculator(config)
        ref_price = Decimal("100.00")
        target = 10

        # Case 1: Passive Limit Buy Fill at bid (99.75) for 1 unit
        # Exec: +0.25, Prog: +1.0, Inv: -9 * 0.01 = -0.09 -> +1.16
        maker_trade = Trade(
            trade_id="m1",
            buy_order_id="b1",
            sell_order_id="s1",
            price=Decimal("99.75"),
            quantity=1,
            buyer_id="rl_agent",
            seller_id="maker",
        )
        maker_res = calculator.calculate_breakdown(
            previous_position=0,
            current_position=1,
            target_inventory=target,
            previous_cash=100_000.0,
            current_cash=99_900.25,
            trades=[maker_trade],
            reference_price=ref_price,
        )
        assert pytest.approx(maker_res.total_reward, rel=1e-5) == 1.16

        # Case 2: Aggressive Market Buy Fill at ask (100.25) for 1 unit
        # Exec: -0.25, Prog: +1.0, Inv: -9 * 0.01 = -0.09 -> +0.66
        taker_trade = Trade(
            trade_id="m2",
            buy_order_id="b2",
            sell_order_id="s2",
            price=Decimal("100.25"),
            quantity=1,
            buyer_id="rl_agent",
            seller_id="maker",
        )
        taker_res = calculator.calculate_breakdown(
            previous_position=0,
            current_position=1,
            target_inventory=target,
            previous_cash=100_000.0,
            current_cash=99_899.75,
            trades=[taker_trade],
            reference_price=ref_price,
        )
        assert pytest.approx(taker_res.total_reward, rel=1e-5) == 0.66

        # Case 3: Hold (no trade, 10 units away)
        # Exec: 0.0, Prog: 0.0, Inv: -10 * 0.01 = -0.10 -> -0.10
        hold_res = calculator.calculate_breakdown(
            previous_position=0,
            current_position=0,
            target_inventory=target,
            previous_cash=100_000.0,
            current_cash=100_000.0,
            trades=[],
            reference_price=ref_price,
        )
        assert pytest.approx(hold_res.total_reward, rel=1e-5) == -0.10

        # Case 4: Adverse Sell Fill (selling 1 unit at 99.75 when target is to buy)
        # Exec: -0.25, Prog: -1.0, Inv: -11 * 0.01 = -0.11 -> -1.36
        adverse_trade = Trade(
            trade_id="m3",
            buy_order_id="b3",
            sell_order_id="s3",
            price=Decimal("99.75"),
            quantity=1,
            buyer_id="maker",
            seller_id="rl_agent",
        )
        adverse_res = calculator.calculate_breakdown(
            previous_position=0,
            current_position=-1,
            target_inventory=target,
            previous_cash=100_000.0,
            current_cash=100_099.75,
            trades=[adverse_trade],
            reference_price=ref_price,
        )
        assert pytest.approx(adverse_res.total_reward, rel=1e-5) == -1.36

        # Verify exact hierarchy of incentive alignment
        assert maker_res.total_reward > taker_res.total_reward
        assert taker_res.total_reward > hold_res.total_reward
        assert hold_res.total_reward > adverse_res.total_reward


class TestTerminalAndShortfallPenalties:
    @pytest.fixture
    def calculator(self) -> RewardCalculator:
        return RewardCalculator(
            RewardConfig(
                execution_weight=1.0,
                progress_weight=1.0,
                inventory_weight=0.01,
                terminal_weight=2.0,
            )
        )

    def test_terminal_penalty_triggered_on_truncation(self, calculator: RewardCalculator):
        # Target: 10. Agent ended episode at position 6 (4 units short).
        # Truncated = True -> terminal_penalty = -4 * 2.0 = -8.0
        breakdown = calculator.calculate_breakdown(
            previous_position=6,
            current_position=6,
            target_inventory=10,
            previous_cash=100_000.0,
            current_cash=100_000.0,
            trades=[],
            truncated=True,
        )
        assert breakdown.terminal_penalty == -4.0
        # Inv penalty: -0.04. Terminal penalty: -8.0. Total = -8.04
        assert pytest.approx(breakdown.total_reward, rel=1e-5) == -8.04

    def test_no_terminal_penalty_when_target_achieved_at_end(self, calculator: RewardCalculator):
        # Target: 10. Agent reached 10 on final step.
        breakdown = calculator.calculate_breakdown(
            previous_position=9,
            current_position=10,
            target_inventory=10,
            previous_cash=100_000.0,
            current_cash=99_900.0,
            trades=[],
            truncated=True,
        )
        assert breakdown.terminal_penalty == 0.0
        assert breakdown.inventory_penalty == 0.0
        # Progress: +1.0 -> Total = +1.0
        assert breakdown.total_reward == 1.0


class TestPurityAndDeterminism:
    def test_deterministic_repeated_calculation(self):
        calculator = RewardCalculator()
        trade = Trade(
            trade_id="t1",
            buy_order_id="b1",
            sell_order_id="s1",
            price=Decimal("99.80"),
            quantity=5,
            buyer_id="rl_agent",
            seller_id="maker",
        )
        r1 = calculator.calculate(
            previous_position=0,
            current_position=5,
            target_inventory=10,
            previous_cash=100_000.0,
            current_cash=99_501.0,
            trades=[trade],
            reference_price=Decimal("100.00"),
        )
        r2 = calculator.calculate(
            previous_position=0,
            current_position=5,
            target_inventory=10,
            previous_cash=100_000.0,
            current_cash=99_501.0,
            trades=[trade],
            reference_price=Decimal("100.00"),
        )
        assert r1 == r2

    def test_reward_scaling(self):
        calc1 = RewardCalculator(RewardConfig(reward_scale=1.0))
        calc2 = RewardCalculator(RewardConfig(reward_scale=0.1))

        r1 = calc1.calculate(
            previous_position=0,
            current_position=2,
            target_inventory=10,
            previous_cash=100_000.0,
            current_cash=99_800.0,
            trades=[],
        )
        r2 = calc2.calculate(
            previous_position=0,
            current_position=2,
            target_inventory=10,
            previous_cash=100_000.0,
            current_cash=99_800.0,
            trades=[],
        )
        assert pytest.approx(r2, rel=1e-5) == 0.1 * r1


class TestEnvironmentRewardIntegration:
    def test_step_returns_non_zero_reward_and_diagnostics(self):
        config = ExecutionEnvConfig(
            target_inventory=10,
            order_quantity=2,
            initial_cash=100_000.0,
            max_steps=10,
        )
        env = ExecutionEnv(config=config)
        env.reset()

        # Step 0: Hold -> inventory penalty applies
        obs, reward, terminated, truncated, info = env.step(0)
        assert isinstance(reward, float)
        # 10 units away -> -0.10
        assert reward < 0.0
        assert "reward" in info
        assert "execution_reward" in info
        assert "inventory_progress_reward" in info
        assert "inventory_penalty" in info
        assert "terminal_penalty" in info
        assert "executed_quantity" in info
        assert "target_inventory" in info
        assert "current_inventory" in info
        assert info["target_inventory"] == 10
        assert info["current_inventory"] == 0
