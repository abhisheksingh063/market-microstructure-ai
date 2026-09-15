"""Reinforcement Learning environment for market microstructure execution.

This module provides a Gymnasium-compatible environment (ExecutionEnv) that wraps
the simulation and exchange engine (OrderBook, MatchingEngine, SimulationClock,
AgentScheduler, MetricsCollector, SimulationReplay) for training and evaluating
RL trading agents.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional, Sequence, Union

import gymnasium as gym
import numpy as np

from agents.base import BaseAgent
from core.events import EventBus
from core.models import Order, OrderBook, Trade
from core.price_history import PriceHistory
from matching.engine import MatchingEngine
from rl.actions import ActionConfig, ActionHandler
from rl.rewards import RewardBreakdown, RewardCalculator, RewardConfig
from rl.state import StateBuilder, StateConfig
from simulation.clock import DEFAULT_SIMULATION_START_TIME, DEFAULT_STEP_INTERVAL, SimulationClock
from simulation.metrics import MetricsCollector
from simulation.replay import SimulationReplay
from simulation.scheduler import AgentScheduler


class ExecutionAgent(BaseAgent):
    """Internal agent model representing the RL policy in the market exchange.

    Tracks portfolio state (cash, position, trades, PnL) using the standard
    BaseAgent accounting invariants as the single source of truth.
    """

    def __init__(
        self,
        agent_id: str = "rl_agent",
        name: str = "RL Execution Agent",
        initial_cash: float = 100_000.0,
        initial_inventory: int = 0,
    ) -> None:
        super().__init__(agent_id=agent_id, name=name, initial_cash=initial_cash)
        self.initial_cash = initial_cash
        self.initial_inventory = initial_inventory
        self.position = initial_inventory
        self.cash = initial_cash

    def generate_order(self, order_book: OrderBook, step: int = 0) -> Optional[Order]:
        """Unused by ExecutionAgent since orders are submitted via env.step(action)."""
        return None

    def on_trade(self, trade: Trade, order: Order) -> None:
        """Update portfolio state on trade execution using double-entry accounting."""
        self.total_trades += 1
        trade_val = float(trade.price * trade.quantity)
        if trade.buyer_id == self.agent_id:
            self.position += trade.quantity
            self.cash -= trade_val
        elif trade.seller_id == self.agent_id:
            self.position -= trade.quantity
            self.cash += trade_val

    def reset(self) -> None:
        """Reset agent portfolio back to initial state."""
        self.cash = self.initial_cash
        self.position = self.initial_inventory
        self.total_trades = 0
        self.total_pnl = 0.0


@dataclass
class ExecutionEnvConfig:
    """Configuration parameters for the ExecutionEnv Gymnasium environment."""

    max_steps: int = 100
    initial_cash: float = 100_000.0
    initial_inventory: int = 0
    target_inventory: int = 0
    order_quantity: int = 1
    step_interval: Union[timedelta, float, int] = DEFAULT_STEP_INTERVAL
    start_time: datetime = DEFAULT_SIMULATION_START_TIME
    symbol: str = "BTC-USDT"
    seed: Optional[int] = None
    background_agents: Optional[Sequence[BaseAgent]] = None
    render_modes: list[str] = field(default_factory=list)
    state_config: Optional[StateConfig] = None
    action_config: Optional[ActionConfig] = None
    reward_config: Optional[RewardConfig] = None
    dynamic_episode_seeds: bool = False


class ExecutionEnv(gym.Env):
    """Gymnasium environment wrapping market microstructure simulation.

    Action Space (M31 Action Space Design):
        Configurable via ActionConfig. Default is Discrete(5):
            0: Hold / No-op
            1: Market Buy
            2: Market Sell
            3: Limit Buy (at best bid / mid / default price)
            4: Limit Sell (at best ask / mid / default price)
        Also supports multi-level Discrete, Continuous Box, and MultiDiscrete spaces.

    Observation Space (M30 15-Feature Vector):
        Box(shape=(15,), dtype=np.float32) defined via StateBuilder.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: Optional[ExecutionEnvConfig] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # Merge config or explicit kwargs
        if config is not None:
            self.config = config
        else:
            self.config = ExecutionEnvConfig(
                max_steps=kwargs.get("max_steps", 100),
                initial_cash=kwargs.get("initial_cash", 100_000.0),
                initial_inventory=kwargs.get("initial_inventory", 0),
                target_inventory=kwargs.get("target_inventory", 0),
                order_quantity=kwargs.get("order_quantity", 1),
                step_interval=kwargs.get("step_interval", DEFAULT_STEP_INTERVAL),
                start_time=kwargs.get("start_time", DEFAULT_SIMULATION_START_TIME),
                symbol=kwargs.get("symbol", "BTC-USDT"),
                seed=kwargs.get("seed", None),
                background_agents=kwargs.get("background_agents", None),
                state_config=kwargs.get("state_config", None),
                action_config=kwargs.get("action_config", None),
                reward_config=kwargs.get("reward_config", None),
            )

        # Action Space configured via ActionHandler
        effective_action_config = self.config.action_config or ActionConfig(
            order_quantity=self.config.order_quantity
        )
        self.action_handler = ActionHandler(effective_action_config)
        self.action_space = self.action_handler.action_space

        # Observation Space constructed from StateBuilder
        self.state_builder = StateBuilder(self.config.state_config)
        self.observation_space = self.state_builder.observation_space

        # Reward Calculator
        self.reward_calculator = RewardCalculator(self.config.reward_config)

        self.current_step = 0
        self._episode_count = 0
        self.agent = ExecutionAgent(
            agent_id="rl_agent",
            name="RL Execution Agent",
            initial_cash=self.config.initial_cash,
            initial_inventory=self.config.initial_inventory,
        )

        self._init_simulation_components()

    def _init_simulation_components(self) -> None:
        """Instantiate clean simulation infrastructure."""
        self.event_bus = EventBus()
        self.order_book = OrderBook()
        self.matching_engine = MatchingEngine(
            order_book=self.order_book,
            event_bus=self.event_bus,
        )
        self.clock = SimulationClock(
            start_time=self.config.start_time,
            step_interval=self.config.step_interval,
        )
        self.price_history = PriceHistory(event_bus=self.event_bus)
        self.metrics = MetricsCollector(
            clock=self.clock,
            event_bus=self.event_bus,
        )
        self.replay = SimulationReplay(
            clock=self.clock,
            event_bus=self.event_bus,
        )
        self.scheduler = AgentScheduler(clock=self.clock)

        # Background agents
        self._background_agents: list[BaseAgent] = []
        if self.config.background_agents:
            self._background_agents = [copy.copy(a) for a in self.config.background_agents]
            for a in self._background_agents:
                self.scheduler.register(a, interval=self.config.step_interval)
    # ── Gymnasium Lifecycle ────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to a clean initial state.

        Returns:
            tuple of (initial_observation, info_dict).
        """
        super().reset(seed=seed)

        if seed is not None:
            effective_seed = seed
            self._episode_count = 0
        elif self.config.dynamic_episode_seeds and self.config.seed is not None:
            effective_seed = self.config.seed + self._episode_count
            self._episode_count += 1
        else:
            effective_seed = self.config.seed

        if effective_seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(effective_seed)

        # Re-initialize clean simulation state
        self._init_simulation_components()

        # Reset agent portfolio
        self.agent.reset()
        self.current_step = 0

        # Reset background agents if any
        for idx, a in enumerate(self._background_agents):
            agent_seed = effective_seed + idx if effective_seed is not None else None
            a.reset(seed=agent_seed)

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(
        self, action: Any
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one simulation step with the provided RL action.

        Returns:
            tuple of (observation, reward, terminated, truncated, info).
        """
        # Capture pre-action portfolio state
        prev_position = self.agent.position
        prev_cash = float(self.agent.cash)
        step_agent_trades: list[Trade] = []

        # 1. Translate Action to Domain Order and Execute
        order = self.action_handler.create_order(
            action=action,
            order_book=self.order_book,
            agent_id=self.agent.agent_id,
            timestamp=self.clock.now(),
        )
        if order is not None:
            trade_results = self.matching_engine.process_order(order)
            for tr in trade_results:
                self.agent.on_trade(tr.trade, order)
                step_agent_trades.append(tr.trade)

        # 2. Step Background Agents (if due)
        due_agent_ids = self.scheduler.get_due_agents(advance_schedule=True)
        for bg_agent in self._background_agents:
            if bg_agent.agent_id in due_agent_ids:
                if hasattr(bg_agent, "generate_orders"):
                    orders = bg_agent.generate_orders(self.order_book, self.current_step)
                else:
                    single_order = bg_agent.generate_order(self.order_book, self.current_step)
                    orders = [single_order] if single_order is not None else []

                for o in orders:
                    if o is not None:
                        o.timestamp = self.clock.now()
                        bg_trades = self.matching_engine.process_order(o)
                        for btr in bg_trades:
                            bg_agent.on_trade(btr.trade, o)
                            if (
                                btr.trade.buyer_id == self.agent.agent_id
                                or btr.trade.seller_id == self.agent.agent_id
                            ):
                                self.agent.on_trade(btr.trade, o)
                                step_agent_trades.append(btr.trade)

        # 3. Advance Simulation Clock & Step Counter
        self.clock.tick()
        self.current_step += 1

        # 4. Determine Termination and Truncation
        terminated = False
        truncated = self.current_step >= self.config.max_steps

        # 5. Resolve Benchmark Reference Price
        if self.order_book.mid_price is not None:
            ref_price = self.order_book.mid_price
        elif self.order_book.best_bid is not None:
            ref_price = self.order_book.best_bid
        elif self.order_book.best_ask is not None:
            ref_price = self.order_book.best_ask
        elif len(self.price_history.get_history()) > 0:
            ref_price = self.price_history.get_history()[-1].price
        else:
            ref_price = self.reward_calculator.config.default_price

        # 6. Calculate Reward
        reward_breakdown = self.reward_calculator.calculate_breakdown(
            previous_position=prev_position,
            current_position=self.agent.position,
            target_inventory=self.config.target_inventory,
            previous_cash=prev_cash,
            current_cash=float(self.agent.cash),
            trades=step_agent_trades,
            agent_id=self.agent.agent_id,
            reference_price=ref_price,
            terminated=terminated,
            truncated=truncated,
        )
        reward = reward_breakdown.total_reward

        # 7. Observation and Info
        obs = self._get_observation()
        info = self._get_info(reward_breakdown=reward_breakdown)

        return obs, reward, terminated, truncated, info

    # ── Observation Construction (Delegates to StateBuilder) ───

    def _get_observation(self) -> np.ndarray:
        """Construct the state observation vector via StateBuilder."""
        return self.state_builder.build(
            order_book=self.order_book,
            agent=self.agent,
            current_step=self.current_step,
            max_steps=self.config.max_steps,
            target_inventory=self.config.target_inventory,
            initial_cash=self.config.initial_cash,
            metrics=self.metrics,
            price_history=self.price_history,
        )

    def _get_info(
        self, reward_breakdown: Optional[RewardBreakdown] = None
    ) -> dict[str, Any]:
        """Construct diagnostic information dictionary."""
        mid = self.order_book.mid_price
        return {
            "step": self.current_step,
            "simulation_time": self.clock.now(),
            "cash": float(self.agent.cash),
            "position": self.agent.position,
            "total_trades": self.agent.total_trades,
            "mid_price": float(mid) if mid is not None else None,
            "best_bid": float(self.order_book.best_bid)
            if self.order_book.best_bid is not None
            else None,
            "best_ask": float(self.order_book.best_ask)
            if self.order_book.best_ask is not None
            else None,
            "reward": reward_breakdown.total_reward if reward_breakdown else 0.0,
            "execution_reward": (
                reward_breakdown.execution_reward if reward_breakdown else 0.0
            ),
            "inventory_progress_reward": (
                reward_breakdown.inventory_progress_reward if reward_breakdown else 0.0
            ),
            "inventory_penalty": (
                reward_breakdown.inventory_penalty if reward_breakdown else 0.0
            ),
            "terminal_penalty": (
                reward_breakdown.terminal_penalty if reward_breakdown else 0.0
            ),
            "executed_quantity": (
                reward_breakdown.executed_quantity if reward_breakdown else 0
            ),
            "target_inventory": self.config.target_inventory,
            "current_inventory": self.agent.position,
        }


# Alias for compatibility
MarketMicrostructureEnv = ExecutionEnv

__all__ = [
    "ExecutionEnv",
    "ExecutionEnvConfig",
    "ExecutionAgent",
    "MarketMicrostructureEnv",
]
