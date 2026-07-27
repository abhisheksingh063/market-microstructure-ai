from decimal import Decimal

from core.enums import SimulationStatus
from core.events import EventBus, EventType
from core.models import Order, OrderSide, OrderType
from simulation.orchestrator import SimulationOrchestrator, SimulationParameters


def test_initial_state():
    orch = SimulationOrchestrator()
    assert orch.status == SimulationStatus.PENDING
    assert orch.current_step == 0
    assert not orch.is_running


def test_configure():
    orch = SimulationOrchestrator()
    params = SimulationParameters(total_steps=100, name="test-sim")
    orch.configure(params)
    assert orch.params.name == "test-sim"
    assert orch.params.total_steps == 100


def test_start_stop():
    orch = SimulationOrchestrator()
    orch.configure(SimulationParameters(total_steps=1000, tick_duration_ms=1))
    orch.register_agent(_DummyAgent("dummy"))

    import threading
    t = threading.Thread(target=orch.start_sync)
    t.start()

    import time
    time.sleep(0.05)
    orch.stop()
    t.join(timeout=5)

    assert orch.current_step > 0
    assert not orch.is_running


def test_pause_resume():
    orch = SimulationOrchestrator()
    orch.configure(SimulationParameters(total_steps=2000, tick_duration_ms=1))
    orch.register_agent(_DummyAgent("dummy"))

    import threading
    t = threading.Thread(target=orch.start_sync)
    t.start()

    import time
    time.sleep(0.03)
    orch.pause()
    step_before = orch.current_step
    time.sleep(0.03)
    # Allow 1 step of slop for the race between pause() and tick boundary
    assert abs(orch.current_step - step_before) <= 1

    orch.resume()
    time.sleep(0.03)
    orch.stop()
    t.join(timeout=3)
    assert orch.current_step > step_before


def test_reset():
    orch = SimulationOrchestrator()
    orch.register_agent(_DummyAgent("dummy"))
    orch.configure(SimulationParameters(total_steps=100, tick_duration_ms=1))
    orch.start_sync()
    assert orch.current_step > 0

    orch.reset()
    assert orch.current_step == 0
    assert orch.status == SimulationStatus.PENDING


def test_events_emitted():
    bus = EventBus()
    orch = SimulationOrchestrator(event_bus=bus)
    orch.register_agent(_DummyAgent("dummy"))
    orch.configure(SimulationParameters(total_steps=10, tick_duration_ms=1))

    events = []
    for event_type in EventType:
        bus.subscribe(event_type, lambda e: events.append(e.type))

    orch.start_sync()

    assert EventType.SIMULATION_STARTED in events
    assert EventType.SIMULATION_COMPLETED in events
    assert EventType.SIMULATION_TICK in events


class _DummyAgent:
    """Minimal agent for testing the orchestrator."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.name = agent_id
        self.cash = 100_000
        self.position = 0
        self.total_trades = 0
        self.total_pnl = 0.0

    def generate_order(self, order_book, step):
        return Order(
            agent_id=self.agent_id,
            side=OrderSide.BUY if step % 2 == 0 else OrderSide.SELL,
            order_type=OrderType.LIMIT,
            price=Decimal("100"),
            quantity=10,
        )

    def on_trade(self, trade, order):
        self.total_trades += 1

    def reset(self):
        self.cash = 100_000
        self.position = 0
        self.total_trades = 0
        self.total_pnl = 0.0
