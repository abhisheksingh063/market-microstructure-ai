from decimal import Decimal

from agents.base import BaseAgent
from core.models import Order, OrderSide, Trade


def test_base_agent_initial_state():
    agent = _ConcreteAgent("test-agent", "TestAgent")
    assert agent.agent_id == "test-agent"
    assert agent.name == "TestAgent"
    assert agent.cash == 100_000.0
    assert agent.position == 0
    assert agent.total_trades == 0


def test_base_agent_on_trade():
    agent = _ConcreteAgent("a1", "Agent1")
    trade = Trade(
        buy_order_id="buy-1",
        sell_order_id="sell-1",
        price=Decimal("100"),
        quantity=10,
        buyer_id="a1",
        seller_id="a2",
    )
    order = Order(agent_id="a1", side=OrderSide.BUY, quantity=10)

    agent.on_trade(trade, order)
    assert agent.total_trades == 1
    assert agent.position == -10  # sign: buy => -1 * 10


def test_base_agent_reset():
    agent = _ConcreteAgent("a1", "Agent1", initial_cash=50_000)
    agent.cash = 10_000
    agent.position = 100
    agent.reset()
    assert agent.cash == 100_000.0  # Hardcoded default in base
    assert agent.position == 0
    assert agent.total_trades == 0


class _ConcreteAgent(BaseAgent):
    def generate_order(self, order_book, step):
        return None
