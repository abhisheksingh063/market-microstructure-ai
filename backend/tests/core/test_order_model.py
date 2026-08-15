"""Milestone 7 — Order domain model tests.

Verifies simulation/agent association, type-safe side and order type
representation, price/quantity validation, and fill invariants of the
core Order model.
"""

from decimal import Decimal

import pytest

from core.constants import MAX_ORDER_PRICE, MIN_ORDER_PRICE
from core.enums import OrderSide, OrderStatus, OrderType
from core.exceptions import InvalidOrderError
from core.models import Order


def test_simulation_id_defaults_to_none():
    order = Order()
    assert order.simulation_id is None


def test_simulation_id_association():
    order = Order(simulation_id=7, quantity=10)
    assert order.simulation_id == 7


def test_side_must_be_enum():
    with pytest.raises(InvalidOrderError):
        Order(side="buy", quantity=1)


def test_order_type_must_be_enum():
    with pytest.raises(InvalidOrderError):
        Order(order_type="limit", quantity=1)


def test_buy_and_sell_sides():
    assert Order(side=OrderSide.BUY, quantity=1).side is OrderSide.BUY
    assert Order(side=OrderSide.SELL, quantity=1).side is OrderSide.SELL


def test_market_and_limit_types():
    assert Order(order_type=OrderType.MARKET, quantity=1).order_type is OrderType.MARKET
    assert Order(order_type=OrderType.LIMIT, quantity=1).order_type is OrderType.LIMIT


def test_negative_quantity_rejected():
    with pytest.raises(InvalidOrderError):
        Order(quantity=-1)


def test_price_below_minimum_rejected():
    with pytest.raises(InvalidOrderError):
        Order(price=Decimal("0"), quantity=1)


def test_price_above_maximum_rejected():
    with pytest.raises(InvalidOrderError):
        Order(price=MAX_ORDER_PRICE + Decimal("1"), quantity=1)


def test_price_bounds_accepted():
    assert Order(price=MIN_ORDER_PRICE, quantity=1).price == MIN_ORDER_PRICE
    assert Order(price=MAX_ORDER_PRICE, quantity=1).price == MAX_ORDER_PRICE


def test_fill_exceeding_remaining_rejected():
    order = Order(quantity=10)
    with pytest.raises(InvalidOrderError):
        order.fill(11)
    assert order.remaining == 10


def test_fill_non_positive_quantity_rejected():
    order = Order(quantity=10)
    with pytest.raises(InvalidOrderError):
        order.fill(0)
    with pytest.raises(InvalidOrderError):
        order.fill(-5)


def test_fill_inactive_order_rejected():
    order = Order(quantity=10)
    order.fill(10)
    assert order.status is OrderStatus.FILLED
    with pytest.raises(InvalidOrderError):
        order.fill(1)


def test_fill_progression_partial_to_filled():
    order = Order(quantity=100)
    order.fill(40)
    assert order.filled_quantity == 40
    assert order.remaining == 60
    assert order.status is OrderStatus.PARTIAL
    order.fill(60)
    assert order.remaining == 0
    assert order.status is OrderStatus.FILLED
