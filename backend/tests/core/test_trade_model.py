from decimal import Decimal

import pytest

from core.exceptions import InvalidTradeError
from core.models import Trade


def _trade(**overrides) -> Trade:
    kwargs = dict(
        buy_order_id="buy-1",
        sell_order_id="sell-1",
        price=Decimal("100.50"),
        quantity=50,
        buyer_id="a1",
        seller_id="a2",
    )
    kwargs.update(overrides)
    return Trade(**kwargs)


def test_trade_defaults():
    trade = _trade()
    assert trade.trade_id
    assert len(trade.trade_id) == 16
    assert trade.simulation_id is None
    assert trade.timestamp is not None


def test_trade_simulation_association():
    trade = _trade(simulation_id=7)
    assert trade.simulation_id == 7


def test_trade_immutable():
    trade = _trade()
    with pytest.raises(AttributeError):
        trade.quantity = 100


def test_trade_rejects_zero_quantity():
    with pytest.raises(InvalidTradeError):
        _trade(quantity=0)


def test_trade_rejects_negative_quantity():
    with pytest.raises(InvalidTradeError):
        _trade(quantity=-5)


def test_trade_rejects_out_of_range_price():
    with pytest.raises(InvalidTradeError):
        _trade(price=Decimal("0"))
    with pytest.raises(InvalidTradeError):
        _trade(price=Decimal("2_000_000"))


def test_trade_rejects_missing_buy_order():
    with pytest.raises(InvalidTradeError):
        _trade(buy_order_id="")


def test_trade_rejects_missing_sell_order():
    with pytest.raises(InvalidTradeError):
        _trade(sell_order_id="")


def test_trade_rejects_missing_buyer():
    with pytest.raises(InvalidTradeError):
        _trade(buyer_id="")


def test_trade_rejects_missing_seller():
    with pytest.raises(InvalidTradeError):
        _trade(seller_id="")
