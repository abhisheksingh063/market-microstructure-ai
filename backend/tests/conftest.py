import pytest


@pytest.fixture
def sample_order_book():
    from core.models import OrderBook
    return OrderBook()
