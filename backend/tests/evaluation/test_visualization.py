"""Unit tests for Milestone 27 — Price Visualization.

Tests verify:
- Empty price history line chart
- Single and multiple price observations line chart
- Chronological ordering of unsorted inputs without mutating input
- OHLC/candlestick chart with empty, single, and multiple candles
- Volume-over-time bar chart with empty and populated data
- Buy vs sell volume breakdown from Trade sequences and MetricsCollector
- Combined 2-panel market overview chart
- Exact Decimal price handling
- Timezone-aware timestamp support
- Input immutability
- Deterministic chart contents
- Figure byte and base64 serialization
- Resource cleanup (closing figures)
"""

from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure

from core.analytics import MarketAnalytics
from core.models import Candle, PriceObservation, Trade
from evaluation.visualization import (
    figure_to_base64,
    figure_to_bytes,
    plot_buy_sell_volume,
    plot_candlestick,
    plot_market_overview,
    plot_price_history,
    plot_volume,
)
from simulation.metrics import MarketMetrics, MetricsCollector


@pytest.fixture(autouse=True)
def cleanup_figures():
    """Ensure all created matplotlib figures are closed after each test."""
    yield
    plt.close("all")


class TestPriceHistoryVisualization:
    def test_empty_price_history(self):
        fig, ax = plot_price_history([])
        assert isinstance(fig, Figure)
        assert ax.get_title() == "Price History"
        assert len(ax.lines) == 0  # No data lines drawn

    def test_single_price_observation(self):
        ts = datetime(2026, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        obs = [PriceObservation(timestamp=ts, price=Decimal("100.50"), quantity=10)]

        fig, ax = plot_price_history(obs)
        assert len(ax.lines) == 1
        x_data, y_data = ax.lines[0].get_data()
        assert len(x_data) == 1
        assert y_data[0] == 100.50

    def test_multiple_price_observations(self):
        base_time = datetime(2026, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        obs = [
            PriceObservation(
                timestamp=base_time + timedelta(seconds=i),
                price=Decimal(str(100 + i)),
                quantity=5,
            )
            for i in range(5)
        ]

        fig, ax = plot_price_history(obs)
        assert len(ax.lines) == 1
        _, y_data = ax.lines[0].get_data()
        assert list(y_data) == [100.0, 101.0, 102.0, 103.0, 104.0]

    def test_chronological_ordering_from_unsorted_input(self):
        t1 = datetime(2026, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 1, 1, 9, 30, 10, tzinfo=timezone.utc)
        t3 = datetime(2026, 1, 1, 9, 30, 20, tzinfo=timezone.utc)

        # Pass in reverse chronological order
        obs = [
            PriceObservation(timestamp=t3, price=Decimal("105.00"), quantity=10),
            PriceObservation(timestamp=t1, price=Decimal("95.00"), quantity=10),
            PriceObservation(timestamp=t2, price=Decimal("100.00"), quantity=10),
        ]

        fig, ax = plot_price_history(obs)
        _, y_data = ax.lines[0].get_data()
        # Must be plotted chronologically: 95.0, 100.0, 105.0
        assert list(y_data) == [95.0, 100.0, 105.0]

        # Verify input was not mutated
        assert obs[0].timestamp == t3
        assert obs[1].timestamp == t1
        assert obs[2].timestamp == t2


class TestCandlestickVisualization:
    def test_empty_candlestick_input(self):
        fig, ax = plot_candlestick([])
        assert isinstance(fig, Figure)
        assert ax.get_title() == "OHLCV Candlestick Chart"
        assert len(ax.patches) == 0

    def test_single_candle(self):
        t1 = datetime(2026, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 1, 1, 9, 31, 0, tzinfo=timezone.utc)
        candle = Candle(
            start_time=t1,
            end_time=t2,
            open=Decimal("100.00"),
            high=Decimal("105.00"),
            low=Decimal("98.00"),
            close=Decimal("103.00"),
            volume=50,
            trade_count=5,
        )

        fig, ax = plot_candlestick([candle])
        # 1 wick line + 1 candle body rectangle
        assert len(ax.lines) == 1
        assert len(ax.patches) == 1

    def test_multiple_bullish_and_bearish_candles(self):
        t = datetime(2026, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        candles = [
            # Bullish candle (close > open)
            Candle(
                start_time=t,
                end_time=t + timedelta(minutes=1),
                open=Decimal("100.00"),
                high=Decimal("105.00"),
                low=Decimal("99.00"),
                close=Decimal("104.00"),
                volume=100,
            ),
            # Bearish candle (close < open)
            Candle(
                start_time=t + timedelta(minutes=1),
                end_time=t + timedelta(minutes=2),
                open=Decimal("104.00"),
                high=Decimal("106.00"),
                low=Decimal("97.00"),
                close=Decimal("98.00"),
                volume=150,
            ),
        ]

        fig, ax = plot_candlestick(candles)
        assert len(ax.lines) == 2
        assert len(ax.patches) == 2

    def test_integration_with_aggregate_ohlcv(self):
        t = datetime(2026, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        obs = [
            PriceObservation(timestamp=t, price=Decimal("100"), quantity=10),
            PriceObservation(
                timestamp=t + timedelta(seconds=20), price=Decimal("105"), quantity=10
            ),
            PriceObservation(
                timestamp=t + timedelta(seconds=40), price=Decimal("98"), quantity=10
            ),
            PriceObservation(
                timestamp=t + timedelta(seconds=55), price=Decimal("102"), quantity=10
            ),
        ]
        candles = MarketAnalytics.generate_candles(obs, interval=60)
        assert len(candles) == 1

        fig, ax = plot_candlestick(candles)
        assert len(ax.patches) == 1


class TestVolumeVisualization:
    def test_empty_volume(self):
        fig, ax = plot_volume([])
        assert isinstance(fig, Figure)
        assert ax.get_title() == "Trading Volume Over Time"
        assert len(ax.patches) == 0

    def test_volume_from_trades_and_candles(self):
        t = datetime(2026, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        trades = [
            Trade(
                trade_id="t1",
                buy_order_id="b1",
                sell_order_id="s1",
                buyer_id="A",
                seller_id="B",
                price=Decimal("100"),
                quantity=15,
                timestamp=t,
            ),
            Trade(
                trade_id="t2",
                buy_order_id="b2",
                sell_order_id="s2",
                buyer_id="A",
                seller_id="C",
                price=Decimal("101"),
                quantity=25,
                timestamp=t + timedelta(seconds=1),
            ),
        ]

        fig, ax = plot_volume(trades)
        assert len(ax.patches) == 2
        heights = [p.get_height() for p in ax.patches]
        assert heights == [15, 25]


class TestBuySellVolumeVisualization:
    def test_buy_sell_volume_from_metrics_collector(self):
        collector = MetricsCollector()
        collector.record_trade(
            Trade(
                trade_id="t1",
                buy_order_id="b1",
                sell_order_id="s1",
                buyer_id="A",
                seller_id="B",
                price=Decimal("100"),
                quantity=30,
            )
        )

        fig, ax = plot_buy_sell_volume(collector)
        assert len(ax.patches) == 2
        heights = [p.get_height() for p in ax.patches]
        assert heights == [30, 30]

    def test_buy_sell_volume_from_market_metrics(self):
        metrics = MarketMetrics(total_buy_volume=100, total_sell_volume=100)
        fig, ax = plot_buy_sell_volume(metrics)
        assert len(ax.patches) == 2
        assert [p.get_height() for p in ax.patches] == [100, 100]

    def test_buy_sell_volume_from_trades(self):
        t = datetime(2026, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        trades = [
            Trade(
                trade_id="t1",
                buy_order_id="b1",
                sell_order_id="s1",
                buyer_id="A",
                seller_id="B",
                price=Decimal("100"),
                quantity=40,
                timestamp=t,
            )
        ]
        fig, ax = plot_buy_sell_volume(trades)
        assert len(ax.patches) == 2
        assert [p.get_height() for p in ax.patches] == [40, 40]


class TestMarketOverviewAndSerialization:
    def test_market_overview_two_panels(self):
        t = datetime(2026, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        obs = [
            PriceObservation(timestamp=t + timedelta(seconds=i), price=Decimal("100"), quantity=5)
            for i in range(3)
        ]

        fig, axes = plot_market_overview(price_data=obs)
        assert len(axes) == 2
        ax_price, ax_vol = axes
        assert ax_price.get_title() == "Market Overview"
        assert ax_vol.get_title() == "Trading Volume"
        assert len(ax_price.lines) == 1
        assert len(ax_vol.patches) == 3

    def test_figure_to_bytes_and_base64_serialization(self):
        fig, ax = plot_price_history([])
        raw_bytes = figure_to_bytes(fig, format="png")
        assert isinstance(raw_bytes, bytes)
        assert raw_bytes.startswith(b"\x89PNG\r\n\x1a\n")  # Standard PNG header

        b64_str = figure_to_base64(fig, format="png")
        assert isinstance(b64_str, str)
        decoded = base64.b64decode(b64_str)
        assert decoded == raw_bytes


class TestImmutabilityAndDeterminism:
    def test_input_immutability(self):
        t = datetime(2026, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        obs_original = [
            PriceObservation(timestamp=t + timedelta(seconds=10), price=Decimal("105"), quantity=5),
            PriceObservation(timestamp=t, price=Decimal("100"), quantity=10),
        ]
        obs_copy = list(obs_original)

        plot_price_history(obs_original)
        plot_volume(obs_original)

        assert obs_original == obs_copy
        assert obs_original[0].price == Decimal("105")
        assert obs_original[1].price == Decimal("100")

    def test_deterministic_chart_contents(self):
        t = datetime(2026, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        obs = [
            PriceObservation(timestamp=t, price=Decimal("100.25"), quantity=10),
            PriceObservation(
                timestamp=t + timedelta(seconds=5), price=Decimal("101.50"), quantity=20
            ),
        ]

        fig1, ax1 = plot_price_history(obs)
        fig2, ax2 = plot_price_history(obs)

        _, y1 = ax1.lines[0].get_data()
        _, y2 = ax2.lines[0].get_data()

        assert list(y1) == list(y2)
        assert ax1.get_title() == ax2.get_title()
