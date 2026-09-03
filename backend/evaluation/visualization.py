"""Reusable visualization and chart generation layer for Market Microstructure Simulator.

Produces standalone matplotlib figures for price histories, OHLCV candlesticks,
volume over time, and buy/sell volume breakdowns from simulation outputs without
mutating source financial data or coupling to web dashboards.
"""

from __future__ import annotations

import base64
import io
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from core.models import Candle, PriceObservation, Trade

if TYPE_CHECKING:
    from simulation.metrics import MarketMetrics, MetricsCollector


# ── Price History Line Chart ────────────────────────────────────


def plot_price_history(
    data: Sequence[
        Union[
            PriceObservation,
            Trade,
            dict[str, Any],
            tuple[datetime, Union[Decimal, float, int]],
        ]
    ],
    title: str = "Price History",
    ax: Optional[Axes] = None,
    figsize: tuple[float, float] = (10.0, 5.0),
) -> tuple[Figure, Axes]:
    """Plot price over time as a chronological line chart.

    Args:
        data: Sequence of PriceObservation, Trade, dicts, or (timestamp, price) tuples.
        title: Title of the chart.
        ax: Optional existing matplotlib Axes to draw upon.
        figsize: Figure dimensions if creating a new figure.

    Returns:
        A tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Extract and sort data without mutating input
    points: list[tuple[datetime, float]] = []
    for item in data:
        ts, price = _extract_timestamp_and_price(item)
        if ts is not None and price is not None:
            points.append((ts, float(price)))

    points.sort(key=lambda p: p[0])

    if not points:
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Price ($)")
        ax.text(
            0.5,
            0.5,
            "No price data available",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            color="gray",
            fontsize=12,
        )
        ax.grid(True, linestyle="--", alpha=0.5)
        return fig, ax

    timestamps, prices = zip(*points)
    ax.plot(
        timestamps,
        prices,
        color="#1f77b4",
        linewidth=1.8,
        marker="o" if len(points) <= 20 else None,
        markersize=4,
        label="Price",
    )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Time", fontsize=10)
    ax.set_ylabel("Price ($)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.autofmt_xdate()

    return fig, ax


# ── Candlestick / OHLCV Chart ───────────────────────────────────


def plot_candlestick(
    candles: Sequence[Union[Candle, dict[str, Any]]],
    title: str = "OHLCV Candlestick Chart",
    ax: Optional[Axes] = None,
    figsize: tuple[float, float] = (10.0, 6.0),
) -> tuple[Figure, Axes]:
    """Plot an OHLCV candlestick chart using native matplotlib shapes.

    Args:
        candles: Sequence of Candle domain models or dicts with OHLC fields.
        title: Title of the chart.
        ax: Optional existing matplotlib Axes.
        figsize: Figure dimensions if creating a new figure.

    Returns:
        A tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Extract and sort candles without mutating input
    parsed_candles: list[dict[str, Any]] = []
    for c in candles:
        parsed = _extract_candle_info(c)
        if parsed is not None:
            parsed_candles.append(parsed)

    parsed_candles.sort(key=lambda c: c["start_time"])

    if not parsed_candles:
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Price ($)")
        ax.text(
            0.5,
            0.5,
            "No candlestick data available",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            color="gray",
            fontsize=12,
        )
        ax.grid(True, linestyle="--", alpha=0.5)
        return fig, ax

    # Render each candlestick
    # Calculate appropriate width in days/fraction of day for matplotlib date axes
    if len(parsed_candles) > 1:
        time_deltas = [
            (
                parsed_candles[i]["start_time"]
                - parsed_candles[i - 1]["start_time"]
            ).total_seconds()
            for i in range(1, len(parsed_candles))
        ]
        avg_delta = sum(time_deltas) / len(time_deltas)
        width_days = (avg_delta * 0.7) / 86400.0
    else:
        width_days = 0.0005  # Default ~43 seconds for single candle

    for c in parsed_candles:
        dt_num = mdates.date2num(c["start_time"])
        o = float(c["open"])
        h = float(c["high"])
        low = float(c["low"])
        close = float(c["close"])

        is_bullish = close >= o
        body_color = "#2ca02c" if is_bullish else "#d62728"
        wick_color = "#2ca02c" if is_bullish else "#d62728"

        # High/low wick line
        ax.plot(
            [dt_num, dt_num],
            [low, h],
            color=wick_color,
            linewidth=1.2,
            zorder=2,
        )

        # Candle body
        body_bottom = min(o, close)
        body_height = max(abs(close - o), 0.001)  # Ensure visible height even for doji
        rect = Rectangle(
            (dt_num - width_days / 2, body_bottom),
            width_days,
            body_height,
            facecolor=body_color,
            edgecolor=wick_color,
            linewidth=1.0,
            zorder=3,
        )
        ax.add_patch(rect)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Time", fontsize=10)
    ax.set_ylabel("Price ($)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.autoscale_view()
    fig.autofmt_xdate()

    return fig, ax


# ── Volume Over Time ───────────────────────────────────────────


def plot_volume(
    data: Sequence[Union[Candle, Trade, PriceObservation, dict[str, Any]]],
    title: str = "Trading Volume Over Time",
    ax: Optional[Axes] = None,
    figsize: tuple[float, float] = (10.0, 4.0),
) -> tuple[Figure, Axes]:
    """Plot trading volume as a bar chart over time.

    Args:
        data: Sequence of Candle, Trade, PriceObservation, or dicts.
        title: Title of the chart.
        ax: Optional existing matplotlib Axes.
        figsize: Figure dimensions if creating a new figure.

    Returns:
        A tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Extract and sort volume items without mutating input
    items: list[tuple[datetime, int]] = []
    for el in data:
        ts, vol = _extract_timestamp_and_volume(el)
        if ts is not None and vol is not None:
            items.append((ts, vol))

    items.sort(key=lambda x: x[0])

    if not items:
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Volume")
        ax.text(
            0.5,
            0.5,
            "No volume data available",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            color="gray",
            fontsize=12,
        )
        ax.grid(True, linestyle="--", alpha=0.5)
        return fig, ax

    timestamps, volumes = zip(*items)
    dt_nums = [mdates.date2num(t) for t in timestamps]

    if len(dt_nums) > 1:
        deltas = [
            dt_nums[i] - dt_nums[i - 1]
            for i in range(1, len(dt_nums))
            if dt_nums[i] > dt_nums[i - 1]
        ]
        width = min(deltas) * 0.7 if deltas else 0.0005
    else:
        width = 0.0005

    ax.bar(
        dt_nums,
        volumes,
        width=width,
        color="#3498db",
        alpha=0.85,
        edgecolor="#2980b9",
        align="center",
    )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Time", fontsize=10)
    ax.set_ylabel("Volume", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.autoscale_view()
    fig.autofmt_xdate()

    return fig, ax


# ── Buy vs Sell Volume Breakdown ───────────────────────────────


def plot_buy_sell_volume(
    trades_or_metrics: Union[
        Sequence[Union[Trade, dict[str, Any]]],
        MetricsCollector,
        MarketMetrics,
    ],
    title: str = "Buy vs Sell Volume",
    ax: Optional[Axes] = None,
    figsize: tuple[float, float] = (8.0, 5.0),
) -> tuple[Figure, Axes]:
    """Plot a comparison of total buy volume vs sell volume.

    Args:
        trades_or_metrics: Sequence of Trade objects, or MetricsCollector/MarketMetrics instance.
        title: Title of the chart.
        ax: Optional existing matplotlib Axes.
        figsize: Figure dimensions.

    Returns:
        A tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    buy_vol = 0
    sell_vol = 0

    # Inspect input type
    if hasattr(trades_or_metrics, "get_market_metrics"):
        # MetricsCollector instance
        mm = trades_or_metrics.get_market_metrics()
        buy_vol = mm.total_buy_volume
        sell_vol = mm.total_sell_volume
    elif hasattr(trades_or_metrics, "total_buy_volume"):
        # MarketMetrics instance
        buy_vol = getattr(trades_or_metrics, "total_buy_volume", 0)
        sell_vol = getattr(trades_or_metrics, "total_sell_volume", 0)
    elif isinstance(trades_or_metrics, (list, tuple)):
        # Sequence of Trades
        for item in trades_or_metrics:
            if isinstance(item, Trade):
                buy_vol += item.quantity
                sell_vol += item.quantity
            elif isinstance(item, dict):
                qty = int(item.get("quantity", 0))
                buy_vol += qty
                sell_vol += qty

    categories = ["Buy Volume", "Sell Volume"]
    values = [buy_vol, sell_vol]
    colors = ["#2ca02c", "#d62728"]

    bars = ax.bar(categories, values, color=colors, alpha=0.85, width=0.5)

    # Add numeric labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:,}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("Volume", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    return fig, ax


# ── Combined Market Overview ────────────────────────────────────


def plot_market_overview(
    price_data: Sequence[Union[PriceObservation, Trade, dict[str, Any]]],
    candles: Optional[Sequence[Union[Candle, dict[str, Any]]]] = None,
    title: str = "Market Overview",
    figsize: tuple[float, float] = (12.0, 8.0),
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Create a 2-panel chart: upper panel with price/candles, lower panel with volume.

    Args:
        price_data: Price observations or trades.
        candles: Optional candlestick sequence to render instead of line chart.
        title: Main chart title.
        figsize: Figure dimensions.

    Returns:
        A tuple of (Figure, (ax_price, ax_volume)).
    """
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [2.5, 1]}
    )

    if candles and len(candles) > 0:
        plot_candlestick(candles=candles, title=title, ax=ax_top)
    else:
        plot_price_history(data=price_data, title=title, ax=ax_top)

    # Volume on lower panel
    volume_source: Sequence[Any] = candles if (candles and len(candles) > 0) else price_data
    plot_volume(data=volume_source, title="Trading Volume", ax=ax_bottom)

    fig.tight_layout()
    return fig, (ax_top, ax_bottom)


# ── Serialization Utilities ─────────────────────────────────────


def figure_to_bytes(fig: Figure, format: str = "png", dpi: int = 100) -> bytes:
    """Serialize a matplotlib Figure to image bytes.

    Args:
        fig: matplotlib Figure.
        format: Image format (e.g. 'png', 'svg', 'jpeg').
        dpi: Dots per inch resolution.

    Returns:
        Raw image bytes.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format=format, dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def figure_to_base64(fig: Figure, format: str = "png", dpi: int = 100) -> str:
    """Serialize a matplotlib Figure to a base64 encoded string.

    Args:
        fig: matplotlib Figure.
        format: Image format (e.g. 'png', 'svg').
        dpi: Dots per inch resolution.

    Returns:
        Base64-encoded image string.
    """
    raw_bytes = figure_to_bytes(fig, format=format, dpi=dpi)
    return base64.b64encode(raw_bytes).decode("utf-8")


# ── Internal Extraction Helpers ─────────────────────────────────


def _extract_timestamp_and_price(
    item: Any,
) -> tuple[Optional[datetime], Optional[Decimal]]:
    if isinstance(item, PriceObservation):
        return item.timestamp, item.price
    if isinstance(item, Trade):
        return item.timestamp, item.price
    if isinstance(item, (tuple, list)) and len(item) == 2:
        ts, p = item
        if isinstance(ts, datetime):
            return ts, Decimal(str(p))
    if isinstance(item, dict):
        ts = item.get("timestamp") or item.get("start_time")
        p = item.get("price") or item.get("close")
        if isinstance(ts, datetime) and p is not None:
            return ts, Decimal(str(p))
    return None, None


def _extract_candle_info(item: Any) -> Optional[dict[str, Any]]:
    if isinstance(item, Candle):
        return {
            "start_time": item.start_time,
            "open": item.open,
            "high": item.high,
            "low": item.low,
            "close": item.close,
            "volume": item.volume,
        }
    if isinstance(item, dict):
        ts = item.get("start_time") or item.get("timestamp")
        if isinstance(ts, datetime):
            return {
                "start_time": ts,
                "open": Decimal(str(item.get("open", 0))),
                "high": Decimal(str(item.get("high", 0))),
                "low": Decimal(str(item.get("low", 0))),
                "close": Decimal(str(item.get("close", 0))),
                "volume": int(item.get("volume", 0)),
            }
    return None


def _extract_timestamp_and_volume(item: Any) -> tuple[Optional[datetime], Optional[int]]:
    if isinstance(item, Candle):
        return item.start_time, item.volume
    if isinstance(item, (PriceObservation, Trade)):
        return item.timestamp, item.quantity
    if isinstance(item, dict):
        ts = item.get("timestamp") or item.get("start_time")
        vol = item.get("volume") or item.get("quantity")
        if isinstance(ts, datetime) and vol is not None:
            return ts, int(vol)
    return None, None


__all__ = [
    "plot_price_history",
    "plot_candlestick",
    "plot_volume",
    "plot_buy_sell_volume",
    "plot_market_overview",
    "figure_to_bytes",
    "figure_to_base64",
]

