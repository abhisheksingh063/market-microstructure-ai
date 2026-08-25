"""Market analytics engine for calculating OHLCV candlesticks and price metrics.

Consumes PriceHistory and PriceObservation feeds without coupling
directly to MatchingEngine internals.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence, Union

from core.enums import TimeInterval
from core.exceptions import InvalidIntervalError
from core.models import Candle, PriceObservation
from core.price_history import PriceHistory

logger = logging.getLogger(__name__)


def parse_interval_seconds(interval: Union[TimeInterval, str, int, timedelta]) -> int:
    """Parse a time interval into integer seconds.

    Supports:
        - TimeInterval enum (e.g. TimeInterval.ONE_MINUTE)
        - Strings: "1m", "5m", "15m", "1h", "1min", "5min", "1hour", "60s"
        - Positive integers (seconds)
        - datetime.timedelta objects

    Raises:
        InvalidIntervalError if interval is non-positive or invalid format.
    """
    if isinstance(interval, TimeInterval):
        if interval == TimeInterval.ONE_MINUTE:
            return 60
        if interval == TimeInterval.FIVE_MINUTES:
            return 300
        if interval == TimeInterval.FIFTEEN_MINUTES:
            return 900
        if interval == TimeInterval.ONE_HOUR:
            return 3600
        interval = interval.value

    if isinstance(interval, timedelta):
        secs = int(interval.total_seconds())
        if secs <= 0:
            raise InvalidIntervalError("Interval duration must be positive")
        return secs

    if isinstance(interval, int):
        if interval <= 0:
            raise InvalidIntervalError("Interval seconds must be positive")
        return interval

    if not isinstance(interval, str):
        raise InvalidIntervalError(f"Unsupported interval type: {type(interval).__name__}")

    s = interval.strip().lower()
    if not s:
        raise InvalidIntervalError("Interval string cannot be empty")

    # Standard shorthand lookups
    lookups = {
        "1m": 60,
        "1min": 60,
        "1minute": 60,
        "5m": 300,
        "5min": 300,
        "5minutes": 300,
        "15m": 900,
        "15min": 900,
        "15minutes": 900,
        "1h": 3600,
        "1hr": 3600,
        "1hour": 3600,
        "60m": 3600,
    }
    if s in lookups:
        return lookups[s]

    try:
        if s.endswith("sec") or s.endswith("secs"):
            unit_len = 4 if s.endswith("secs") else 3
            val = int(s[:-unit_len].strip())
            mult = 1
        elif s.endswith("min") or s.endswith("mins"):
            unit_len = 4 if s.endswith("mins") else 3
            val = int(s[:-unit_len].strip())
            mult = 60
        elif s.endswith("hour") or s.endswith("hours"):
            unit_len = 5 if s.endswith("hours") else 4
            val = int(s[:-unit_len].strip())
            mult = 3600
        elif s.endswith("hr") or s.endswith("hrs"):
            unit_len = 3 if s.endswith("hrs") else 2
            val = int(s[:-unit_len].strip())
            mult = 3600
        elif s.endswith("s"):
            val = int(s[:-1].strip())
            mult = 1
        elif s.endswith("m"):
            val = int(s[:-1].strip())
            mult = 60
        elif s.endswith("h"):
            val = int(s[:-1].strip())
            mult = 3600
        else:
            val = int(s)
            mult = 1

        total_secs = val * mult
        if total_secs <= 0:
            raise InvalidIntervalError(f"Interval duration must be positive: '{interval}'")
        return total_secs
    except (ValueError, TypeError) as exc:
        raise InvalidIntervalError(f"Invalid interval format: '{interval}'") from exc


class MarketAnalytics:
    """Analytics engine for computing OHLCV candlesticks and price metrics.

    Can be used statically or instantiated with an attached PriceHistory instance.
    """

    def __init__(self, price_history: Optional[PriceHistory] = None) -> None:
        self.price_history = price_history

    @staticmethod
    def generate_candles(
        observations: Sequence[PriceObservation],
        interval: Union[TimeInterval, str, int, timedelta] = TimeInterval.ONE_MINUTE,
        simulation_id: Optional[int] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> list[Candle]:
        """Aggregate PriceObservation records into deterministic OHLCV candles.

        Interval boundary semantics: [start, end) — half-open intervals.
        Aligned to standard UTC clock boundaries.
        Empty intervals are omitted from the result (no invented prices).

        Args:
            observations: Sequence of trade price observations.
            interval: Time interval (e.g. TimeInterval.ONE_MINUTE, "5m", 300).
            simulation_id: Filter by simulation ID if specified.
            start_time: Filter observations with timestamp >= start_time.
            end_time: Filter observations with timestamp <= end_time.
            limit: Maximum number of candles to return.

        Returns:
            List of Candle objects with exact Decimal open/high/low/close and integer volume.
        """
        interval_seconds = parse_interval_seconds(interval)

        # Apply simulation and time bounds filters
        filtered: list[PriceObservation] = []
        for obs in observations:
            if simulation_id is not None and obs.simulation_id != simulation_id:
                continue
            if start_time is not None and obs.timestamp < start_time:
                continue
            if end_time is not None and obs.timestamp > end_time:
                continue
            filtered.append(obs)

        if not filtered:
            return []

        # Sort observations deterministically by (timestamp, trade_id)
        sorted_obs = sorted(filtered, key=lambda o: (o.timestamp, o.trade_id))

        # Bin observations into interval buckets [start, end)
        buckets: dict[tuple[datetime, datetime], list[PriceObservation]] = {}
        for obs in sorted_obs:
            ts = obs.timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            epoch_sec = int(ts.timestamp())
            interval_start_epoch = (epoch_sec // interval_seconds) * interval_seconds
            interval_start = datetime.fromtimestamp(interval_start_epoch, tz=timezone.utc)
            interval_end = interval_start + timedelta(seconds=interval_seconds)

            key = (interval_start, interval_end)
            if key not in buckets:
                buckets[key] = []
            buckets[key].append(obs)

        candles: list[Candle] = []
        # Sort buckets chronologically by start time
        for (i_start, i_end), bucket_obs in sorted(buckets.items(), key=lambda x: x[0][0]):
            sim_id = simulation_id if simulation_id is not None else bucket_obs[0].simulation_id
            c = Candle(
                simulation_id=sim_id,
                start_time=i_start,
                end_time=i_end,
                open=bucket_obs[0].price,
                high=max(o.price for o in bucket_obs),
                low=min(o.price for o in bucket_obs),
                close=bucket_obs[-1].price,
                volume=sum(o.quantity for o in bucket_obs),
                trade_count=len(bucket_obs),
            )
            candles.append(c)

        if limit is not None and limit >= 0:
            candles = candles[:limit]

        return candles

    @classmethod
    def get_ohlcv(
        cls,
        price_history: Union[PriceHistory, Sequence[PriceObservation]],
        interval: Union[TimeInterval, str, int, timedelta] = TimeInterval.ONE_MINUTE,
        simulation_id: Optional[int] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> list[Candle]:
        """Generate OHLCV candles directly from PriceHistory or observations."""
        if isinstance(price_history, PriceHistory) or hasattr(price_history, "get_history"):
            observations = price_history.get_history(
                simulation_id=simulation_id,
                start_time=start_time,
                end_time=end_time,
            )
        else:
            observations = price_history

        return cls.generate_candles(
            observations=observations,
            interval=interval,
            simulation_id=simulation_id,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )

    def get_candles(
        self,
        interval: Union[TimeInterval, str, int, timedelta] = TimeInterval.ONE_MINUTE,
        simulation_id: Optional[int] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> list[Candle]:
        """Generate candles using the attached PriceHistory instance."""
        if self.price_history is None:
            return []
        return self.get_ohlcv(
            price_history=self.price_history,
            interval=interval,
            simulation_id=simulation_id,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )


__all__ = [
    "TimeInterval",
    "parse_interval_seconds",
    "MarketAnalytics",
]
