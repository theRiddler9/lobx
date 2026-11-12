"""Market microstructure analytics: Stoikov micro-price, VPIN toxicity, and spread decomposition."""

from __future__ import annotations

from bisect import bisect_left
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

from lobx.types import BookSnapshot, Fill, Side


@dataclass(frozen=True, slots=True)
class VpinPoint:
    """A single VPIN toxicity calculation point."""

    ts: int
    vpin: float
    bucket_index: int


def compute_stoikov_microprice(
    snapshots: Sequence[BookSnapshot],
) -> list[tuple[int, float]]:
    """Compute Stoikov's analytical micro-price for each snapshot.

    Formula:
        P_micro = (Q_bid * P_ask + Q_ask * P_bid) / (Q_bid + Q_ask)

    When resting buy depth exceeds sell depth, the micro-price shifts closer to the ask,
    reflecting impending upward price pressure.
    """
    series: list[tuple[int, float]] = []
    for snap in snapshots:
        if not snap.bids or not snap.asks:
            continue
        p_bid, q_bid = snap.bids[0]
        p_ask, q_ask = snap.asks[0]
        total_q = q_bid + q_ask
        if total_q == 0:
            continue
        micro_px = (q_bid * p_ask + q_ask * p_bid) / total_q
        series.append((snap.ts, float(micro_px)))
    return series


def compute_vpin(
    fills: Sequence[Fill],
    bucket_volume: int = 200,
    window_buckets: int = 15,
) -> list[VpinPoint]:
    """Calculate Volume-Synchronized Probability of Toxicity (VPIN).

    Divides transaction volume into equal-sized volume buckets. For each bucket,
    computes absolute volume imbalance between buyer-initiated and seller-initiated
    trades. The rolling average over ``window_buckets`` produces the VPIN index in [0, 1].

    Reference:
        Easley, D., López de Prado, M. M., & O'Hara, M. (2012). Flow Toxicity and
        Liquidity in a High-Frequency World. The Review of Financial Studies.
    """
    if bucket_volume <= 0 or window_buckets <= 0:
        raise ValueError("bucket_volume and window_buckets must be strictly positive")

    if not fills:
        return []

    imbalances: list[tuple[int, float]] = []  # (bucket_end_ts, abs_imbalance)
    curr_buy = 0
    curr_sell = 0
    curr_vol = 0
    last_ts = fills[0].ts

    for f in fills:
        rem_fill_qty = f.qty
        last_ts = f.ts
        while rem_fill_qty > 0:
            needed = bucket_volume - curr_vol
            traded = min(rem_fill_qty, needed)

            if f.taker_side is Side.BUY:
                curr_buy += traded
            else:
                curr_sell += traded

            curr_vol += traded
            rem_fill_qty -= traded

            if curr_vol >= bucket_volume:
                bucket_imb = float(abs(curr_buy - curr_sell))
                imbalances.append((last_ts, bucket_imb))
                curr_buy = 0
                curr_sell = 0
                curr_vol = 0

    if len(imbalances) < window_buckets:
        return []

    # Rolling average of imbalances over window_buckets
    vpin_series: list[VpinPoint] = []
    window: deque[float] = deque(maxlen=window_buckets)
    running_sum = 0.0

    for idx, (ts, imb) in enumerate(imbalances):
        if len(window) == window_buckets:
            running_sum -= window[0]
        window.append(imb)
        running_sum += imb

        if len(window) == window_buckets:
            vpin_val = running_sum / (window_buckets * bucket_volume)
            vpin_series.append(VpinPoint(ts=ts, vpin=float(vpin_val), bucket_index=idx))

    return vpin_series


@dataclass(frozen=True, slots=True)
class ExecutionDecomposition:
    """Microstructural decomposition of an execution fill."""

    fill_ts: int
    fill_price: int
    qty: int
    arrival_mid: float
    future_mid: float
    effective_spread_bps: float
    realized_spread_bps: float
    price_impact_bps: float


def decompose_executions(
    fills: Sequence[Fill],
    snapshots: Sequence[BookSnapshot],
    horizon_ns: int = 5_000_000_000,
    opening_price: int = 10_000,
) -> list[ExecutionDecomposition]:
    """Decompose fills into effective spread, realized spread, and price impact.

    Formulas:
        Effective Half-Spread = side_sign * (fill_price - mid_arrival) / mid_arrival * 10,000 bps
        Price Impact = side_sign * (mid_future - mid_arrival) / mid_arrival * 10,000 bps
        Realized Spread = Effective Half-Spread - Price Impact
    """
    if not fills:
        return []

    snap_times = [s.ts for s in snapshots]
    snap_mids: list[float] = []
    for s in snapshots:
        bb = s.bids[0][0] if s.bids else None
        ba = s.asks[0][0] if s.asks else None
        if bb is not None and ba is not None:
            snap_mids.append((bb + ba) / 2.0)
        else:
            snap_mids.append(float(opening_price))

    results: list[ExecutionDecomposition] = []

    for f in fills:
        # Side sign: +1 for buyers (hurt if price paid > mid), -1 for sellers
        side_sign = 1.0 if f.taker_side is Side.BUY else -1.0

        # Find snapshot closest to fill_ts
        idx_now = bisect_left(snap_times, f.ts)
        idx_now = min(idx_now, len(snap_mids) - 1)
        mid_now = snap_mids[idx_now]

        # Find snapshot at fill_ts + horizon_ns
        target_future_ts = f.ts + horizon_ns
        idx_future = bisect_left(snap_times, target_future_ts)
        idx_future = min(idx_future, len(snap_mids) - 1)
        mid_future = snap_mids[idx_future]

        eff_spread = side_sign * (f.price - mid_now) / mid_now * 10_000.0
        impact = side_sign * (mid_future - mid_now) / mid_now * 10_000.0
        realized = eff_spread - impact

        results.append(
            ExecutionDecomposition(
                fill_ts=f.ts,
                fill_price=f.price,
                qty=f.qty,
                arrival_mid=mid_now,
                future_mid=mid_future,
                effective_spread_bps=float(eff_spread),
                realized_spread_bps=float(realized),
                price_impact_bps=float(impact),
            )
        )

    return results
