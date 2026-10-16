from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.compute as pc
except ImportError as exc:
    pa = None
    pc = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@dataclass
class ArrowRun:
    events: int
    batches: int
    elapsed_s: float
    events_per_s: float
    selected_rows: int
    checksum: float


def require_arrow() -> None:
    if pa is None:
        raise RuntimeError(
            "Apache Arrow is required. Install with "
            "pip install 'pyarrow>=25,<27'."
        ) from _IMPORT_ERROR


def dataframe_to_arrow(df):
    require_arrow()
    return pa.Table.from_pandas(df, preserve_index=False)


def q_where(table, expression):
    require_arrow()
    return table.filter(expression)


def q_select(table, columns: list[str]):
    require_arrow()
    return table.select(columns)


def q_by(table, keys: list[str], aggregations: list[tuple[str, str]]):
    require_arrow()
    return table.group_by(keys).aggregate(aggregations)


def q_within(table, column: str, lower: float, upper: float):
    require_arrow()
    if lower > upper:
        raise ValueError("lower must be less than or equal to upper")
    col = table[column]
    mask = pc.and_(
        pc.greater_equal(col, lower),
        pc.less_equal(col, upper),
    )
    return table.filter(mask)


def process_batches(batches: Iterable):
    """Process Arrow RecordBatches without collecting the full stream in memory."""
    require_arrow()
    start = time.perf_counter()
    events = 0
    selected = 0
    batch_count = 0
    checksum = 0.0

    for batch in batches:
        table = pa.Table.from_batches([batch])
        columns = set(table.column_names)
        missing = {"ts_ns", "bid_px_1", "ask_px_1"} - columns
        if missing:
            raise ValueError(
                "Arrow batch is missing required columns: "
                + ", ".join(sorted(missing))
            )

        bid_px = pc.cast(table["bid_px_1"], pa.float64())
        ask_px = pc.cast(table["ask_px_1"], pa.float64())

        # Raw L2 batches require best-level sizes. Already-featured batches may
        # supply queue_imbalance and skip this calculation.
        if "queue_imbalance" not in columns:
            missing_sizes = {"bid_sz_1", "ask_sz_1"} - columns
            if missing_sizes:
                raise ValueError(
                    "Arrow batch needs bid_sz_1 and ask_sz_1 when "
                    "queue_imbalance is not supplied"
                )
            bid_size = pc.cast(table["bid_sz_1"], pa.float64())
            ask_size = pc.cast(table["ask_sz_1"], pa.float64())
            denom = pc.add(bid_size, ask_size)
            denom = pc.max_element_wise(
                denom,
                pa.scalar(1e-12, type=pa.float64()),
            )
            qi = pc.divide(pc.subtract(bid_size, ask_size), denom)
            table = table.append_column("queue_imbalance", qi)

        if "mid" not in columns:
            mid = pc.multiply(pc.add(ask_px, bid_px), 0.5)
            table = table.append_column("mid", mid)
        else:
            mid = pc.cast(table["mid"], pa.float64())

        # Recompute spread from the prices and replace stale supplied values
        # rather than appending a duplicate column with the same name.
        spread = pc.divide(
            pc.multiply(pc.subtract(ask_px, bid_px), pa.scalar(1e4)),
            mid,
        )
        if "spread_bps" in table.column_names:
            table = table.drop(["spread_bps"])
        table = table.append_column("spread_bps", spread)

        table = q_where(
            table,
            pc.and_(
                pc.greater_equal(table["queue_imbalance"], -0.9),
                pc.less_equal(table["queue_imbalance"], 0.9),
            ),
        )
        table = q_select(
            table,
            ["ts_ns", "mid", "spread_bps", "queue_imbalance"],
        )

        batch_count += 1
        events += batch.num_rows
        selected += table.num_rows
        checksum += float(pc.sum(table["spread_bps"]).as_py() or 0.0)

    elapsed = time.perf_counter() - start
    return ArrowRun(
        events=events,
        batches=batch_count,
        elapsed_s=elapsed,
        events_per_s=events / elapsed if elapsed else 0.0,
        selected_rows=selected,
        checksum=checksum,
    )


def scan_parquet(
    path: str | Path,
    batch_size: int = 1_000_000,
    columns: list[str] | None = None,
):
    """Stream a Parquet dataset as Arrow RecordBatches."""
    require_arrow()
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    import pyarrow.dataset as ds

    dataset = ds.dataset(str(path), format="parquet")
    return dataset.scanner(columns=columns, batch_size=batch_size).to_batches()


def synthetic_arrow_stream(
    events: int,
    batch_size: int = 1_000_000,
    seed: int = 7,
):
    require_arrow()
    events = int(events)
    batch_size = int(batch_size)
    if events < 0:
        raise ValueError("events must be non-negative")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    rng = np.random.default_rng(seed)
    remaining = events
    timestamp = 1_700_000_000_000_000_000
    mid_level = 100.0

    while remaining:
        n = min(batch_size, remaining)
        mid = mid_level + np.cumsum(rng.normal(0.0, 2e-5, n))
        mid_level = float(mid[-1])
        spread = np.maximum(
            0.01,
            0.04 + 0.01 * rng.lognormal(0.0, 0.25, n),
        )
        bid = mid - spread / 2
        ask = mid + spread / 2
        bsz = rng.lognormal(2.2, 0.25, n)
        asz = rng.lognormal(2.2, 0.25, n)
        imbalance = (bsz - asz) / np.maximum(bsz + asz, 1e-8)

        yield pa.record_batch(
            {
                "ts_ns": pa.array(
                    np.arange(
                        timestamp,
                        timestamp + n * 1_000,
                        1_000,
                        dtype=np.int64,
                    )
                ),
                "mid": pa.array(mid),
                "bid_px_1": pa.array(bid),
                "ask_px_1": pa.array(ask),
                "queue_imbalance": pa.array(imbalance),
            }
        )
        timestamp += n * 1_000
        remaining -= n
