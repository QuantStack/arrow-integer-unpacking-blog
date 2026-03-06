import pathlib
import platform
import re

import polars as pl
import pyarrow.parquet as pq


def count_header_lines(path: pathlib.Path) -> int:
    """Return the number of lines to skip before the CSV header row.

    Reads the file lazily, looking for the first line that matches a
    CSV-style header pattern: comma-separated words (e.g. "name,iterations,real_time").
    Returns the 0-based line index, i.e. the number of preceding lines to skip.
    """
    header_re = re.compile(r"^(\w+,)+\w+\s*$")
    with path.open() as f:
        for i, line in enumerate(f):
            if header_re.match(line):
                return i
    raise ValueError(f"No CSV header found in {path}")


def clean_benchmark(df: pl.LazyFrame) -> pl.LazyFrame:
    return (
        df
        # Remove pre-aggregated data generating heterogenous rows
        .filter(~pl.col("name").str.contains(r"_(mean|median|stddev|cv)$"))
        # Remove skipped benchmarks
        .filter(~pl.col("cpu_time").is_null())
        # Unused / all nulls
        .drop(
            "label",
            "error_occurred",
            "error_message",
            "bytes_per_second",
            "items_per_second",
        )
        # Parse benchmark name into idividual components
        .with_columns(
            pl.col("name")
            .str.extract_groups(
                r"BM_Unpack(?P<unpacked_type>\w+)/"
                r"(?P<arch>\w+)-(?P<func>\w+)/"
                r"(?P<packed_width>\d+)/(?P<num_values>\d+)"
            )
            .struct.unnest()
        )
        # Extract unpacked size from width, assuming 8 bits bool
        .with_columns(
            pl.when(pl.col("unpacked_type").str.to_lowercase() == "bool")
            .then(pl.lit("Uint8"))
            .otherwise(pl.col("unpacked_type"))
            .str.to_lowercase()
            .str.strip_prefix("uint")
            .cast(pl.Int32)
            .alias("unpacked_width"),
        )
        # Cast parsed integer components
        .with_columns(
            pl.col("num_values").cast(pl.Int32), pl.col("packed_width").cast(pl.Int32)
        )
        # Remove mangled name column
        .drop("name")
        # Cast remaining strings to categorical
        .with_columns(pl.col(pl.String).cast(pl.Categorical))
    )


def is_emscripten() -> bool:
    return "emscripten" in platform.system().lower()


def read_parquet(path: pathlib.Path) -> pl.DataFrame:
    # Parquet not available in Polars enscripten-wasm32
    if is_emscripten():
        return pl.from_arrow(pq.read_table(path))
    return pl.read_parquet(path)
