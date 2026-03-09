"""
Minimal matplotlib port of seaborn.relplot (kind="line" only).

Supports only Polars DataFrames. Raises NotImplementedError for unsupported
parameters or types.

AI generated.

Usage
-----
    import polars as pl
    from relplot import relplot

    fig = relplot(
        data=df,
        x="time",
        y="value",
        hue="model",
        palette={"model_a": (0.2, 0.4, 0.8), "model_b": (0.9, 0.3, 0.2)},
        style="variant",
        dashes={"solid_line": (1, 0), "dashed": (5, 5), "dotdash": (3, 5, 1, 5)},
        col="dataset",
        aspect=1.5,
        facet_kws={"sharex": False},
    )
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
import polars as pl

type RGB = tuple[float, float, float]
type DashSeq = tuple[int, ...]


class FacetGrid:
    """Minimal stand-in for seaborn.FacetGrid.

    Exposes the same surface used for post-hoc tweaking:
        g.figure / g.fig   → the matplotlib Figure
        g.axes             → the 2-D ndarray of Axes (always shape (1, n))
        g.axes.flat        → flat iterator over every Axes
    """

    __slots__ = ("_fig", "_axes")

    def __init__(self, fig: matplotlib.figure.Figure, axes: np.ndarray) -> None:
        self._fig = fig
        self._axes = axes  # shape (1, n_facets)

    @property
    def figure(self) -> matplotlib.figure.Figure:
        return self._fig

    @property
    def fig(self) -> matplotlib.figure.Figure:
        return self._fig

    @property
    def axes(self) -> np.ndarray:
        return self._axes

    def savefig(self, *args, **kwargs) -> None:
        self._fig.savefig(*args, **kwargs)


def relplot(
    *,
    data: pl.DataFrame,
    x: str,
    y: str,
    kind: str = "line",
    hue: str | None = None,
    palette: dict[str, RGB] | None = None,
    style: str | None = None,
    dashes: dict[str, DashSeq] | None = None,
    col: str | None = None,
    col_order: list[str] | None = None,
    aspect: float = 1.0,
    height: float = 5.0,
    facet_kws: dict[str, object] | None = None,
) -> FacetGrid:
    """Minimal relplot — line kind only, Polars DataFrames only.

    Parameters
    ----------
    data : pl.DataFrame
    x, y : str
        Column names for the horizontal / vertical axes.
    kind : str
        Only ``"line"`` is implemented.
    hue : str | None
        Column whose distinct values map to colours.  When provided,
        *palette* **must** also be provided as ``dict[str, RGB tuple]``.
    palette : dict[str, tuple[float,float,float]] | None
        Mapping from each *hue* level to an RGB tuple with components in
        [0, 1].  Required when *hue* is set.
    style : str | None
        Column whose distinct values map to dash patterns.  When provided,
        *dashes* **must** also be provided as ``dict[str, tuple[int, ...]]``.
    dashes : dict[str, tuple[int, ...]] | None
        Mapping from each *style* level to a matplotlib on/off ink sequence.
        ``(1, 0)`` or ``()`` → solid line; ``(5, 5)`` → simple dashes;
        ``(3, 5, 1, 5)`` → dash-dot, etc.
    col : str | None
        Column used to facet the data into side-by-side subplots.
    aspect : float
        Width-to-height ratio of each facet (width = aspect × height).
    height : float
        Height in inches of each facet.
    facet_kws : dict | None
        Currently only ``{"sharex": bool}`` is supported.  *sharey* is
        always ``True``.

    Returns
    -------
    FacetGrid
    """

    # ── Guards ────────────────────────────────────────────────────────
    if kind != "line":
        raise NotImplementedError(f"Only kind='line' is implemented, got {kind!r}")
    if not isinstance(data, pl.DataFrame):
        raise NotImplementedError(
            f"Only polars.DataFrame is supported, got {type(data).__name__}"
        )
    for name, label in [(x, "x"), (y, "y")]:
        if name not in data.columns:
            raise ValueError(f"{label}={name!r} is not a column in the DataFrame")

    if hue is not None:
        if palette is None:
            raise NotImplementedError(
                "When hue is set, palette must be provided as dict[str, RGB tuple]"
            )
        if not isinstance(palette, dict):
            raise NotImplementedError(
                f"palette must be a dict mapping hue levels to RGB tuples, "
                f"got {type(palette).__name__}"
            )
        if hue not in data.columns:
            raise ValueError(f"hue={hue!r} is not a column in the DataFrame")

    if style is not None:
        if dashes is None:
            raise NotImplementedError(
                "When style is set, dashes must be provided as "
                "dict[str, tuple[int, ...]]"
            )
        if not isinstance(dashes, dict):
            raise NotImplementedError(
                f"dashes must be a dict mapping style levels to dash tuples, "
                f"got {type(dashes).__name__}"
            )
        if style not in data.columns:
            raise ValueError(f"style={style!r} is not a column in the DataFrame")

    if not isinstance(aspect, int | float):
        raise NotImplementedError(
            f"aspect must be a float, got {type(aspect).__name__}"
        )

    sharex: bool = True
    if facet_kws is not None:
        unsupported = set(facet_kws) - {"sharex"}
        if unsupported:
            raise NotImplementedError(
                f"Unsupported facet_kws keys: {unsupported}. "
                f"Only 'sharex' is implemented."
            )
        sharex = facet_kws.get("sharex", True)
        if not isinstance(sharex, bool):
            raise NotImplementedError("facet_kws['sharex'] must be a bool")

    # ── Resolve facet columns ─────────────────────────────────────────
    if col is not None:
        if col not in data.columns:
            raise ValueError(f"col={col!r} is not a column in the DataFrame")
        if col_order is not None:
            available = set(_unique_sorted(data, col))
            col_levels = [lv for lv in col_order if lv in available]
        else:
            col_levels = _unique_sorted(data, col)
        n_facets = len(col_levels)
    else:
        col_levels = [None]
        n_facets = 1

    # ── Resolve hue / style levels ────────────────────────────────────
    if hue is not None:
        hue_levels = _unique_sorted(data, hue)
        missing = set(hue_levels) - palette.keys()
        if missing:
            raise ValueError(f"palette is missing entries for hue levels: {missing}")
    else:
        hue_levels = [None]

    if style is not None:
        style_levels = _unique_sorted(data, style)
        missing_s = set(style_levels) - dashes.keys()
        if missing_s:
            raise ValueError(f"dashes is missing entries for style levels: {missing_s}")
    else:
        style_levels = [None]

    # ── Work out whether we'll have a legend ────────────────────────
    has_legend = hue is not None or style is not None

    # ── Create figure + axes ──────────────────────────────────────────
    # Each facet is exactly (aspect * height) × height.  If there is a
    # legend we add extra width so the *plot area* keeps its aspect ratio
    # rather than being squashed.
    legend_inches = 1.5 if has_legend else 0.0
    fig_w = aspect * height * n_facets + legend_inches
    fig, axes = plt.subplots(
        nrows=1,
        ncols=n_facets,
        figsize=(fig_w, height),
        sharex=sharex,
        sharey=True,
        squeeze=False,
    )
    ax_row = axes[0]

    # ── Plot each facet ───────────────────────────────────────────────
    for facet_idx, col_level in enumerate(col_levels):
        ax = ax_row[facet_idx]

        facet_df = (
            data.filter(pl.col(col).cast(pl.Utf8) == col_level)
            if col is not None
            else data
        )

        for h_level in hue_levels:
            for s_level in style_levels:
                group = facet_df
                if hue is not None:
                    group = group.filter(pl.col(hue).cast(pl.Utf8) == h_level)
                if style is not None:
                    group = group.filter(pl.col(style).cast(pl.Utf8) == s_level)

                if group.is_empty():
                    continue

                group = group.sort(x)

                color = palette[h_level] if hue is not None else None
                linestyle = (
                    _dash_to_linestyle(dashes[s_level]) if style is not None else "-"
                )

                label_parts = [
                    s
                    for s in (
                        h_level if hue is not None else None,
                        s_level if style is not None else None,
                    )
                    if s is not None
                ]
                label = " / ".join(label_parts) or None

                ax.plot(
                    group[x].to_list(),
                    group[y].to_list(),
                    color=color,
                    linestyle=linestyle,
                    label=label,
                )

        if col is not None:
            ax.set_title(f"{col} = {col_level}")

        ax.set_xlabel(x)
        if facet_idx == 0:
            ax.set_ylabel(y)
        else:
            ax.set_ylabel("")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ── Legend ─────────────────────────────────────────────────────────
    # Place legend outside the last axes so it never overlaps data.
    handles_map: dict[str, object] = {}
    for ax in ax_row:
        for h, lbl in zip(*ax.get_legend_handles_labels()):
            handles_map.setdefault(lbl, h)

    if handles_map:
        ax_row[-1].legend(
            handles_map.values(),
            handles_map.keys(),
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
        )

    fig.tight_layout()

    return FacetGrid(fig, axes)


# ── Helpers ───────────────────────────────────────────────────────────


def _unique_sorted(df: pl.DataFrame, column: str) -> list[str]:
    return df.select(pl.col(column).cast(pl.Utf8)).to_series().unique().sort().to_list()


def _dash_to_linestyle(seq: DashSeq) -> str | tuple[int, DashSeq]:
    """Convert an on/off ink tuple to a matplotlib *linestyle* value.

    Seaborn's ``dashes`` dict values are bare ``(on, off, ...)`` tuples.
    Matplotlib expects either a named style or ``(offset, (on, off, ...))``.

    ``()`` and ``(1, 0)`` are treated as solid.
    """
    if not seq or seq == (1, 0):
        return "-"
    return (0, seq)
