import polars as pl
import matplotlib.pyplot as plt
import matplotlib
import ipywidgets as widgets
import numpy.polynomial as ply

from results_dashboard.widgets import MultiSelect, GroupedMultiSelect
from results_dashboard.plot import relplot


def build_type_sizes(df: pl.DataFrame) -> dict[str, int]:
    cols = ["unpacked_width", "unpacked_type"]
    unpacked_df = df[cols].unique().sort(by=cols)
    unpacked_dict = {
        r["unpacked_type"]: r["unpacked_width"]
        for r in unpacked_df.iter_rows(named=True)
    }
    unpacked_dict["Bool"] = 1
    return unpacked_dict


def make_unpacked_type_wt(df: pl.DataFrame) -> MultiSelect:
    unpacked_dict = build_type_sizes(df)
    return MultiSelect(
        options=list(unpacked_dict.keys()),
        default_filter=lambda v: "32" in v,
    )


def make_arch_funcs(df: pl.DataFrame) -> dict[str, list[str]]:
    cols = ["arch", "func"]
    out = {}
    for arch, func in df.select(cols).unique().sort(by=cols).iter_rows():
        out.setdefault(arch, []).append(func)
    return out


def make_func_wt(arch_funcs: dict[str, list[str]]) -> GroupedMultiSelect:
    return GroupedMultiSelect(
        arch_funcs, default_filter=lambda cat, s: s.endswith("Old") or s.endswith("New")
    )


def make_packed_width_one_pair_wt(unpacked_type_wt: MultiSelect, df: pl.DataFrame):
    unpacked_dict = build_type_sizes(df)
    packed_width_one_wt = widgets.BoundedIntText(
        value=1,
        min=0,
        max=max(unpacked_dict[t] for t in unpacked_type_wt.value),
        step=1,
        layout={"width": "60px"},
    )
    unpacked_dict = build_type_sizes(df)
    packed_width_one_slider_wt = widgets.IntSlider(
        min=0,
        max=packed_width_one_wt.max,
        value=packed_width_one_wt.value,
        step=1,
        orientation="horizontal",
        readout=False,
    )
    widgets.jslink(
        (packed_width_one_wt, "value"), (packed_width_one_slider_wt, "value")
    )
    widgets.jslink((packed_width_one_wt, "max"), (packed_width_one_slider_wt, "max"))

    def _set_max(val):
        packed_width_one_wt.max = max(unpacked_dict[t] for t in unpacked_type_wt.value)

    unpacked_type_wt.observe(_set_max, names="value")

    return packed_width_one_wt, packed_width_one_slider_wt


def build_palette(arch_funcs: dict[str, list[str]]) -> dict[str, RGB]:
    all_funcs: set[str] = set()
    for fs in arch_funcs.values():
        all_funcs.update(fs)

    n = len(all_funcs)
    cmap = matplotlib.colormaps[f"tab{10 if n <= 10 else 20}"]
    colors = [cmap(i)[:3] for i in range(n)]

    return dict(zip(sorted(all_funcs), colors))


def build_dashes(arch_funcs: dict[str, list[str]]):
    dash_styles = [(1, 0), (5, 5), (2, 2), (5, 2, 2, 2), (3, 1, 1, 1, 1, 1)]
    arches = sorted([a for a in arch_funcs.keys() if a != "Dynamic"])
    arches.append("Dynamic")  # Put last
    return dict(zip(arches, dash_styles))


def arch_func_filter(arch_funcs: dict[str, list[str]]):
    out = pl.lit(False)
    for arch, funcs in arch_funcs.items():
        out = out | ((pl.col("arch") == arch) & pl.col("func").is_in(funcs))
    return out


def raw_plot(
    df: pl.DataFrame,
    unpacked_types: str,
    packed_width: int,
    arch_funcs: dict[str, list[str]],
    x_axis: tuple[int, int],
    out,
    **kwargs,
):
    selected_funcs = [f for fs in arch_funcs.values() for f in fs]
    data = (
        df.lazy()
        .filter(pl.col("unpacked_type").is_in(unpacked_types))
        .filter(pl.col("packed_width") == packed_width)
        .filter(arch_func_filter(arch_funcs))
        .filter(
            (x_axis[0] <= pl.col("num_values")) & (pl.col("num_values") <= x_axis[1])
        )
        .collect()
    )
    all_funcs = df["func"].unique()  # Quick because categorical
    splits = {
        "hue": "func" if len(selected_funcs) > 1 else None,
        "col": "unpacked_type" if len(unpacked_types) > 1 else None,
        "style": "arch",
    }
    with out:
        out.clear_output(wait=True)
        g = relplot(
            kind="line",
            data=data,
            x="num_values",
            y="cpu_time",
            **splits,
            **kwargs,
        )
        g.figure.subplots_adjust(top=0.9)
        g.fig.suptitle(
            "Unpack performance{func}{type}".format(
                func="" if len(selected_funcs) != 1 else f" for {funcs[0]} function",
                type="" if len(unpacked_types) != 1 else f" on {unpacked_types[0]}",
            )
        )
        for ax in g.axes.flat:
            ax.set_ylabel(f"CPU time (ns)")
            ax.set_xlabel(f"Integers to unpack (count)")
        plt.show()


def make_x_axis_wt(df: pl.DataFrame) -> widgets.IntRangeSlider:
    x_min, x_max = df.select(
        [
            pl.col("num_values").min().alias("min"),
            pl.col("num_values").max().alias("max"),
        ]
    ).row(0)
    return widgets.IntRangeSlider(
        value=[x_min, x_max],
        min=x_min,
        max=x_max,
        step=10,
        description="Horizontal axis:",
        style={"description_width": "initial"},
        orientation="horizontal",
        readout=True,
        readout_format="d",
    )


def series_linear_regression(series, *args, **kwargs):
    x, y = series
    b, m = ply.Polynomial.fit(x.to_numpy(), y.to_numpy(), deg=1).convert().coef
    return pl.Series([{"slope": m, "intercept": b}])


def linear_regression(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.group_by(["unpacked_type", "arch", "func", "packed_width", "unpacked_width"])
        .agg(
            pl.map_groups(
                exprs=["num_values", "cpu_time"],
                function=series_linear_regression,
                return_dtype=pl.Struct({"slope": pl.Float64, "intercept": pl.Float64}),
                returns_scalar=True,
            ).alias("fit")
        )
        .unnest("fit")
        .with_columns(
            (10**9 / pl.col("slope")).alias("values_per_second"),
            pl.col("intercept").alias("latency_ns"),
        )
        .drop("slope", "intercept")
    )


def plot_speed(
    df: pl.DataFrame,
    unpacked_types: str,
    arch_funcs: dict[str, list[str]],
    out,
    **kwargs,
):
    selected_funcs = [f for fs in arch_funcs.values() for f in fs]
    data = (
        df.lazy()
        .filter(pl.col("unpacked_type").is_in(unpacked_types))
        .filter(arch_func_filter(arch_funcs))
        .collect()
    )
    all_funcs = df["func"].unique()  # Quick because categorical
    splits = {
        "hue": "func" if len(selected_funcs) > 1 else None,
        "col": "unpacked_type" if len(unpacked_types) > 1 else None,
        "style": "arch",
    }
    with out:
        out.clear_output(wait=True)
        g = relplot(
            kind="line",
            data=data,
            x="packed_width",
            y="values_per_second",
            **splits,
            **kwargs,
        )
        g.figure.subplots_adjust(top=0.9)
        g.fig.suptitle(
            "Unpack speed {func}{type}".format(
                func="" if len(selected_funcs) != 1 else f" for {funcs[0]} function",
                type="" if len(unpacked_types) != 1 else f" on {unpacked_types[0]}",
            )
        )
        for ax in g.axes.flat:
            scale_10 = 9
            ax.yaxis.set_major_formatter(lambda x, _: f"{x / 10**scale_10:.1f}")
            ax.set_ylabel(f"speed (×10{'⁰¹²³⁴⁵⁶⁷⁸⁹'[scale_10]} int/s)")
            ax.set_xlabel(f"Packed width (bit)")
        plt.show()
