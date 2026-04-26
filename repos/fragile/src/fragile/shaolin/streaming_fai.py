from collections.abc import Callable
import functools

from bokeh.models import HoverTool
import holoviews as hv
from holoviews.streams import Pipe
import numpy as np
import pandas as pd
import panel as pn
import param
from shaolin.utils import find_closest_point

from fragile.benchmarks import OptimBenchmark
from fragile.shaolin.dimension_mapper import (
    AlphaDim,
    ColorDim,
    Dimensions,
    is_string_column,
    organize_widgets,
    SizeDim,
    widget_priority,
)


def select_index(xy_columns: list[str] | None = None, df: pd.DataFrame | None = None) -> Callable:
    """Decorator that transforms a function by finding the index of the closest point.

    This decorator takes the `xy_columns` and an optional `df` as arguments and returns a \
    new function that takes `ix` and `df` as arguments. Inside the decorator, \
    it calls the original function with the index of the closest point and the dataframe.

    If `df` is provided when using the decorator, it will be used as the default dataframe \
    for the decorated function.

    Args:
        xy_columns (List[str], optional): List of column names to be used for x and y
                                          coordinates. Defaults to ["x", "y"].
        df (pd.DataFrame, optional): Default dataframe to be used in the decorated function.
                                      If not provided, the dataframe must be passed as an argument
                                      when calling the decorated function. Defaults to None.

    Returns:
        Callable: The new function that takes `ix` and `df` as arguments.

    """
    if xy_columns is None:
        xy_columns = ["x", "y"]

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(x: float, y: float, df_: pd.DataFrame = None) -> Callable:
            nonlocal df
            if df_ is not None:
                df = df_
            if df is None:
                msg = (
                    "A dataframe must be provided when calling the "
                    "decorated function, or when using the decorator."
                )
                raise ValueError(msg)
            points = df[xy_columns].values
            ix = find_closest_point(points, x, y)
            return func(ix, df)

        return wrapper

    return decorator


def get_segments(parent, x, y):
    return x[parent], y[parent], x, y


def plot_edges(df, x_col="x", y_col="y", parent_col="parent", **kwargs):
    parent, x, y = df[parent_col].values, df[x_col].values, df[y_col].values
    segs = x[parent], y[parent], x, y
    kwargs["line_color"] = kwargs.get("line_color", "black")
    return hv.Segments(segs).opts(**kwargs)


def plot_nodes(df, x_col: str = "x", y_col: str = "y", **kwargs):
    return hv.Scatter(df, kdims=[x_col], vdims=[y_col]).opts(**kwargs)


def plot_graph(df, edge_kwargs=None, **node_kwargs):
    edge_kwargs = edge_kwargs or {}
    return plot_edges(df, **edge_kwargs) * plot_nodes(df, **node_kwargs)


def draw_benchmark(
    benchmark,
    benchmark_lims: str = "bounds",
    x_data: np.ndarray | None = None,
    y_data: np.ndarray | None = None,
):
    x, y = benchmark.best_state[:2]
    if benchmark_lims == "bounds":
        x_low, y_low = benchmark.bounds.low[:2].numpy(force=True).tolist()
        x_high, y_high = benchmark.bounds.high[:2].numpy(force=True).tolist()
    elif benchmark_lims == "data":
        x_low, y_low = (x_data.min(), y_data.min()) if x_data is not None else (None, None)
        x_high, y_high = (x_data.max(), y_data.max()) if x_data is not None else (None, None)
    else:
        x_low, y_low = None, None
        x_high, y_high = None, None
    return hv.Scatter(([x], [y])).opts(
        marker="star", color="red", size=8, xlim=(x_low, x_high), ylim=(y_low, y_high), alpha=0.7
    )


def view_plot(
    self,
    df: pd.DataFrame,
    dim_x: str = "x",
    dim_y: str = "y",
    draw_edges: bool = True,
    benchmark_lims: str = "data",
    ignore_cols: tuple[str] | None = ("states",),
    **kwargs,
):
    # df = self.df
    if ignore_cols:
        df = df.drop(columns=[c for c in ignore_cols if c in df.columns])
    hover_cols = [c for c in df.columns if not is_string_column(df, c)]
    tooltips = [("Index", "$index")] + [
        (n.capitalize().replace("_", " "), f"@{n}") for n in hover_cols
    ]
    hover = HoverTool(tooltips=tooltips)
    plot = hv.Scatter(df, kdims=[dim_x], vdims=[dim_y, *hover_cols]).opts(
        width=self.width.value,
        height=self.height.value,
        title="",
        framewise=True,
        colorbar=True,
        tools=[hover],
        **kwargs,
    )
    if draw_edges:
        plot = plot_edges(df, x_col=dim_x, y_col=dim_y) * plot
    if isinstance(self.fai.env, OptimBenchmark) and dim_x == "x" and dim_y == "y":
        self.xlim_mode.visible = True
        plot *= draw_benchmark(
            self.fai.env,
            benchmark_lims=benchmark_lims,
            x_data=df[dim_x].values,
            y_data=df[dim_y].values,
        )
    else:
        self.xlim_mode.visible = False
    return plot


def plot_table(data, **kwargs):
    default_kwargs = {"width": 150}
    kwargs = {**default_kwargs, **kwargs}
    return hv.Table(data).opts(**kwargs)


class InteractiveFai(param.Parameterized):
    def __init__(
        self,
        fai,
        ignore_cols: tuple[str] | None = None,
        n_cols=3,
        default_x_col: str | None = "x",
        default_y_col: str | None = "y",
    ):
        self.n_cols = n_cols
        self.fai = fai
        df = pd.DataFrame(fai.to_dict())
        summary = pd.DataFrame(fai.summary(), index=["value"]).T.reset_index()
        super().__init__()
        self.data_pipe = Pipe(data=df) if isinstance(df, pd.DataFrame) else df
        self.summary_pipe = Pipe(data=summary)

        if ignore_cols is None:
            ignore_cols = tuple(c for c in df.columns if is_string_column(self.df, c))
        self.ignore_cols = ignore_cols
        self.width = pn.widgets.EditableIntSlider(
            name="width", start=400, end=2000, value=1000, width=200
        )
        self.height = pn.widgets.EditableIntSlider(
            name="height", start=400, end=2000, value=600, width=200
        )
        self.xlim_mode = pn.widgets.Select(
            name="Plot boundaries",
            options=["data", "bounds", "best+data"],
            value="data",
            width=100,
        )
        self.df_dims = Dimensions(
            self.data_pipe,
            self.n_cols,
            size=SizeDim,
            color=ColorDim,
            alpha=AlphaDim,
        )
        valid_columns = self.df_dims.dimensions["size"].valid_cols
        default_x_col = default_x_col or valid_columns[0]
        default_y_col = default_y_col or valid_columns[1]
        self.sel_x = pn.widgets.Select(
            name="x column", options=valid_columns, value=default_x_col, width=150
        )
        self.sel_y = pn.widgets.Select(
            name="y column", options=valid_columns, value=default_y_col, width=150
        )
        streams = self.df_dims.streams
        streams["dim_x"] = self.sel_x.param.value
        streams["dim_y"] = self.sel_y.param.value
        streams["df"] = self.data_pipe.param.data
        streams["benchmark_lims"] = self.xlim_mode.param.value
        self.dmap = hv.DynamicMap(functools.partial(view_plot, self=self), streams=streams)
        self.tap_stream = hv.streams.Tap(source=self.dmap, x=np.nan, y=np.nan)
        self.summary_dmap = hv.DynamicMap(plot_table, streams=[self.summary_pipe.param.data])

    @property
    def df(self):
        return self.data_pipe.data

    def send(self, fai):
        self.data_pipe.send(pd.DataFrame(fai.to_dict()))
        self.summary_pipe.send(pd.DataFrame(fai.summary(), index=["value"]).T.reset_index())

    def reset(self, fai):
        self.data_pipe.send(pd.DataFrame(fai.to_dict()))
        self.summary_pipe.send(pd.DataFrame(fai.summary(), index=["value"]).T.reset_index())

    def bind_to_stream(self, function: Callable):
        return pn.bind(function, x=self.tap_stream.param.x, y=self.tap_stream.param.y)

    def bind_tap(self, func: Callable, df: pd.DataFrame | None = None) -> Callable:
        """Bind a function to the tap event of the plot."""

        @functools.wraps(func)
        def wrapper(x: float, y: float, df_: pd.DataFrame = None) -> Callable:
            nonlocal df
            if df_ is not None:
                df = df_
            if df is None:
                df = self.data_pipe.data
            xy_cols = [self.sel_x.value, self.sel_y.value]
            points = df[xy_cols].values.astype(float)
            ix = find_closest_point(points, x, y)
            return func(ix, df)

        return self.bind_to_stream(wrapper)

    @param.depends("sel_x.value", "sel_y.value", "data_pipe.data")
    def update_lims(self):
        self.dmap = self.dmap.redim.range(
            x=(self.df[self.sel_x.value].min(), self.df[self.sel_x.value].max()),
            y=(self.df[self.sel_y.value].min(), self.df[self.sel_y.value].max()),
        )

    def layout(self):
        all_dims = self.df_dims.dimensions
        dimensions = dict(sorted(all_dims.items(), key=widget_priority))
        widgets = [dimension.panel() for dimension in dimensions.values()]
        return pn.Column(
            pn.Row(
                pn.Column(self.sel_x, self.sel_y, self.xlim_mode),
                pn.Column(self.height, self.width),
            ),
            organize_widgets(widgets, self.n_cols, sizing_mode="stretch_width"),
        )

    def view(self):
        hv_panel = pn.Row(
            pn.pane.HoloViews(self.summary_dmap, width=150), pn.pane.HoloViews(self.dmap)
        )
        self.height.link(hv_panel[0], value="height")
        self.width.link(hv_panel[0], value="width")
        return pn.Row(hv_panel, self.update_lims)

    def __panel__(self):
        return pn.Column(self.layout, self.view)
