from collections.abc import Callable
import functools
from typing import Any

import holoviews as hv
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import numpy as np
import pandas as pd
import panel as pn
import param

# import torch
from shaolin.dimension_mapper import (
    AlphaDim,
    ColorDim,
    Dimensions,
    is_string_column,
    LineWidthDim,
    organize_widgets,
    SizeDim,
    widget_priority,
)
from shaolin.utils import find_closest_point


Value = np.ndarray | list  # | torch.Tensor
DictValues = dict[str, Value]
DEFAULT_ROOT_ID = np.uint64(0)
DEFAULT_FIRST_NODE_ID = np.uint64(1)


def create_graphviz_layout(
    graph: nx.DiGraph | nx.Graph,
    top_to_bottom: bool = False,
    prog: str = "dot",
    root: int | str | None = DEFAULT_ROOT_ID,
    args: str = "",
) -> dict[str, tuple[float, float]]:
    """Create a layout for a graph using Graphviz.

    This function uses the Graphviz software to create a layout for a graph. \
    The layout can be either top-to-bottom or left-to-right, depending on the \
    value of the `top_to_bottom` parameter.

    Args:
        graph (nx.DiGraph | nx.Graph): The graph for which to create a layout.
        top_to_bottom (bool, optional): If True, create a top-to-bottom layout.
                                        If False, create a left-to-right layout.
                                        Defaults to False.
        prog (str, optional): The Graphviz layout program to use. Defaults to "dot".
        root (int | str, optional): The id of the root node for the Graphviz layout.
                                    Defaults to DEFAULT_ROOT_ID.
        args (str, optional): Additional arguments to pass to the Graphviz layout program.
                              Defaults to "".

    Returns:
        dict[str, tuple[float, float]]: A dictionary mapping node names to their \
        positions in the layout.

    """
    orig_pos = graphviz_layout(graph, prog=prog, root=root, args=args)  # Branches go top to bottom
    if top_to_bottom:
        return orig_pos
    # We rotate the layout to go from left to right
    return {k: (-y, x) for k, (x, y) in orig_pos.items()}


def nodes_as_df(
    graph: nx.Graph | nx.DiGraph, pos: dict[str, tuple[float, float]] | None = None
) -> pd.DataFrame:
    """Convert the nodes of a graph into a pandas DataFrame.

    Args:
        graph (nx.Graph | nx.DiGraph): The graph whose nodes are to be converted.
        pos (dict[str, tuple[float, float]], optional): A dictionary mapping node \
            names to their positions. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame representing the nodes of the graph. \
            If pos is provided, the DataFrame will also include the positions of the nodes.

    """
    index, data = tuple(zip(*graph.nodes(data=True)))
    df = pd.DataFrame(index=index, data=data)
    if pos is None:
        return df
    df_pos = pd.DataFrame.from_dict(pos, orient="index", columns=["x", "y"])
    return pd.concat([df, df_pos], axis=1)


def edges_as_df(graph: nx.Graph | nx.DiGraph, data: bool = True) -> pd.DataFrame:
    """Convert the edges of a graph into a pandas DataFrame.

    Args:
        graph (nx.Graph | nx.DiGraph): The graph whose edges are to be converted.
        data (bool, optional): Whether to include the edge attributes in the DataFrame. \
            Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame representing the edges of the graph.

    """
    start, end, *data_ = tuple(zip(*graph.edges(data=data)))
    edges = np.array([list(start), list(end)]).T
    edge_data = pd.DataFrame(columns=["from", "to"], data=edges)
    if data_:
        edge_data = pd.concat([edge_data, pd.DataFrame(data_[0])], axis=1)
    return edge_data


def parse_attributes(attrs: DictValues) -> dict[str, Any]:
    """Parses the attributes of a node or edge.

    Args:
        attrs (dict[str, Value]): The attributes to be parsed.

    Returns:
        dict: A dictionary of parsed attributes. If an attribute's value is a \
                    list of length 1, a torch tensor with one element, or a numpy array \
                    with one element, the value is extracted from the list/tensor/array.

    """
    new_attrs = {}
    for attr, value in attrs.items():
        if isinstance(value, list) and len(value) == 1:
            new_attrs[attr] = value[0]
        elif isinstance(value, np.ndarray) and value.size == 1:
            new_attrs[attr] = value.item()
        # elif torch.is_tensor(value) and value.numel() == 1:
        #    new_attrs[attr] = value.item()
    return new_attrs


def simplify_graph_attributes(
    graph: nx.DiGraph | nx.Graph,
    default_node_attrs: DictValues = None,
    default_edge_attrs: DictValues = None,
) -> tuple[nx.DiGraph | nx.Graph, dict[int | str, int]]:
    """Simplifies the attributes of a graph's nodes and edges.

    Args:
        graph (nx.DiGraph | nx.Graph): The graph whose attributes are to be simplified.
        default_node_attrs (DictWalker, optional): Default attributes for nodes. \
                    Defaults to None.
        default_edge_attrs (DictWalker, optional): Default attributes for edges. \
                    Defaults to None.

    Returns:
        tuple: A tuple containing the new graph with simplified attributes \
                    and a dictionary mapping the original node names to the new node names.

    """
    new_graph = nx.DiGraph() if isinstance(graph, nx.DiGraph) else nx.Graph()
    new_node_names = {DEFAULT_ROOT_ID: 0, DEFAULT_FIRST_NODE_ID: 1}
    ix = 2
    for node, attrs in graph.nodes(data=True):
        if node not in new_node_names:
            new_node_names[node] = ix
            ix += 1
        default_attrs = {} if default_node_attrs is None else default_node_attrs
        new_attrs = {**default_attrs, **parse_attributes(attrs)}
        new_graph.add_node(new_node_names[node], **new_attrs)

    for src, dst, attrs in graph.edges(data=True):
        default_attrs = {} if default_edge_attrs is None else default_edge_attrs
        new_attrs = {**default_attrs, **parse_attributes(attrs)}
        new_graph.add_edge(new_node_names[src], new_node_names[dst], **new_attrs)

    return new_graph, new_node_names


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


def plot_graph(
    df_nodes: pd.DataFrame,
    df_edges: pd.DataFrame,
    dim_x: str = "x",
    dim_y: str = "y",
    node_kdims=None,
    ignore_node_cols: tuple[str] | None = ("states",),
    **kwargs,
):
    if ignore_node_cols:
        df_nodes = df_nodes.drop(columns=list(ignore_node_cols))
    node_kdims = [dim_x, dim_y, "index"] if node_kdims is None else node_kdims
    nodes = hv.Nodes(df_nodes, kdims=node_kdims)
    # If you are mapping a visual property to a column that is shared between
    # df_edges and df_nodes the plot will break, so let's make sure the columns
    # do not overlap.
    dfe = df_edges[[x for x in df_edges.columns if x not in df_nodes.columns]]
    # if dim_x == "x":
    #    kwargs["xaxis"] = None
    # if dim_y == "y":
    #    kwargs["yaxis"] = None
    return hv.Graph((dfe, nodes)).opts(**kwargs)


class InteractiveGraph(param.Parameterized):
    def __init__(
        self,
        df_nodes: pd.DataFrame,
        df_edges: pd.DataFrame,
        ignore_node_cols: tuple[str] | None = None,
        n_cols=3,
    ):
        self.n_cols = n_cols
        super().__init__()
        self.df_nodes = df_nodes

        if ignore_node_cols is None:
            ignore_node_cols = tuple(
                c for c in df_nodes.columns if is_string_column(self.df_nodes, c)
            )
        self.df_edges = df_edges
        self._fix_df_edges_columns()
        self.ignore_node_cols = ignore_node_cols
        self.width = pn.widgets.IntSlider(name="width", start=400, end=2000, value=1000)
        self.height = pn.widgets.IntSlider(name="height", start=400, end=2000, value=600)
        self.node_dims = Dimensions(
            self.df_nodes,
            self.n_cols,
            node_size=SizeDim,
            node_color=ColorDim,
            node_alpha=AlphaDim,
        )
        self.edge_dims = Dimensions(
            self.df_edges,
            self.n_cols,
            edge_color=(ColorDim, {"default": "black"}),
            edge_alpha=AlphaDim,
            edge_line_width=LineWidthDim,
        )
        valid_columns = self.node_dims.dimensions["node_size"].valid_cols
        self.sel_x = pn.widgets.Select(name="x", options=valid_columns, value="x")
        self.sel_y = pn.widgets.Select(name="y", options=valid_columns, value="y")
        streams = {**self.node_dims.streams, **self.edge_dims.streams}
        streams["dim_x"] = self.sel_x.param.value
        streams["dim_y"] = self.sel_y.param.value
        self.dmap = hv.DynamicMap(self.view, streams=streams)
        self.tap_stream = hv.streams.Tap(source=self.dmap, x=np.nan, y=np.nan)

    def _fix_df_edges_columns(self):
        rename = {k: f"{k}_" for k in self.df_edges.columns if k in self.df_nodes.columns}
        self.df_edges = self.df_edges.rename(columns=rename)

    def bind_to_stream(self, function: callable):
        return pn.bind(function, x=self.tap_stream.param.y, y=self.tap_stream.param.x)

    @param.depends("sel_x.value", "sel_y.value")
    def update_lims(self):
        self.dmap = self.dmap.redim.range(
            x=(self.df_nodes[self.sel_x.value].min(), self.df_nodes[self.sel_x.value].max()),
            y=(self.df_nodes[self.sel_y.value].min(), self.df_nodes[self.sel_y.value].max()),
        )

    def view(self, **kwargs):
        return plot_graph(
            df_nodes=self.df_nodes,
            df_edges=self.df_edges,
            ignore_node_cols=self.ignore_node_cols,
            **kwargs,
        ).opts(
            width=self.width.value,
            height=self.height.value,
            title="",
            framewise=True,
            colorbar=True,
        )

    def layout(self):
        all_dims = {**self.node_dims.dimensions, **self.edge_dims.dimensions}
        dimensions = dict(sorted(all_dims.items(), key=widget_priority))
        widgets = [dimension.panel() for dimension in dimensions.values()]
        return pn.Column(
            organize_widgets(widgets, self.n_cols),
            pn.Row(self.sel_x, self.sel_y, self.height, self.width),
        )

    def panel(self):
        hv_panel = pn.pane.HoloViews(self.dmap)
        self.height.link(hv_panel[0], value="height")
        self.width.link(hv_panel[0], value="width")
        return pn.Row(hv_panel, self.update_lims)
