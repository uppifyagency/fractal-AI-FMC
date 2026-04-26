from typing import Any

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import param
from shaolin.colormaps import ColorMap


def is_string_column(df, column_name):
    if isinstance(df[column_name].dtype, pd.CategoricalDtype):
        return False
    return df[column_name].apply(lambda x: isinstance(x, str)).any()


def is_bool_column(df, column_name):
    return df[column_name].apply(lambda x: isinstance(x, bool | np.bool_)).any()


def normalize_array(array, min_val, max_val):
    array, min_val, max_val = array.astype(float), float(min_val), float(max_val)
    return (array - min_val) / (max_val - min_val)


def rank_values(array):
    return np.argsort(np.argsort(array))


def organize_widgets(widgets, n_cols, sizing_mode="stretch_both"):
    rows = [pn.Row(*widgets[i : i + n_cols]) for i in range(0, len(widgets), n_cols)]
    return pn.Column(*rows, sizing_mode=sizing_mode)


class DimensionMapper(param.Parameterized):
    dim_name = param.String(per_instance=True, precedence=-1)
    value = param.ClassSelector(class_=object, default=None, per_instance=True, precedence=-1)

    def __init__(
        self,
        df: pd.DataFrame,
        name,
        default_value,
        value_range: tuple[int, int] | tuple[int, int, int],
        ignore_cols: tuple[str, ...] = (),
        resolution: int = 100,
        default_range: tuple[int, int] | tuple[float, float] | None = None,
        ignore_string_cols: bool = True,
        epsilon: float = 1e-5,
    ):
        self.epsilon = epsilon
        self.ignore_string_cols = ignore_string_cols
        self.ignore_cols = ignore_cols
        self._df = df
        self.default_value = default_value
        self.valid_cols = self.get_valid_columns()
        self.value_range = self.init_range_widget(value_range, resolution, default_range)
        self.is_bool_col = False
        self.std_col = 0
        self.max_col = 1
        self.min_col = 0
        self.len_col = 1
        self.norm_limits = None
        self.button_check = pn.widgets.CheckButtonGroup(
            name="Transform",
            button_type="primary",
            button_style="outline",
            options=["Invert", "Log scale", "Rank"],
            value=[],
            width=225,
        )
        self.column = pn.widgets.Select(
            name="Column", options=[None, *self.valid_cols], value=None, width=150
        )

        super().__init__(dim_name=name)
        self.get_value()

    @property
    def df(self):
        return self._df if isinstance(self._df, pd.DataFrame) else self._df.data

    def get_valid_columns(self):
        cols = [c for c in self.df.columns if c not in self.ignore_cols]
        if self.ignore_string_cols:
            cols = [c for c in cols if not is_string_column(self.df, c)]
        return sorted(cols)

    def init_range_widget(self, val_range, resolution, default_range):
        start, end = val_range[:2]
        has_step = len(val_range) == 3  # noqa: PLR2004 (start, end, step)
        step = val_range[2] if has_step else (end - start) / resolution
        default_range = default_range or (start, end)
        return pn.widgets.EditableRangeSlider(
            name="Range",
            start=start,
            end=end,
            step=step,
            value=default_range,
            width=225,
        )

    def get_dim_markdown(self):
        pretty_name = self.dim_name.capitalize()
        pretty_name = " ".join(pretty_name.split("_"))
        return pn.pane.Markdown(f"**{pretty_name}**")

    @param.depends("column.value")
    def update_ui(self):
        use_default = self.column.value is None or bool(self.std_col == 0)
        self.default_value.visible = use_default
        self.value_range.visible = not use_default
        self.button_check.visible = not use_default

    @param.depends("column.value")
    def update_col_values(self):
        if self.column.value is None:
            return

        self.is_bool_col = is_bool_column(self.df, self.column.value)
        self.max_col = self.df[self.column.value].max()
        self.min_col = self.df[self.column.value].min()
        self.len_col = self.df[self.column.value].values.shape[0]
        self.std_col = self.df[self.column.value].std()

    def panel(self):
        return pn.Column(
            pn.Row(self.get_dim_markdown, self.column, self.default_value),
            self.value_range,
            self.button_check,
            self.update_col_values,
            self.update_ui,
            self.get_value,
            min_height=60,
            max_height=150,
            height_policy="min",
        )

    @param.depends(
        "column.value", "value_range.value", "button_check.value", "default_value.value"
    )
    def get_value(self):
        if self.column.value is None or self.std_col == 0:
            self.value = self.default_value.value
            return
        will_invert = "Invert" in self.button_check.value
        will_log = "Log scale" in self.button_check.value and not self.is_bool_col
        # There is a bug when normalizing boolean values
        # that throws a ZeroDivisionError. To avoid it
        # set a small limit manually.
        limits = (self.epsilon, 1.0) if self.is_bool_col else (self.min_col, self.max_col)
        value = hv.dim(self.column.value)
        if "Rank" in self.button_check.value:
            value = rank_values(value)
            limits = (0, self.len_col)

        normed_val = normalize_array(value, limits[0], limits[1])
        if will_invert:
            normed_val = normed_val * -1 + 1
        if will_log:
            normed_val = normalize_array(
                np.log(normed_val + self.epsilon), np.log(self.epsilon), 1
            )
        self.value = (
            normed_val * (self.value_range.value[1] - self.value_range.value[0])
            + self.value_range.value[0]
        )


class SizeDim(DimensionMapper):
    def __init__(
        self,
        df: pd.DataFrame,
        name="size",
        default=8,
        value_range: tuple[int, int] | tuple[int, int, int] = (0, 25),
        ignore_cols: tuple[str, ...] = (),
        resolution: int = 100,
        default_range: tuple[int, int] | tuple[float, float] | None = (1, 10),
        ignore_string_cols: bool = True,
        epsilon: float = 1e-7,
    ):
        default_widget = pn.widgets.FloatInput(name="Default", value=default, width=100)
        super().__init__(
            name=name,
            df=df,
            default_value=default_widget,
            value_range=value_range,
            ignore_cols=ignore_cols,
            resolution=resolution,
            default_range=default_range,
            ignore_string_cols=ignore_string_cols,
            epsilon=epsilon,
        )


class AlphaDim(DimensionMapper):
    def __init__(
        self,
        df: pd.DataFrame,
        name="alpha",
        default=1.0,
        value_range: tuple[int, int] | tuple[int, int, int] = (0.0, 1.0),
        ignore_cols: tuple[str, ...] = (),
        resolution: int = 100,
        default_range: tuple[int, int] | tuple[float, float] | None = (0.1, 1.0),
        ignore_string_cols: bool = True,
        epsilon: float = 1e-7,
    ):
        default_widget = pn.widgets.FloatInput(name="Default", value=default, width=100)
        super().__init__(
            name=name,
            df=df,
            default_value=default_widget,
            value_range=value_range,
            ignore_cols=ignore_cols,
            resolution=resolution,
            default_range=default_range,
            ignore_string_cols=ignore_string_cols,
            epsilon=epsilon,
        )


class LineWidthDim(DimensionMapper):
    def __init__(
        self,
        df: pd.DataFrame,
        name="line_width",
        default=2.0,
        value_range: tuple[int, int] | tuple[int, int, int] = (0.0, 6.0),
        ignore_cols: tuple[str, ...] = (),
        resolution: int = 100,
        default_range: tuple[int, int] | tuple[float, float] | None = (0.5, 3),
        ignore_string_cols: bool = True,
        epsilon: float = 1e-7,
    ):
        default_widget = pn.widgets.FloatInput(name="Default", value=default, width=100)
        super().__init__(
            name=name,
            df=df,
            default_value=default_widget,
            value_range=value_range,
            ignore_cols=ignore_cols,
            resolution=resolution,
            default_range=default_range,
            ignore_string_cols=ignore_string_cols,
            epsilon=epsilon,
        )


class ColorDim(DimensionMapper):
    cmap = param.ClassSelector(class_=object, default="viridis", per_instance=True, precedence=-1)

    def __init__(
        self,
        df: pd.DataFrame,
        name="color",
        default="#30a2da",
        value_range: tuple[int, int] | tuple[int, int, int] = (0.0, 1.0),
        ignore_cols: tuple[str, ...] = (),
        resolution: int = 100,
        default_range: tuple[int, int] | tuple[float, float] | None = (0.0, 1.0),
        ignore_string_cols: bool = True,
        epsilon: float = 1e-7,
    ):
        default_widget = pn.widgets.ColorPicker(name="Default", value=default, width=100)
        self.colormap_widget = ColorMap()

        super().__init__(
            name=name,
            df=df,
            default_value=default_widget,
            value_range=value_range,
            ignore_cols=ignore_cols,
            resolution=resolution,
            default_range=default_range,
            ignore_string_cols=ignore_string_cols,
            epsilon=epsilon,
        )
        self.value_range.visible = False

    @param.depends(
        "column.value", "value_range.value", "button_check.value", "default_value.value"
    )
    def get_value(self):
        if self.column.value is None or self.std_col == 0:
            self.value = self.default_value.value
            return
        will_invert = "Invert" in self.button_check.value
        will_log = "Log scale" in self.button_check.value and not self.is_bool_col
        value = hv.dim(self.column.value)
        if self.is_bool_col:
            # There is a bug when normalizing boolean values
            # that throws a ZeroDivisionError. To avoid it
            # we manually categorize the mappings.
            value = hv.dim(self.column.value).categorize({
                np.nan: np.nan,
                0.0: 0.0,
                1.0: 1.0,
                "False": 0.0,
                "True": 1.0,
            })
            # return

        limits = (self.min_col, self.max_col)
        if "Rank" in self.button_check.value:
            value = rank_values(value)
            limits = (0, self.len_col)

        normed_val = normalize_array(value, limits[0], limits[1])
        if will_invert:
            normed_val = normed_val * -1 + 1
        if will_log:
            normed_val = normalize_array(
                np.log(normed_val + self.epsilon), np.log(self.epsilon), 1
            )
        self.value = (
            normed_val * (self.value_range.value[1] - self.value_range.value[0])
            + self.value_range.value[0]
        )

    @param.depends("column.value")
    def update_ui(self):
        use_default = self.column.value is None
        std_0 = bool(self.std_col == 0)
        self.default_value.visible = use_default or std_0
        self.colormap_widget.cmap_widget.visible = not use_default and not std_0
        self.colormap_widget.autocomplete.visible = not use_default and not std_0
        self.button_check.visible = not use_default and not self.is_bool_col and not std_0

    @param.depends("colormap_widget.value")
    def update_cmap(self):
        self.cmap = self.colormap_widget.value

    @param.depends("column.value")
    def panel(self):
        return pn.Column(
            pn.Row(self.get_dim_markdown, self.column, self.default_value),
            self.button_check,
            self.colormap_widget.view(),
            self.value_range,
            self.update_col_values,
            self.update_ui,
            self.update_cmap,
            self.get_value,
        )


def add_to_streams(streams: dict, dim):
    name = dim.dim_name
    if not isinstance(dim, ColorDim):
        streams[name] = dim.param.value
    else:
        streams[name] = dim.param.value
        if name.endswith("color"):
            name = name[: -len("color")]
        name = name.removesuffix("_")
        cmap_name = f"{name}_cmap" if name else "cmap"
        streams[cmap_name] = dim.param.cmap
    return streams


def widget_priority(tup):
    if isinstance(tup[1], SizeDim):
        return 0
    if isinstance(tup[1], AlphaDim):
        return 1
    if isinstance(tup[1], ColorDim):
        return 2
    return 3


class Dimensions:
    def __init__(
        self,
        df,
        n_cols=3,
        **kwargs: (
            DimensionMapper | type[DimensionMapper] | tuple[type[DimensionMapper], dict[str, Any]]
        ),
    ):
        self.dimensions = self.init_dimensions(df, **kwargs)
        self.widgets = []
        self.n_cols = n_cols

    @property
    def streams(self):
        streams = {}
        for v in self.dimensions.values():
            streams = add_to_streams(streams, v)
        return streams

    def init_dimensions(self, df, **kwargs):
        dimensions = {}

        for k, v in kwargs.items():
            if isinstance(v, DimensionMapper):
                mapper = v
            else:
                kwargs = {"df": df, "name": k}
                v_ = v
                if isinstance(v, tuple):
                    kwargs.update(v[1])
                    v_ = v[0]
                mapper = v_(**kwargs)
            dimensions[k] = mapper

        return dict(sorted(dimensions.items(), key=widget_priority))

    def panel(self):
        if not self.widgets:
            self.widgets = [dimension.panel() for dimension in self.dimensions.values()]
        return organize_widgets(self.widgets, self.n_cols)
