from functools import partial

import holoviews as hv
from holoviews.streams import Pipe
import numpy as np
import panel as pn
import plangym
from plangym.utils import process_frame

from fragile.actions import UniformDtSampler
from fragile.core import FaiRunner
from fragile.shaolin.stream_plots import Image, RGB
from fragile.videogames import aggregate_visits, MontezumaTree


PYRAMID = [
    [-1, -1, -1, 0, 1, 2, -1, -1, -1],
    [-1, -1, 3, 4, 5, 6, 7, -1, -1],
    [-1, 8, 9, 10, 11, 12, 13, 14, -1],
    [15, 16, 17, 18, 19, 20, 21, 22, 23],
]
EMPTY_ROOMS = [
    (0, 0),
    (0, 1),
    (0, 2),
    (0, 6),
    (0, 7),
    (0, 8),
    (1, 0),
    (1, 1),
    (1, 7),
    (1, 8),
    (2, 0),
    (2, 8),
]


def get_rooms_xy(pyramid=None) -> np.ndarray:
    """Get the tuple that encodes the provided room."""
    pyramid = pyramid if pyramid is not None else PYRAMID
    n_rooms = max(max(row) for row in pyramid) + 1
    rooms_xy = []
    for room in range(n_rooms):
        for y, loc in enumerate(PYRAMID):
            if room in loc:
                room_xy = [loc.index(room), y]
                rooms_xy.append(room_xy)
                break
    return np.array(rooms_xy)


def get_pyramid_layout(room_h=160, room_w=160, channels=3, pyramid=None, empty_rooms=None):
    pyramid = pyramid if pyramid is not None else PYRAMID
    ph, pw = len(pyramid), len(pyramid[0])
    all_rooms = np.zeros((room_h * ph, room_w * pw, channels))
    return set_empty_rooms(all_rooms, empty_rooms, height=room_h, width=room_w)


def set_empty_rooms(all_rooms, empty_rooms=None, height=160, width=160):
    empty_rooms = empty_rooms if empty_rooms is not None else EMPTY_ROOMS
    val = np.array([255, 255, 255], dtype=np.uint8)
    for i, j in empty_rooms:
        all_rooms[i * height : (i + 1) * height, j * width : (j + 1) * width] = val
    return all_rooms


def draw_rooms(env, pyramid_layout=None, height=160, width=160):
    pyramid_layout = pyramid_layout if pyramid_layout is not None else get_pyramid_layout()
    for n_room, room in env.rooms.items():
        i, j = env.get_room_xy(n_room)
        coord_x, coord_x1 = j * width, (j + 1) * width
        coord_y, coord_y1 = i * height, (i + 1) * height
        pyramid_layout[coord_x:coord_x1, coord_y:coord_y1, :] = room
    return pyramid_layout


def to_pyramid_coords(observ, room_xy, width=160, height=160):
    x, y, room = (
        observ[:, 0].astype(np.int64),
        observ[:, 1].astype(np.int64),
        observ[:, 2].astype(np.int64),
    )
    room_coords = room_xy[room]
    offset_coords = room_coords * np.array([width, height])
    return np.array([x, y]).T + offset_coords


def to_plot_coords(room_coords, width=160, height=160):
    plot_x = (room_coords[:, 0]) / (width - 1) - 0.5
    plot_y = ((height - 1) - room_coords[:, 1]) / (height - 1) - 0.5
    return plot_x, plot_y


def draw_pyramid(data, pyramid_layout=None):
    return hv.RGB(draw_rooms(data, pyramid_layout)).opts(
        width=1440, height=640, xaxis=None, yaxis=None
    )


def draw_tree_pyramid(data, max_x: int = 1440, max_y: int = 640, room_xy=None):
    room_xy = room_xy if room_xy is not None else get_rooms_xy()
    if not data:
        return hv.Segments(bgcolor=None) * hv.Scatter(bgcolor=None)
    observ = data.observ.cpu().numpy().astype(np.int64)
    room_coords = to_pyramid_coords(observ, room_xy)
    parents = data.parent.cpu().numpy()
    room_coords = room_coords.astype(np.float64)
    room_coords[:, 0] /= float(data.env.gym_env._x_repeat)
    room_coords = room_coords.astype(np.int64)
    plot_x, plot_y = to_plot_coords(room_coords, width=max_x, height=max_y)
    segs = plot_x[parents], plot_y[parents], plot_x, plot_y
    edges = hv.Segments(segs).opts(line_color="white", bgcolor=None)
    nodes = hv.Scatter((plot_x, plot_y)).opts(
        size=2, bgcolor=None, color="red", line_width=0.01, xaxis=None, yaxis=None
    )
    return edges * nodes


def draw_tree_best_room(data, width=160, height=160):
    if not data:
        return hv.Segments(bgcolor=None) * hv.Scatter(bgcolor=None)
    room_coords = data.observ.cpu().numpy().astype(np.int64)
    room = room_coords[:, 2][data.cum_reward.argmax().cpu().item()]
    room_ix = room_coords[:, 2] == room
    parents = data.parent.cpu().numpy()[room_ix]
    room_coords = room_coords.astype(np.float64)
    room_coords[:, 0] /= float(data.env.gym_env._x_repeat)
    room_coords = room_coords.astype(np.int64)
    room_coords = room_coords[room_ix]
    plot_x, plot_y = to_plot_coords(room_coords, width=width, height=height)
    segs = plot_x[parents], plot_y[parents], plot_x, plot_y
    edges = hv.Segments(segs).opts(line_color="black", bgcolor=None)
    nodes = hv.Scatter((plot_x, plot_y)).opts(size=2, bgcolor=None, color="red")
    return edges * nodes


class MontezumaDisplay:
    def __init__(
        self,
    ):
        self.best_rgb = RGB()
        self.room_grey = Image(cmap="greys")
        self.visits_image = Image(alpha=0.7, xaxis=None, yaxis=None, cmap="fire", bgcolor=None)
        self.visits = np.zeros((24, 160, 160), dtype=np.int32) * np.nan
        self.rooms = np.zeros((24, 160, 160))
        self.visited_rooms = []
        self.pipe_tree = Pipe()
        self.room_pipe = Pipe()
        self._curr_best = -1
        # self.tree_best_room = hv.DynamicMap(draw_tree_best_room, streams=[self.pipe_tree])
        self.tree_pyramid = hv.DynamicMap(
            partial(draw_tree_pyramid, room_xy=get_rooms_xy()), streams=[self.pipe_tree]
        )
        self.pyramid = hv.DynamicMap(draw_pyramid, streams=[self.room_pipe])

    def reset(self, fai):  # noqa: ARG002
        self.visited_rooms = []
        self.visits = np.zeros((24, 160, 160), dtype=np.int32) * np.nan
        self.rooms = np.zeros((24, 160, 160))

    def send(self, fai):
        best_ix = fai.cum_reward.argmax().cpu().item()
        best_rgb = fai.img[best_ix]
        if best_ix != self._curr_best:
            self.best_rgb.send(best_rgb)
            self._curr_best = best_ix

        observ = fai.observ.cpu().numpy().astype(np.float64)
        observ[:, 0] /= int(fai.env.gym_env._x_repeat)
        observ = observ.astype(np.int64)
        room_ix = observ[:, 2]
        best_room_ix = room_ix[best_ix]
        for ix in np.unique(room_ix):
            if ix not in self.visited_rooms:
                # batch_ix =
                room_img = fai.img[room_ix == ix][0][50:]
                fai.env.gym_env.rooms[ix] = room_img  # fai.env.gym_env.rooms.get(ix, room_img)
                self.visited_rooms.append(ix)
                self.room_pipe.send(fai.env.gym_env)
                self.rooms[ix] = process_frame(room_img, mode="L").copy()

        self.room_grey.send(self.rooms[best_room_ix])

        visits = fai.visits[best_room_ix][None]
        visits = aggregate_visits(visits, block_size=fai.agg_block_size, upsample=True)[0]
        visits[visits == 0] = np.nan
        self.visits_image.send(visits)
        self.pipe_tree.send(fai)

    def __panel__(self):
        return pn.Column(
            pn.Row(
                self.best_rgb.plot,
                self.room_grey.plot * self.visits_image.plot,
                # self.room_grey.plot * self.tree_best_room,
            ),
            self.pyramid * self.tree_pyramid,
        )


def run_serve_montezuma(**kwargs):
    hv.extension("bokeh")
    pn.extension("tabulator", theme="dark", throttled=True)
    env = plangym.make(
        "PlanMontezuma-v0",
        obs_type="coords",
        return_image=True,
        frameskip=3,
        check_death=True,
        episodic_life=False,
        n_workers=5,
        ray=True,
    )

    n_walkers = 100000
    plot = MontezumaDisplay()
    fai = MontezumaTree(
        max_walkers=n_walkers,
        env=env,
        dt_sampler=UniformDtSampler(min_dt=1, max_dt=5),
        device="cpu",
        min_leafs=15,
        start_walkers=15,
        state_shape=(11,),
    )
    runner = FaiRunner(fai, 1000000, plot=plot)
    dashboard = pn.panel(pn.Column(runner, plot))
    template = pn.template.BootstrapTemplate(
        title="Montezuma's revenge",
        main=pn.panel(dashboard),
        # sidebar=pn.panel(app.plot_sidebar),
        # main_layout=None,
    ).servable()
    return pn.serve(template, **kwargs)
