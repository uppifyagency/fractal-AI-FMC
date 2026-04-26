import numpy as np
from plangym.utils import process_frame

from fragile.shaolin.stream_plots import RGB


class MontezumaDisplay:
    def __init__(
        self,
    ):
        self.best_rgb = RGB()
        self.visits = np.zeros((24, 160, 160), dtype=np.int32)
        self.rooms = np.zeros((24, 160, 160))
        self.visited_rooms = []

    def reset(
        self,
    ):
        self.visited_rooms = []
        self.visits = np.zeros((24, 160, 160), dtype=np.int32) * np.nan
        self.rooms = np.zeros((24, 160, 160))

    def send(self, fai):
        best_ix = fai.cum_reward.argmax().cpu().item()
        best_rgb = fai.rgb[best_ix]
        self.best_rgb.send(best_rgb)

        observ = fai.observ.cpu().numpy().astype(np.int64)
        observ[:, 0] /= 2
        room_ix = observ[:, 2]
        self.visits[observ[:, 2], observ[:, 1], observ[:, 0]] = np.where(
            np.isnan(self.visits[observ[:, 2], observ[:, 1], observ[:, 0]]),
            1,
            self.visits[observ[:, 2], observ[:, 1], observ[:, 0]] + 1,
        )
        for ix in np.unique(room_ix):
            if ix not in self.visited_rooms:
                self.visited_rooms.append(ix)
                batch_ix = np.argmin(room_ix == ix)
                self.rooms[ix] = process_frame(fai.rgb[batch_ix][50:], mode="L")
