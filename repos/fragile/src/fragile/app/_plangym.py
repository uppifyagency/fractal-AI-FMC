import holoviews as hv
import panel as pn
import plangym

from fragile.core import FaiRunner, FractalTree
from fragile.shaolin.stream_plots import RGB


hv.extension("bokeh")
pn.extension("tabulator")


class PlanGymDisplay:
    def __init__(
        self,
    ):
        self.best_img = RGB()
        self._curr_best = -1

    def reset(self, fai):  # noqa: ARG002
        return

    def send(self, fai):
        best_ix = fai.cum_reward.argmax().cpu().item()
        best_img = fai.img[best_ix]
        if best_ix != self._curr_best:
            self.best_img.send(best_img)
            self._curr_best = best_ix

    def __panel__(self):
        return pn.Column(
            pn.Row(
                self.best_img.plot,
                # self.room_grey.plot * self.tree_best_room,
            ),
        )


def main():
    env = plangym.make(
        domain_name="walker",
        task_name="stand",
        obs_type="coords",
        return_image=True,
        frameskip=1,
        n_workers=10,
        ray=True,
    )

    n_walkers = 10000
    plot = PlanGymDisplay()
    fai = FractalTree(
        max_walkers=n_walkers, env=env, device="cpu", min_leafs=500, start_walkers=500
    )
    runner = FaiRunner(fai, 1000000, plot=plot)
    pn.panel(pn.Column(runner, plot)).servable()


# main()
