import flogging
import holoviews as hv
import panel as pn

from fragile.actions import GaussianForce
from fragile.benchmarks import Sphere
from fragile.core import FaiRunner
from fragile.functions import FunctionTree
from fragile.shaolin.streaming_fai import InteractiveFai


hv.extension("bokeh")
pn.extension("tabulator", theme="dark")


def main():
    flogging.setup(allow_trailing_dot=True)
    n_dims = 2
    env = Sphere(n_dims)

    n_walkers = 10000
    fai = FunctionTree(
        max_walkers=n_walkers,
        env=env,
        # dt_sampler=UniformDtSampler(min_dt=1, max_dt=3),
        policy=GaussianForce(std=2.5, min=-10.0, max=10.0),
        # policy=RandomGaussianPolicy(std=25., min=-100.0, max=100.0),
        device="cpu",
        min_leafs=20,
        start_walkers=50,
        minimize=True,
        state_shape=(n_dims,),
    )
    plot = InteractiveFai(fai)
    runner = FaiRunner(fai, 1000000, plot=plot)
    return pn.panel(pn.Column(runner, plot)).servable()


# if __name__ == "__main__":
main()
