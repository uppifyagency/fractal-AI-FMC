import time

import pandas as pd
import panel as pn
import param


class FaiRunner(param.Parameterized):
    is_running = param.Boolean(default=False)

    def __init__(self, fai, n_steps, plot=None):
        super().__init__()
        self.reset_btn = pn.widgets.Button(icon="restore", button_type="primary")
        self.play_btn = pn.widgets.Button(icon="player-play", button_type="primary")
        self.pause_btn = pn.widgets.Button(icon="player-pause", button_type="primary")
        self.step_btn = pn.widgets.Button(name="Step", button_type="primary")
        self.progress = pn.indicators.Progress(
            name="Progress", value=0, width=600, max=n_steps, bar_color="primary"
        )
        self.sleep_val = pn.widgets.FloatInput(value=0.01, width=60)
        self.table = pn.widgets.DataFrame(height=50, width=600)
        self.fai = fai
        self.n_steps = n_steps
        self.curr_step = 0
        self.plot = plot

    @param.depends("reset_btn.value")
    def on_reset_click(self):
        self.fai.reset()
        self.curr_step = 0
        self.progress.value = 1
        self.curr_step = 0
        self.play_btn.disabled = False
        self.pause_btn.disabled = True
        self.step_btn.disabled = False
        self.is_running = False
        self.progress.bar_color = "primary"
        summary = pd.DataFrame(self.fai.summary(), index=[0])
        self.table.value = summary
        if self.plot is not None:
            self.plot.send(self.fai)

    @param.depends("play_btn.value")
    def on_play_click(self):
        self.play_btn.disabled = True
        self.pause_btn.disabled = False
        self.is_running = True

    @param.depends("pause_btn.clicks")
    def on_pause_click(self):
        self.play_btn.disabled = False
        self.pause_btn.disabled = True
        self.is_running = False

    @param.depends("step_btn.value")
    def on_step_click(self):
        self.is_running = True
        self.run()
        self.is_running = False

    def run(self):
        if not self.is_running:
            return
        self.fai.step_tree()
        self.curr_step += 1
        self.progress.value = self.curr_step
        if self.curr_step >= self.n_steps:
            self.is_running = False
            self.progress.bar_color = "success"
            self.step_btn.disabled = True
            self.play_btn.disabled = True
            self.pause_btn.disabled = True
        summary = pd.DataFrame(self.fai.summary(), index=[0])
        self.table.value = summary
        if self.fai.oobs.sum().cpu().item() == self.fai.n_walkers - 1:
            self.is_running = False
            self.progress.bar_color = "danger"
        if self.plot is not None:
            self.plot.send(self.fai)
        time.sleep(self.sleep_val.value)

    def __panel__(self):
        pn.state.add_periodic_callback(self.run, period=1)
        return pn.Column(
            self.table,
            self.progress,
            pn.Row(
                self.play_btn,
                self.pause_btn,
                self.reset_btn,
                self.step_btn,
                pn.pane.Markdown("**Sleep**"),
                self.sleep_val,
            ),
            self.on_play_click,
            self.on_pause_click,
            self.on_reset_click,
            self.on_step_click,
        )
