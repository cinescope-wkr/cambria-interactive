"""Callbacks used during training and/or evaluation."""

import csv
import glob
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.experimental.callbacks import Callback as HydraCallback
from omegaconf import DictConfig
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    EvalCallback,
    ProgressBarCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

from cambrian.envs import MjCambrianEnv
from cambrian.ml.model import MjCambrianModel
from cambrian.utils.logger import get_logger
from cambrian.utils.vision_recording import (
    MjCambrianVisionRecorder,
    compose_side_by_side,
    extract_agent_vision_frame,
    resize_frame,
)


class MjCambrianPlotMonitorCallback(BaseCallback):
    """Should be used with an EvalCallback to plot the evaluation results.

    This callback will take the monitor.csv file produced by the VecMonitor and
    plot the results and save it as an image. Should be passed as the
    `callback_after_eval` for the EvalCallback.

    Args:
        logdir (Path | str): The directory where the evaluation results are stored. The
            evaluations.npz file is expected to be at `<logdir>/<filename>.csv`. The
            resulting plot is going to be stored at
            `<logdir>/evaluations/<filename>.png`.
        filename (Path | str): The filename of the monitor file. The saved file will be
            `<logdir>/<filename>.csv`. And the resulting plot will be saved as
            `<logdir>/evaluations/<filename>.png`.
    """

    parent: EvalCallback

    def __init__(self, logdir: Path | str, filename: Path | str, n_episodes: int = 1):
        self.logdir = Path(logdir)
        self.filename = Path(filename)
        self.filename_csv = self.filename.with_suffix(".csv")
        self.filename_png = self.filename.with_suffix(".png")
        self.evaldir = self.logdir / "evaluations"
        self.evaldir.mkdir(parents=True, exist_ok=True)

        self.n_episodes = n_episodes
        self.n_calls = 0

    def _on_step(self) -> bool:
        if not (self.logdir / self.filename_csv).exists():
            get_logger().warning(f"No {self.filename_csv} file found.")
            return

        # Temporarily set the monitor ext so that the right file is read
        old_ext = Monitor.EXT
        Monitor.EXT = str(self.filename_csv)
        x, y = ts2xy(load_results(self.logdir), "timesteps")
        Monitor.EXT = old_ext
        if len(x) <= 20 or len(y) <= 20:
            get_logger().warning(f"Not enough {self.filename} data to plot.")
            return True
        original_x, original_y = x.copy(), y.copy()

        get_logger().info(f"Plotting {self.filename} results at {self.evaldir}")

        def moving_average(data, window=1):
            return np.convolve(data, np.ones(window), "valid") / window

        n = min(len(y) // 10, 1000)
        y = y.astype(float)

        if self.n_episodes > 1:
            assert len(y) % self.n_episodes == 0, (
                "n_episodes must be a common factor of the"
                f" number of episodes in the {self.filename} data."
            )
            y = y.reshape(-1, self.n_episodes).mean(axis=1)
        else:
            y = moving_average(y, window=n)

        x = moving_average(x, window=n).astype(int)

        # Make sure the x, y are of the same length
        min_len = min(len(x), len(y))
        x, y = x[:min_len], y[:min_len]

        plt.plot(x, y)
        plt.plot(original_x, original_y, color="grey", alpha=0.3)

        plt.xlabel("Number of Timesteps")
        plt.ylabel("Rewards")
        plt.savefig(self.evaldir / self.filename.with_suffix(".png"))
        plt.cla()

        return True


class MjCambrianEvalCallback(EvalCallback):
    """Overwrites the default EvalCallback to support saving visualizations at the same
    time as the evaluation.

    Note:
        Only the first environment is visualized
    """

    def _init_callback(self):
        self.log_path = Path(self.log_path)
        self.n_evals = 0
        self._vision_recorder: MjCambrianVisionRecorder | None = None
        self._vision_record_source: str = "scene"
        self._vision_config: Dict[str, Any] = {}

        # Delete all the existing renders
        for f in glob.glob(str(self.log_path / "vis_*")):
            get_logger().info(f"Deleting {f}")
            Path(f).unlink()

        super()._init_callback()

    def _on_step(self) -> bool:
        # Early exit
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True

        env: MjCambrianEnv = self.eval_env.envs[0].unwrapped
        renderer_config = env.config.renderer
        record_source = getattr(renderer_config, "record_source", "scene")
        use_vision_recording = record_source in {"agent_vision", "side_by_side"}

        # Add some overlays
        # env.overlays["Exp"] = env.config.expname # TODO
        env.overlays["Best Mean Reward"] = f"{self.best_mean_reward:.2f}"
        env.overlays["Total Timesteps"] = f"{self.num_timesteps}"

        # Run the evaluation
        get_logger().info(f"Starting {self.n_eval_episodes} evaluation run(s)...")
        render_for_eval = self.render
        if use_vision_recording and self.render:
            vision_config = getattr(renderer_config, "vision", {}) or {}
            self._vision_record_source = record_source
            self._vision_config = vision_config
            self._vision_recorder = MjCambrianVisionRecorder(
                fps=int(vision_config.get("fps", 50)),
                output_size=vision_config.get("output_size", None),
                save_mode=renderer_config.save_mode,
                orientation_mode=vision_config.get("orientation_mode", "raw"),
                label_mode=vision_config.get("labels", "off"),
            )
            # Disable SB3's internal per-step render; we capture via callback.
            self.render = False
        else:
            self._vision_recorder = None
            self._vision_record_source = "scene"
            self._vision_config = {}

        env.record(self.render)
        try:
            continue_training = super()._on_step()
        finally:
            self.render = render_for_eval

        if self.render:
            # Save the visualization
            filename = Path(f"vis_{self.n_evals}")
            if self._vision_recorder is not None:
                self._vision_recorder.save(self.log_path / filename)
            else:
                env.save(self.log_path / filename)
            env.record(False)

        if self.render:
            # Copy the most recent gif to latest.gif so that we can just watch this file
            for f in self.log_path.glob(str(filename.with_suffix(".*"))):
                shutil.copy(f, f.with_stem("latest"))

        self._vision_recorder = None
        self._vision_record_source = "scene"
        self._vision_config = {}
        self.n_evals += 1
        return continue_training

    def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]):
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
        env: MjCambrianEnv = self.eval_env.envs[0].unwrapped

        if self._vision_recorder is not None:
            self._capture_vision_frame(env)

        # If done, do some logging
        if locals_["done"]:
            run = locals_["episode_counts"][locals_["i"]]
            cumulative_reward = env.stashed_cumulative_reward
            get_logger().info(f"Run {run} done. Cumulative reward: {cumulative_reward}")

        super()._log_success_callback(locals_, globals_)

    def _capture_vision_frame(self, env: MjCambrianEnv) -> None:
        if self._vision_recorder is None:
            return

        vision_config = self._vision_config
        label_mode = vision_config.get("labels", "off")
        agent_name = vision_config.get("agent_name", None)
        eye_name = vision_config.get("eye_name", None)

        if self._vision_record_source == "agent_vision":
            title = None
            subtitle = None
            if label_mode in {"minimal", "debug"}:
                title = "Agent vision"
                target_eye = eye_name or "all eyes"
                target_agent = agent_name or next(
                    iter([a.name for a in env.agents.values() if a.trainable]),
                    next(iter(env.agents.keys())),
                )
                subtitle = (
                    f"{target_agent} | {target_eye} | "
                    f"{vision_config.get('orientation_mode', 'raw')}"
                )
            self._vision_recorder.capture_agent(
                env,
                agent_name=agent_name,
                eye_name=eye_name,
                padding=int(vision_config.get("padding", 8)),
                title=title,
                subtitle=subtitle,
            )
            return

        if self._vision_record_source == "side_by_side":
            scene_frame = env.render()
            vision_frame = extract_agent_vision_frame(
                env,
                agent_name=agent_name,
                eye_name=eye_name,
                output_size=None,
                padding=int(vision_config.get("padding", 8)),
                orientation_mode="raw",
                labels="off",
            )
            scene_tensor = resize_frame(scene_frame, None)
            vision_tensor = resize_frame(vision_frame, None)
            scene_height = int(scene_tensor.shape[0])
            scene_width = int(scene_tensor.shape[1])
            vision_aspect = max(
                float(vision_tensor.shape[1]) / max(float(vision_tensor.shape[0]), 1.0),
                1e-6,
            )
            target_vision_width = int(scene_height * vision_aspect)
            target_vision_width = max(target_vision_width, scene_height // 2)
            target_vision_width = min(target_vision_width, int(scene_width * 0.9))
            frame = compose_side_by_side(
                scene_frame,
                vision_frame,
                right_size=(scene_height, target_vision_width),
            )
            self._vision_recorder.capture_frame(frame)
            return


class MjCambrianGPUUsageCallback(BaseCallback):
    """This callback will log the GPU usage at the end of each evaluation.
    We'll log to a csv."""

    parent: EvalCallback

    def __init__(
        self,
        logdir: Path | str,
        logfile: Path | str = "gpu_usage.csv",
        *,
        verbose: int = 0,
    ):
        super().__init__(verbose)

        self.logdir = Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)

        self.logfile = self.logdir / logfile
        with open(self.logfile, "w") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timesteps",
                    "memory_reserved",
                    "max_memory_reserved",
                    "memory_available",
                ]
            )

    def _on_step(self) -> bool:
        if torch.cuda.is_available():
            # Get the GPU usage, log it and save it to the file
            device = torch.cuda.current_device()
            memory_reserved = torch.cuda.memory_reserved(device)
            max_memory_reserved = torch.cuda.max_memory_reserved(device)
            memory_available = torch.cuda.get_device_properties(device).total_memory

            # Log to the output file
            with open(self.logfile, "a") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        self.num_timesteps,
                        memory_reserved,
                        max_memory_reserved,
                        memory_available,
                    ]
                )

            # Log to stdout
            if self.verbose > 0:
                get_logger().debug(subprocess.getoutput("nvidia-smi"))
                get_logger().debug(torch.cuda.memory_summary())

        return True


class MjCambrianSavePolicyCallback(BaseCallback):
    """Should be used with an EvalCallback to save the policy.

    This callback will save the policy at the end of each evaluation. Should be passed
    as the `callback_after_eval` for the EvalCallback.

    Args:
        logdir (Path | str): The directory to store the generated visualizations. The
            resulting visualizations are going to be stored at
            `<logdir>/evaluations/visualization.gif`.
    """

    parent: EvalCallback

    def __init__(
        self,
        logdir: Path | str,
        *,
        verbose: int = 0,
    ):
        super().__init__(verbose)

        self.logdir = Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)

        self.model: MjCambrianModel = None

    def _on_step(self) -> bool:
        self.model.save_policy(self.logdir)

        return True


class MjCambrianPeriodicCheckpointCallback(BaseCallback):
    """Save resumable model checkpoints after each evaluation.

    This callback is intended to run inside `callback_after_eval`, so its `_on_step`
    is called once per evaluation event rather than every environment step.
    """

    parent: EvalCallback

    def __init__(
        self,
        logdir: Path | str,
        *,
        checkpoint_dirname: str = "checkpoints",
        prefix: str = "checkpoint",
        max_keep: int = 0,
        verbose: int = 0,
    ):
        super().__init__(verbose)

        self.logdir = Path(logdir)
        self.checkpoint_dir = self.logdir / checkpoint_dirname
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.max_keep = int(max_keep)
        self.index_path = self.checkpoint_dir / "index.csv"
        if not self.index_path.exists():
            with open(self.index_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["eval_index", "timesteps", "checkpoint_zip"])

    def _checkpoint_base(self, timesteps: int) -> Path:
        return self.checkpoint_dir / f"{self.prefix}_{timesteps:09d}"

    def _prune_old_checkpoints(self) -> None:
        if self.max_keep <= 0:
            return

        checkpoints = sorted(self.checkpoint_dir.glob(f"{self.prefix}_*.zip"))
        if len(checkpoints) <= self.max_keep:
            return

        to_remove = checkpoints[: len(checkpoints) - self.max_keep]
        for checkpoint in to_remove:
            get_logger().info(f"Removing old checkpoint {checkpoint}")
            checkpoint.unlink(missing_ok=True)

    def _on_step(self) -> bool:
        timesteps = int(getattr(self.parent, "num_timesteps", self.num_timesteps))
        eval_index = int(getattr(self.parent, "n_evals", -1))
        checkpoint_base = self._checkpoint_base(timesteps)
        checkpoint_zip = checkpoint_base.with_suffix(".zip")

        get_logger().info(f"Saving periodic checkpoint to {checkpoint_zip}")
        self.model.save(str(checkpoint_base))

        latest_zip = self.checkpoint_dir / "latest.zip"
        shutil.copy2(checkpoint_zip, latest_zip)
        with open(self.checkpoint_dir / "latest.txt", "w") as f:
            f.write(str(checkpoint_zip))

        with open(self.index_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([eval_index, timesteps, checkpoint_zip.name])

        self._prune_old_checkpoints()
        return True


class MjCambrianProgressBarCallback(ProgressBarCallback):
    """Overwrite the default progress bar callback to flush the pbar on deconstruct."""

    def __del__(self):
        """This string will restore the terminal back to its original state."""
        if hasattr(self, "pbar"):
            print("\x1b[?25h")


class MjCambrianCallbackListWithSharedParent(CallbackList):
    def __init__(self, callbacks: Iterable[BaseCallback] | Dict[str, BaseCallback]):
        if isinstance(callbacks, dict):
            callbacks = callbacks.values()

        self.callbacks = []
        super().__init__(list(callbacks))

    @property
    def parent(self):
        return getattr(self.callbacks[0], "parent", None)

    @parent.setter
    def parent(self, parent):
        for cb in self.callbacks:
            cb.parent = parent


# ==================


class MjCambrianSaveConfigCallback(HydraCallback):
    """This callback will save the resolved hydra config to the logdir."""

    def on_run_start(self, config: DictConfig, **kwargs):
        self._save_config(config)

    def on_multirun_start(self, config: DictConfig, **kwargs):
        self._save_config(config)

    def _save_config(self, config: DictConfig):
        from omegaconf import OmegaConf

        config.logdir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config, config.logdir / "full.yaml")
