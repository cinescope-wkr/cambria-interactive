"""Render a cinematic evaluation while sweeping focus distance over time."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from cambrian import MjCambrianConfig, MjCambrianTrainer, run_hydra
from cambrian.eyes.cinematic_optics import MjCambrianCinematicOpticsEye
from cambrian.ml.model import MjCambrianModel
from cambrian.utils import evaluate_policy


def _iter_cinematic_eyes(eye) -> Iterable[MjCambrianCinematicOpticsEye]:
    if isinstance(eye, MjCambrianCinematicOpticsEye):
        yield eye
        return
    if hasattr(eye, "eyes"):
        for nested_eye in eye.eyes.values():
            yield from _iter_cinematic_eyes(nested_eye)


def _set_focus_distance(env, focus_distance: float) -> None:
    for agent in env.agents.values():
        for eye in agent.eyes.values():
            for cinematic_eye in _iter_cinematic_eyes(eye):
                cinematic_eye.set_focus_distance(focus_distance)


def main(
    config: MjCambrianConfig,
    *,
    focus_near: float,
    focus_far: float,
    model_path: str | None,
    output_name: str,
):
    trainer = MjCambrianTrainer(config)
    eval_env = trainer._make_env(config.eval_env, 1, monitor="eval_monitor.csv")
    model = trainer._make_model(eval_env)
    if model_path:
        model = MjCambrianModel.load(model_path, env=eval_env)

    cambrian_env = eval_env.envs[0].unwrapped
    total_steps = max(
        int(config.eval_env.n_eval_episodes) * int(cambrian_env.max_episode_steps), 1
    )

    def step_callback(env):
        progress = min(float(env.num_timesteps) / total_steps, 1.0)
        focus_distance = focus_near + (focus_far - focus_near) * progress
        _set_focus_distance(env, focus_distance)
        return True

    _set_focus_distance(cambrian_env, focus_near)
    evaluate_policy(
        eval_env,
        model,
        int(config.eval_env.n_eval_episodes),
        record_kwargs={
            "path": Path(config.expdir) / output_name,
            "save_mode": config.eval_env.renderer.save_mode,
        },
        step_callback=step_callback,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--focus-near", type=float, default=0.5)
    parser.add_argument("--focus-far", type=float, default=8.0)
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional stable-baselines model path. If omitted, use instantiated model.",
    )
    parser.add_argument("--output-name", type=str, default="rack_focus")
    run_hydra(main, parser=parser)
