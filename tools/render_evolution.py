"""Render checkpoint lineage as a presentation-ready timelapse."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import List

import torch

from cambrian import MjCambrianConfig, MjCambrianTrainer, run_hydra
from cambrian.ml.model import MjCambrianModel
from cambrian.utils import evaluate_policy
from cambrian.utils.vision_recording import (
    MjCambrianVisionRecorder,
    compose_side_by_side,
    extract_agent_vision_frame,
    resize_frame,
)


def _resolve_checkpoints(
    checkpoints: List[str] | None,
    checkpoint_glob: str | None,
) -> List[Path]:
    resolved: List[Path] = []
    if checkpoints:
        resolved.extend(Path(checkpoint).expanduser() for checkpoint in checkpoints)
    if checkpoint_glob:
        resolved.extend(Path(path).expanduser() for path in sorted(glob.glob(checkpoint_glob)))

    unique: List[Path] = []
    seen = set()
    for checkpoint in resolved:
        checkpoint = checkpoint.resolve()
        if checkpoint in seen:
            continue
        seen.add(checkpoint)
        unique.append(checkpoint)
    if not unique:
        raise FileNotFoundError("No checkpoints resolved. Pass --checkpoints or --checkpoint-glob.")
    return unique


def _load_checkpoint(
    model: MjCambrianModel,
    eval_env,
    checkpoint: Path,
) -> MjCambrianModel:
    if checkpoint.suffix == ".zip":
        return MjCambrianModel.load(str(checkpoint), env=eval_env)
    if checkpoint.name == "policy.pt":
        model.load_policy(checkpoint.parent)
        return model
    if checkpoint.suffix == ".pt":
        saved_state_dict = torch.load(checkpoint)
        policy_state_dict = model.policy.state_dict()
        for key in list(saved_state_dict.keys()):
            if key not in policy_state_dict or saved_state_dict[key].shape != policy_state_dict[key].shape:
                del saved_state_dict[key]
        model.policy.load_state_dict(saved_state_dict, strict=False)
        return model
    raise ValueError(f"Unsupported checkpoint format: {checkpoint}")


def _render_checkpoint_segment(
    recorder: MjCambrianVisionRecorder,
    *,
    eval_env,
    model: MjCambrianModel,
    checkpoint: Path,
    num_runs: int,
    agent_name: str | None,
    eye_name: str | None,
    label: str,
    layout: str,
):
    cambrian_env = eval_env.envs[0].unwrapped

    def step_callback(env):
        scene_frame = env.render()
        vision_frame = extract_agent_vision_frame(
            env,
            agent_name=agent_name,
            eye_name=eye_name,
            output_size=None,
        )
        if layout == "side_by_side":
            scene_tensor = resize_frame(scene_frame, None)
            vision_tensor = resize_frame(vision_frame, None)
            scene_height = int(scene_tensor.shape[0])
            scene_width = int(scene_tensor.shape[1])
            vision_aspect = max(float(vision_tensor.shape[1]) / max(float(vision_tensor.shape[0]), 1.0), 1e-6)
            target_vision_width = int(scene_height * vision_aspect)
            target_vision_width = max(target_vision_width, scene_height // 2)
            target_vision_width = min(target_vision_width, int(scene_width * 0.9))
            frame = compose_side_by_side(
                scene_frame,
                vision_frame,
                right_size=(scene_height, target_vision_width),
            )
        elif layout == "vision_only":
            frame = vision_frame
        elif layout == "scene_only":
            frame = scene_frame
        else:
            raise ValueError(
                f"Unknown layout: {layout}. Expected side_by_side|vision_only|scene_only."
            )
        recorder.capture_frame(
            frame,
            title=label,
            subtitle=checkpoint.name,
        )
        return True

    evaluate_policy(
        eval_env,
        model,
        num_runs,
        record_kwargs=None,
        step_callback=step_callback,
    )


def main(
    config: MjCambrianConfig,
    *,
    checkpoints: List[str] | None,
    checkpoint_glob: str | None,
    output_name: str,
    num_runs: int | None,
    agent_name: str | None,
    eye_name: str | None,
    fps: int,
    layout: str,
):
    trainer = MjCambrianTrainer(config)
    eval_env = trainer._make_env(config.eval_env, 1, monitor="eval_monitor.csv")
    model = trainer._make_model(eval_env)
    resolved_checkpoints = _resolve_checkpoints(checkpoints, checkpoint_glob)
    runs = int(num_runs or config.eval_env.n_eval_episodes)

    recorder = MjCambrianVisionRecorder(
        fps=fps,
        output_size=None,
        save_mode=config.eval_env.renderer.save_mode,
        orientation_mode="presentation",
        label_mode="minimal",
    )

    for idx, checkpoint in enumerate(resolved_checkpoints):
        loaded_model = _load_checkpoint(model, eval_env, checkpoint)
        _render_checkpoint_segment(
            recorder,
            eval_env=eval_env,
            model=loaded_model,
            checkpoint=checkpoint,
            num_runs=runs,
            agent_name=agent_name,
            eye_name=eye_name,
            label=f"lineage {idx:02d}",
            layout=layout,
        )

    recorder.save(Path(config.expdir) / output_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="*", default=None)
    parser.add_argument("--checkpoint-glob", type=str, default=None)
    parser.add_argument("--output-name", type=str, default="evolution")
    parser.add_argument("--num-runs", type=int, default=None)
    parser.add_argument("--agent-name", type=str, default=None)
    parser.add_argument("--eye-name", type=str, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--layout",
        type=str,
        default="side_by_side",
        choices=["side_by_side", "vision_only", "scene_only"],
    )
    run_hydra(main, parser=parser)
