"""Utilities for recording agent-centric observation streams."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import imageio
import numpy as np
import torch
import torch.nn.functional as F

from cambrian.renderer.renderer import MjCambrianRendererSaveMode
from cambrian.utils.logger import get_logger

try:
    from PIL import Image, ImageDraw
except ImportError:  # pragma: no cover - optional dependency
    Image = None
    ImageDraw = None


def _as_float_rgb_tensor(frame: Any) -> torch.Tensor:
    if isinstance(frame, dict):
        tiled = [_as_float_rgb_tensor(value) for key, value in sorted(frame.items())]
        return tile_frames(tiled)

    if isinstance(frame, np.ndarray):
        tensor = torch.from_numpy(frame)
    elif isinstance(frame, torch.Tensor):
        tensor = frame.detach()
    else:
        raise TypeError(f"Unsupported frame type: {type(frame)!r}")

    tensor = tensor.float().cpu()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(-1)
    elif tensor.ndim == 3 and tensor.shape[0] in {1, 3} and tensor.shape[-1] not in {
        1,
        3,
    }:
        tensor = tensor.permute(1, 2, 0)

    if tensor.ndim != 3:
        raise ValueError(f"Expected a 2D or 3D frame tensor, got shape {tensor.shape}.")

    if tensor.shape[-1] == 1:
        tensor = tensor.repeat(1, 1, 3)
    elif tensor.shape[-1] > 3:
        tensor = tensor[..., :3]

    max_value = float(tensor.max()) if tensor.numel() > 0 else 1.0
    if max_value > 1.0:
        tensor = tensor / 255.0

    return tensor.clamp(0.0, 1.0)


def _apply_orientation(
    tensor: torch.Tensor, orientation_mode: str = "raw"
) -> torch.Tensor:
    if orientation_mode == "raw":
        return tensor
    if orientation_mode == "presentation":
        return torch.flipud(tensor)
    raise ValueError(
        f"Unknown orientation mode: {orientation_mode}. Expected raw|presentation."
    )


def resize_frame(
    frame: Any,
    output_size: Optional[Sequence[int]],
    *,
    orientation_mode: str = "raw",
) -> torch.Tensor:
    tensor = _apply_orientation(
        _as_float_rgb_tensor(frame),
        orientation_mode=orientation_mode,
    )
    if output_size is None:
        return tensor

    height, width = map(int, output_size)
    resized = F.interpolate(
        tensor.permute(2, 0, 1).unsqueeze(0),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0)


def tile_frames(frames: Sequence[Any], *, padding: int = 8) -> torch.Tensor:
    tensors = [_as_float_rgb_tensor(frame) for frame in frames if frame is not None]
    if not tensors:
        raise ValueError("Cannot tile an empty frame list.")

    if len(tensors) == 1:
        return tensors[0]

    cols = math.ceil(math.sqrt(len(tensors)))
    rows = math.ceil(len(tensors) / cols)
    max_h = max(int(t.shape[0]) for t in tensors)
    max_w = max(int(t.shape[1]) for t in tensors)
    canvas = torch.zeros(
        (
            rows * max_h + max(rows - 1, 0) * padding,
            cols * max_w + max(cols - 1, 0) * padding,
            3,
        ),
        dtype=torch.float32,
    )
    for idx, tensor in enumerate(tensors):
        row, col = divmod(idx, cols)
        y = row * (max_h + padding)
        x = col * (max_w + padding)
        canvas[y : y + tensor.shape[0], x : x + tensor.shape[1]] = tensor
    return canvas


def tile_named_frames(
    frames: Sequence[Tuple[str, Any]],
    *,
    padding: int = 8,
) -> Tuple[torch.Tensor, List[Tuple[str, int, int]]]:
    tensors = [
        (label, _as_float_rgb_tensor(frame))
        for label, frame in frames
        if frame is not None
    ]
    if not tensors:
        raise ValueError("Cannot tile an empty frame list.")

    if len(tensors) == 1:
        label, tensor = tensors[0]
        return tensor, [(label, 8, 8)]

    cols = math.ceil(math.sqrt(len(tensors)))
    rows = math.ceil(len(tensors) / cols)
    max_h = max(int(t.shape[0]) for _, t in tensors)
    max_w = max(int(t.shape[1]) for _, t in tensors)
    canvas = torch.zeros(
        (
            rows * max_h + max(rows - 1, 0) * padding,
            cols * max_w + max(cols - 1, 0) * padding,
            3,
        ),
        dtype=torch.float32,
    )
    labels: List[Tuple[str, int, int]] = []
    for idx, (label, tensor) in enumerate(tensors):
        row, col = divmod(idx, cols)
        y = row * (max_h + padding)
        x = col * (max_w + padding)
        canvas[y : y + tensor.shape[0], x : x + tensor.shape[1]] = tensor
        labels.append((label, x + 8, y + 8))
    return canvas, labels


def compose_side_by_side(
    left: Any,
    right: Any,
    *,
    gap: int = 16,
    left_size: Optional[Sequence[int]] = None,
    right_size: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    left_tensor = resize_frame(left, left_size)
    right_tensor = resize_frame(right, right_size)
    target_height = max(left_tensor.shape[0], right_tensor.shape[0])
    if left_tensor.shape[0] != target_height:
        left_tensor = resize_frame(left_tensor, (target_height, left_tensor.shape[1]))
    if right_tensor.shape[0] != target_height:
        right_tensor = resize_frame(right_tensor, (target_height, right_tensor.shape[1]))

    canvas = torch.zeros(
        (target_height, left_tensor.shape[1] + gap + right_tensor.shape[1], 3),
        dtype=torch.float32,
    )
    canvas[:, : left_tensor.shape[1]] = left_tensor
    canvas[:, left_tensor.shape[1] + gap :] = right_tensor
    return canvas


def annotate_frame(
    frame: Any,
    *,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    eye_labels: Optional[Sequence[Tuple[str, int, int]]] = None,
) -> np.ndarray:
    tensor = _as_float_rgb_tensor(frame)
    image = (tensor * 255.0).to(torch.uint8).numpy()
    if Image is None or ImageDraw is None or (
        title is None and subtitle is None and not eye_labels
    ):
        return image

    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    text_y = 12
    if title:
        draw.text((12, text_y), title, fill=(255, 255, 255))
        text_y += 16
    if subtitle:
        draw.text((12, text_y), subtitle, fill=(200, 220, 230))
    if eye_labels:
        for label, x, y in eye_labels:
            draw.text((x, y), label, fill=(225, 235, 240))
    return np.asarray(pil_image)


def _iter_eye_observations(
    eye: Any,
    *,
    eye_name: Optional[str] = None,
) -> Iterable[Tuple[str, Any]]:
    current_name = getattr(eye, "name", None)
    prev_obs = getattr(eye, "prev_obs", None)
    if prev_obs is not None and (eye_name is None or eye_name == current_name):
        yield current_name or "eye", prev_obs
        return

    if hasattr(eye, "eyes"):
        for nested_eye in eye.eyes.values():
            yield from _iter_eye_observations(nested_eye, eye_name=eye_name)


def extract_agent_vision_frame(
    env: Any,
    *,
    agent_name: Optional[str] = None,
    eye_name: Optional[str] = None,
    output_size: Optional[Sequence[int]] = None,
    padding: int = 8,
    orientation_mode: str = "raw",
    labels: str = "off",
) -> Any:
    agents = env.agents
    if agent_name is None:
        trainable = [
            agent for agent in agents.values() if getattr(agent, "trainable", False)
        ]
        agent = trainable[0] if trainable else next(iter(agents.values()))
    else:
        agent = agents[agent_name]

    observations: List[Tuple[str, Any]] = []
    for eye in agent.eyes.values():
        observations.extend(list(_iter_eye_observations(eye, eye_name=eye_name)))

    if not observations:
        raise ValueError(
            f"No eye observations found for agent={agent.name!r}, eye={eye_name!r}."
        )

    eye_labels = None
    if len(observations) > 1:
        frame, eye_labels = tile_named_frames(observations, padding=padding)
    else:
        _, observation = observations[0]
        frame = _as_float_rgb_tensor(observation)
        eye_labels = [(observations[0][0], 8, 8)] if labels == "debug" else None

    source_height, source_width = frame.shape[0], frame.shape[1]
    frame = resize_frame(frame, output_size, orientation_mode=orientation_mode)

    if labels == "debug":
        if output_size is not None and eye_labels:
            scale_y = frame.shape[0] / max(source_height, 1)
            scale_x = frame.shape[1] / max(source_width, 1)
            eye_labels = [
                (label, int(x * scale_x), int(y * scale_y)) for label, x, y in eye_labels
            ]
        return annotate_frame(frame, eye_labels=eye_labels)

    if labels not in {"off", "minimal", "debug"}:
        raise ValueError(f"Unknown label mode: {labels}. Expected off|minimal|debug.")
    return frame


class MjCambrianVisionRecorder:
    def __init__(
        self,
        *,
        fps: int = 50,
        output_size: Optional[Sequence[int]] = None,
        save_mode: MjCambrianRendererSaveMode = MjCambrianRendererSaveMode.WEBP,
        orientation_mode: str = "raw",
        label_mode: str = "off",
    ):
        self._fps = fps
        self._output_size = tuple(output_size) if output_size is not None else None
        self._save_mode = save_mode
        self._orientation_mode = orientation_mode
        self._label_mode = label_mode
        self._frames: List[np.ndarray] = []

    def capture_frame(
        self,
        frame: Any,
        *,
        title: Optional[str] = None,
        subtitle: Optional[str] = None,
        apply_orientation: bool = True,
    ) -> None:
        tensor = resize_frame(
            frame,
            self._output_size,
            orientation_mode=self._orientation_mode if apply_orientation else "raw",
        )
        annotated = annotate_frame(
            tensor,
            title=title if self._label_mode != "off" else None,
            subtitle=subtitle if self._label_mode != "off" else None,
        )
        self._frames.append(annotated)

    def capture_agent(
        self,
        env: Any,
        *,
        agent_name: Optional[str] = None,
        eye_name: Optional[str] = None,
        padding: int = 8,
        title: Optional[str] = None,
        subtitle: Optional[str] = None,
    ) -> None:
        frame = extract_agent_vision_frame(
            env,
            agent_name=agent_name,
            eye_name=eye_name,
            output_size=self._output_size,
            padding=padding,
            orientation_mode=self._orientation_mode,
            labels=self._label_mode,
        )
        annotated_title = title if self._label_mode in {"minimal", "debug"} else None
        annotated_subtitle = (
            subtitle if self._label_mode in {"minimal", "debug"} else None
        )
        if self._label_mode == "debug":
            self._frames.append(
                annotate_frame(frame, title=annotated_title, subtitle=annotated_subtitle)
            )
        else:
            self.capture_frame(
                frame,
                title=annotated_title,
                subtitle=annotated_subtitle,
                apply_orientation=False,
            )

    def save(self, path: Path | str) -> None:
        if not self._frames:
            get_logger().warning("Vision recorder buffer is empty. Nothing to save.")
            return

        path = Path(path)
        frames = np.stack(self._frames, axis=0)
        duration = 1000 / self._fps

        if self._save_mode & MjCambrianRendererSaveMode.MP4:
            mp4 = path.with_suffix(".mp4")
            try:
                imageio.mimwrite(mp4, frames, fps=self._fps)
            except TypeError:
                get_logger().error(
                    "imageio is not compiled with ffmpeg. "
                    "Install with `pip install imageio[ffmpeg]` for mp4 export."
                )
            else:
                get_logger().debug(f"Saved vision visualization at {mp4}")

        if self._save_mode & MjCambrianRendererSaveMode.PNG:
            png = path.with_suffix(".png")
            imageio.imwrite(png, frames[-1])
            get_logger().debug(f"Saved vision visualization at {png}")

        if self._save_mode & MjCambrianRendererSaveMode.GIF:
            gif = path.with_suffix(".gif")
            imageio.mimwrite(gif, frames, loop=0, duration=duration)
            get_logger().debug(f"Saved vision visualization at {gif}")

        if self._save_mode & MjCambrianRendererSaveMode.WEBP:
            webp = path.with_suffix(".webp")
            imageio.mimwrite(webp, frames, fps=self._fps, lossless=True)
            get_logger().debug(f"Saved vision visualization at {webp}")

    @property
    def frames(self) -> List[np.ndarray]:
        return self._frames
