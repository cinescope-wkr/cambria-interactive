"""Opt-in cinematic renderer path.

The default MjCambrianRenderer path is unchanged. This class is only selected
when a cinematic renderer config is used.
"""

from __future__ import annotations

from typing import Any

from hydra_config import HydraContainerConfig, config_wrapper

from cambrian.renderer.cinematic_fx import MjCambrianCinematicFx, MjCambrianCinematicFxConfig
from cambrian.renderer.renderer import MjCambrianRenderer, MjCambrianRendererConfig


@config_wrapper
class MjCambrianCinematicConfig(HydraContainerConfig):
    enabled: bool = True
    camera_preset: str = "agent_fp_shallow_dof"
    overlay_mode: str = "minimal"
    optical_genome_constraints: Any
    sensor: MjCambrianCinematicFxConfig


@config_wrapper
class MjCambrianCinematicRendererConfig(MjCambrianRendererConfig):
    cinematic: MjCambrianCinematicConfig


class MjCambrianCinematicRenderer(MjCambrianRenderer):
    """Renderer subclass that applies post FX only when cinematic is enabled."""

    def __init__(self, config: MjCambrianCinematicRendererConfig):
        super().__init__(config)
        self._config: MjCambrianCinematicRendererConfig = config
        cinematic_cfg = (
            config.cinematic
            if hasattr(config, "cinematic")
            else config.get("cinematic", {})
        )
        fx_cfg = (
            cinematic_cfg.sensor
            if hasattr(cinematic_cfg, "sensor")
            else cinematic_cfg.get("sensor", {})
        )
        self._cinematic_enabled = (
            cinematic_cfg.enabled
            if hasattr(cinematic_cfg, "enabled")
            else bool(cinematic_cfg.get("enabled", False))
        )
        self._cinematic_fx = MjCambrianCinematicFx(fx_cfg)

    def render(self, *args, **kwargs):
        frame = super().render(*args, **kwargs)
        if frame is None or not self._cinematic_enabled:
            return frame

        # Base renderer can return:
        # - rgb tensor
        # - [rgb, depth]
        # Keep return shape contract intact.
        if isinstance(frame, list):
            if len(frame) == 0:
                return frame
            frame[0] = self._cinematic_fx.apply(frame[0])
            return frame

        return self._cinematic_fx.apply(frame)
