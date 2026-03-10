"""Physically grounded sensor model for the opt-in cinematic path."""

from __future__ import annotations

from typing import Optional

import torch
from hydra_config import HydraContainerConfig, config_wrapper


@config_wrapper
class MjCambrianSensorConfig(HydraContainerConfig):
    enabled: bool = True
    photon_full_well: float = 6000.0
    read_noise_std: float = 3.0
    dark_current: float = 2.0
    fixed_pattern_std: float = 0.01
    analog_gain: float = 1.0


@config_wrapper
class MjCambrianCinematicFxConfig(HydraContainerConfig):
    enabled: bool = True
    sensor: MjCambrianSensorConfig


class MjCambrianCinematicFx:
    """Sensor-end Poisson-Gaussian noise model for cinematic rendering."""

    def __init__(self, config: MjCambrianCinematicFxConfig):
        self._config = config
        self._fixed_pattern: Optional[torch.Tensor] = None

    @staticmethod
    def _get(obj, key: str, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def apply(self, image: torch.Tensor, *, throughput: float = 1.0) -> torch.Tensor:
        if not self._get(self._config, "enabled", False):
            return image

        sensor_cfg = self._get(self._config, "sensor", {})
        if not self._get(sensor_cfg, "enabled", False):
            return image

        image_nchw, squeeze = self._as_nchw(image)

        photon_full_well = float(self._get(sensor_cfg, "photon_full_well", 6000.0))
        read_noise_std = float(self._get(sensor_cfg, "read_noise_std", 3.0))
        dark_current = float(self._get(sensor_cfg, "dark_current", 2.0))
        fixed_pattern_std = float(self._get(sensor_cfg, "fixed_pattern_std", 0.01))
        analog_gain = float(self._get(sensor_cfg, "analog_gain", 1.0))

        throughput = max(float(throughput), 1e-6)
        electrons_mean = (
            torch.clamp(image_nchw, 0.0, 1.0)
            * photon_full_well
            * analog_gain
            * throughput
        ) + dark_current

        shot = torch.poisson(torch.clamp(electrons_mean, min=0.0))
        read = torch.randn_like(shot) * read_noise_std

        if self._fixed_pattern is None or self._fixed_pattern.shape != image_nchw.shape:
            # Fixed-pattern noise is persistent across frames for a given sensor shape.
            self._fixed_pattern = (
                torch.randn_like(image_nchw) * (fixed_pattern_std * photon_full_well)
            ).detach()

        electrons = shot + read + self._fixed_pattern
        out = electrons / max(photon_full_well * analog_gain, 1e-6)
        return torch.clamp(self._from_nchw(out, squeeze), 0.0, 1.0)

    def _as_nchw(self, image: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if image.ndim == 3:
            return image.permute(2, 0, 1).unsqueeze(0), True
        return image, False

    def _from_nchw(self, image: torch.Tensor, squeeze: bool) -> torch.Tensor:
        if squeeze:
            return image.squeeze(0).permute(1, 2, 0)
        return image
