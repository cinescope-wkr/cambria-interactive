"""Principled cinematic optics path.

This path is opt-in and leaves the default RL optics path unchanged.
"""

from __future__ import annotations

from typing import Any, Callable, Self, Tuple

import torch
import torch.nn.functional as F
from hydra_config import HydraContainerConfig, config_wrapper

from cambrian.eyes.eye import MjCambrianEye
from cambrian.eyes.optics import MjCambrianOpticsEye, MjCambrianOpticsEyeConfig


@config_wrapper
class MjCambrianOpticalGenomeConstraintsConfig(HydraContainerConfig):
    abbe_number_range: Tuple[float, float] = (20.0, 65.0)
    focus_distance_range: Tuple[float, float] = (0.25, 12.0)
    lens_thickness_max: float = 0.05
    curvature_profile_abs_max: Tuple[float, float, float] = (0.12, 0.06, 0.02)


@config_wrapper
class MjCambrianOpticalGenomeConfig(HydraContainerConfig):
    abbe_number: float = 50.0
    focus_distance: float = 2.5
    lens_thickness: float = 0.01
    curvature_profile: Tuple[float, float, float] = (0.02, -0.005, 0.001)
    depth_psf_bins: int = 6


@config_wrapper
class MjCambrianCinematicOpticsEyeConfig(MjCambrianOpticsEyeConfig):
    instance: Callable[[Self, str], "MjCambrianCinematicOpticsEye"]
    aperture: Any = None
    cinematic_enabled: bool = True
    optical_genome: Any = None
    optical_genome_constraints: Any = None


class MjCambrianCinematicOpticsEye(MjCambrianOpticsEye):
    """Optics eye with physically controlled dispersion, distortion, and depth blur."""

    _LAMBDA_D = 587.6e-9
    _LAMBDA_F = 486.1e-9
    _LAMBDA_C = 656.3e-9

    def __init__(self, config: MjCambrianCinematicOpticsEyeConfig, name: str):
        super().__init__(config, name)
        self._config: MjCambrianCinematicOpticsEyeConfig
        self._cinematic_enabled = self._cfg_get(config, "cinematic_enabled", False)
        self._optical_genome = self._cfg_get(config, "optical_genome", {})
        self._constraints = self._cfg_get(config, "optical_genome_constraints", {})

    @staticmethod
    def _cfg_get(obj, key: str, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def set_focus_distance(self, focus_distance: float) -> None:
        """Mutable focus hook for rack-focus evaluation scripts."""
        if isinstance(self._optical_genome, dict):
            self._optical_genome["focus_distance"] = focus_distance
        else:
            self._optical_genome.focus_distance = focus_distance

    def _clamp_optical_genome(self) -> dict[str, Any]:
        abbe_range = self._cfg_get(
            self._constraints, "abbe_number_range", (20.0, 65.0)
        )
        focus_range = self._cfg_get(
            self._constraints, "focus_distance_range", (0.25, 12.0)
        )
        lens_thickness_max = float(
            self._cfg_get(self._constraints, "lens_thickness_max", 0.05)
        )
        curvature_max = self._cfg_get(
            self._constraints, "curvature_profile_abs_max", (0.12, 0.06, 0.02)
        )

        abbe_number = float(self._cfg_get(self._optical_genome, "abbe_number", 50.0))
        focus_distance = float(
            self._cfg_get(self._optical_genome, "focus_distance", 2.5)
        )
        lens_thickness = float(
            self._cfg_get(self._optical_genome, "lens_thickness", 0.01)
        )
        curvature_profile = list(
            self._cfg_get(
                self._optical_genome, "curvature_profile", (0.02, -0.005, 0.001)
            )
        )
        depth_psf_bins = int(self._cfg_get(self._optical_genome, "depth_psf_bins", 6))

        clamped_curvature = []
        for coeff, max_abs in zip(curvature_profile, curvature_max):
            clamped_curvature.append(
                max(-float(max_abs), min(float(coeff), float(max_abs)))
            )

        return {
            "abbe_number": max(abbe_range[0], min(abbe_number, abbe_range[1])),
            "focus_distance": max(focus_range[0], min(focus_distance, focus_range[1])),
            "lens_thickness": max(0.0, min(lens_thickness, lens_thickness_max)),
            "curvature_profile": tuple(clamped_curvature),
            "depth_psf_bins": max(2, depth_psf_bins),
        }

    def _channel_refractive_indices(self, abbe_number: float) -> torch.Tensor:
        wavelengths = torch.tensor(
            self._config.wavelengths, device=self._pupil.device, dtype=torch.float64
        )
        n_d = float(self._config.refractive_index)
        delta_n = max((n_d - 1.0) / max(abbe_number, 1e-6), 1e-6)
        slope = delta_n / (self._LAMBDA_C - self._LAMBDA_F)
        refractive = n_d + slope * (self._LAMBDA_D - wavelengths)
        return refractive.to(torch.float32)

    def _apply_curvature_distortion(
        self,
        image: torch.Tensor,
        *,
        refractive_indices: torch.Tensor,
        lens_thickness: float,
        curvature_profile: Tuple[float, float, float],
    ) -> torch.Tensor:
        image_nchw = image.permute(2, 0, 1).unsqueeze(0)
        _, _, h, w = image_nchw.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=image.device),
            torch.linspace(-1.0, 1.0, w, device=image.device),
            indexing="ij",
        )
        radius = torch.sqrt(xx.square() + yy.square()).clamp(min=1e-6)

        # The polynomial defines the surface sag s(r); refraction follows ds/dr.
        coeffs = torch.tensor(curvature_profile, device=image.device, dtype=image.dtype)
        powers = torch.arange(1, coeffs.numel() + 1, device=image.device)
        ds_dr = torch.zeros_like(radius)
        for coeff, power in zip(coeffs, powers):
            ds_dr = ds_dr + (2.0 * power) * coeff * torch.pow(radius, 2.0 * power - 1.0)

        focal_length = float(sum(self._config.focal) / 2.0)
        deviations = (
            (refractive_indices.to(image.device) - 1.0).view(3, 1, 1)
            * lens_thickness
            * ds_dr.unsqueeze(0)
            / max(focal_length, 1e-6)
        )

        base_grid = torch.stack((xx, yy), dim=-1).unsqueeze(0).repeat(3, 1, 1, 1)
        warped = base_grid.clone()
        warped[..., 0] = xx.unsqueeze(0) * (1.0 + deviations)
        warped[..., 1] = yy.unsqueeze(0) * (1.0 + deviations)

        channels = image_nchw.squeeze(0).unsqueeze(1)
        distorted = F.grid_sample(
            channels, warped, mode="bilinear", padding_mode="border", align_corners=True
        )
        return distorted.squeeze(1).permute(1, 2, 0)

    def _build_disk_kernel_bank(
        self, coc_centers: torch.Tensor, refractive_indices: torch.Tensor
    ) -> torch.Tensor:
        focal_ref = float(self._config.refractive_index - 1.0)
        focal_scales = focal_ref / torch.clamp(refractive_indices - 1.0, min=1e-6)
        radii = torch.clamp(
            coc_centers[:, None] * focal_scales[None, :], min=0.35
        )  # [bins, 3]
        max_radius = float(torch.max(radii).item())
        kernel_size = max(3, int(2 * round(max_radius * 2.5) + 1))
        if kernel_size % 2 == 0:
            kernel_size += 1

        coords = torch.linspace(
            -(kernel_size // 2),
            kernel_size // 2,
            kernel_size,
            device=radii.device,
            dtype=radii.dtype,
        )
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        rr2 = xx.square() + yy.square()
        rr2 = rr2.view(1, 1, kernel_size, kernel_size)

        radii2 = radii.view(-1, 3, 1, 1).square()
        edge = 0.75
        kernels = torch.sigmoid((radii2 - rr2) / edge)
        kernels = kernels / torch.clamp(
            kernels.sum(dim=(-1, -2), keepdim=True), min=1e-6
        )
        return kernels.view(-1, 1, kernel_size, kernel_size)

    def _apply_depth_dependent_psf(
        self,
        image: torch.Tensor,
        depth_map: torch.Tensor,
        *,
        refractive_indices: torch.Tensor,
        focus_distance: float,
        depth_psf_bins: int,
    ) -> torch.Tensor:
        coc_pixels = self._circle_of_confusion_pixels(
            depth_map,
            focus_distance=focus_distance,
            focal_length=float(sum(self._config.focal) / 2.0),
            f_number=float(self._config.f_stop),
        )
        max_coc = float(torch.max(coc_pixels).item())
        if max_coc <= 0.5:
            return image

        coc_centers = torch.linspace(
            0.0, max_coc, depth_psf_bins, device=image.device, dtype=image.dtype
        )
        kernels = self._build_disk_kernel_bank(
            coc_centers.to(refractive_indices.dtype), refractive_indices
        ).to(image.dtype)

        input_nchw = image.permute(2, 0, 1).unsqueeze(0)
        blurred = F.conv2d(
            input_nchw, kernels, padding=kernels.shape[-1] // 2, groups=3
        )
        blurred = blurred.view(1, 3, depth_psf_bins, *image.shape[:2]).squeeze(0)
        blurred = blurred.permute(1, 0, 2, 3)  # [bins, 3, h, w]

        bin_width = max(max_coc / max(depth_psf_bins - 1, 1), 1e-6)
        weights = torch.relu(
            1.0
            - torch.abs(coc_pixels.unsqueeze(0) - coc_centers[:, None, None]) / bin_width
        )
        if depth_psf_bins > 1:
            sharp_weight = torch.clamp(1.0 - coc_pixels / bin_width, min=0.0, max=1.0)
            weights[0] = torch.maximum(weights[0], sharp_weight)
        weights = weights / torch.clamp(weights.sum(dim=0, keepdim=True), min=1e-6)

        out = (blurred * weights.unsqueeze(1)).sum(dim=0)
        return out.permute(1, 2, 0)

    def step(self, obs=None):
        if obs is not None:
            image, depth = obs
        else:
            image, depth = self._renderer.render()

        if not self._cinematic_enabled:
            return super().step(obs=(image, depth))

        genome = self._clamp_optical_genome()
        depth_map = self._sanitize_depth_map(depth)
        refractive_indices = self._channel_refractive_indices(genome["abbe_number"]).to(
            image.device
        )

        image = self._apply_curvature_distortion(
            image,
            refractive_indices=refractive_indices,
            lens_thickness=genome["lens_thickness"],
            curvature_profile=genome["curvature_profile"],
        )
        image = self._apply_depth_dependent_psf(
            image,
            depth_map,
            refractive_indices=refractive_indices,
            focus_distance=genome["focus_distance"],
            depth_psf_bins=genome["depth_psf_bins"],
        )

        if self._config.scale_intensity:
            image *= self._scaling_intensity

        image = self._crop(torch.clamp(image, 0.0, 1.0))
        return MjCambrianEye.step(self, obs=image)
