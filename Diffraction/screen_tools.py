"""Observation-plane sampling utilities for Fraunhofer diffraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

import diffraction_config as config
import diffraction_engine as engine

__all__ = [
    "SimulationResult",
    "CentralProfiles",
    "observation_grid",
    "simulate_pattern",
    "extract_central_profiles",
]


@dataclass(frozen=True)
class SimulationResult:
    """Sampled observation coordinates, intensity, and validity report."""

    X: np.ndarray
    Y: np.ndarray
    intensity: np.ndarray
    far_field: engine.FarFieldReport


@dataclass(frozen=True)
class CentralProfiles:
    """Horizontal and vertical intensity cuts through the optical axis."""

    x: np.ndarray
    horizontal: np.ndarray
    y: np.ndarray
    vertical: np.ndarray


def _positive_finite(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite value.")
    return value


def observation_grid(
    screen_half_width: float = config.SCREEN_HALF_WIDTH_MM.to_si(
        config.SCREEN_HALF_WIDTH_MM.default
    ),
    resolution: int = int(config.RESOLUTION.default),
    zoom: float = config.ZOOM.default,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a square, centered observation-plane grid in metres.

    ``screen_half_width`` is the half-width at ``zoom=1``.  Increasing zoom
    reduces the visible physical extent while preserving the sample count.
    """
    half_width = _positive_finite(screen_half_width, "screen_half_width")
    zoom = _positive_finite(zoom, "zoom")
    if isinstance(resolution, bool) or int(resolution) != resolution:
        raise ValueError("resolution must be an integer.")
    resolution = int(resolution)
    if resolution < 3:
        raise ValueError("resolution must be at least 3.")

    visible_half_width = half_width / zoom
    axis = np.linspace(-visible_half_width, visible_half_width, resolution)
    return np.meshgrid(axis, axis)


def simulate_pattern(
    apertures: Sequence[engine.Aperture],
    wavelength_vacuum: float,
    distance: float,
    n: float = 1.0,
    screen_half_width: float = config.SCREEN_HALF_WIDTH_MM.to_si(
        config.SCREEN_HALF_WIDTH_MM.default
    ),
    resolution: int = int(config.RESOLUTION.default),
    zoom: float = config.ZOOM.default,
    max_fresnel_number: float = config.MAX_FRESNEL_NUMBER.default,
    normalize: bool = True,
) -> SimulationResult:
    """Build a guarded 2D Fraunhofer intensity pattern.

    The validity gate intentionally runs before :func:`observation_grid`, so
    an invalid configuration does not allocate a grid or evaluate a field.
    """
    items = tuple(apertures)
    report = engine.require_far_field(
        items,
        wavelength_vacuum,
        distance,
        n=n,
        max_fresnel_number=max_fresnel_number,
    )
    X, Y = observation_grid(screen_half_width, resolution, zoom)
    intensity, _ = engine.fraunhofer_intensity(
        X,
        Y,
        items,
        wavelength_vacuum,
        distance,
        n=n,
        max_fresnel_number=max_fresnel_number,
        normalize=normalize,
    )
    return SimulationResult(X, Y, intensity, report)


def extract_central_profiles(
    X: np.ndarray,
    Y: np.ndarray,
    intensity: np.ndarray,
) -> CentralProfiles:
    """Extract horizontal and vertical cuts nearest the optical axis."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    if X.ndim != 2 or Y.ndim != 2 or intensity.ndim != 2:
        raise ValueError("X, Y, and intensity must be two-dimensional arrays.")
    if X.shape != Y.shape or X.shape != intensity.shape:
        raise ValueError("X, Y, and intensity must have matching shapes.")
    if 0 in intensity.shape:
        raise ValueError("Input arrays cannot be empty.")

    center_row = int(np.argmin(np.abs(Y[:, 0])))
    center_column = int(np.argmin(np.abs(X[0, :])))
    return CentralProfiles(
        x=X[center_row, :].copy(),
        horizontal=intensity[center_row, :].copy(),
        y=Y[:, center_column].copy(),
        vertical=intensity[:, center_column].copy(),
    )
