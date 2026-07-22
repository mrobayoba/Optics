"""Observation-plane sampling for analytical slit and edge Fresnel patterns."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import fresnel_config as config
import fresnel_engine as engine
import screen_tools as shared_screen

__all__ = [
    "FresnelSimulationResult",
    "simulate_pattern",
    "extract_central_profiles",
]


@dataclass(frozen=True)
class FresnelSimulationResult:
    """Sampled Fresnel intensity and its geometry report."""

    X: np.ndarray
    Y: np.ndarray
    intensity: np.ndarray
    report: engine.FresnelReport


def _orientation(value: str) -> str:
    if value not in ("vertical", "horizontal"):
        raise ValueError("orientation must be 'vertical' or 'horizontal'.")
    return value


def _shadow_side(value: str) -> str:
    if value not in ("positive", "negative"):
        raise ValueError("shadow_side must be 'positive' or 'negative'.")
    return value


def simulate_pattern(
    case: engine.FresnelCase | str,
    wavelength_vacuum: float,
    distance: float,
    *,
    slit_width: float | None = None,
    n: float = 1.0,
    illumination: engine.IlluminationKind | str = (
        engine.IlluminationKind.PLANE
    ),
    source_distance: float | None = None,
    orientation: str = config.DEFAULT_ORIENTATION,
    shadow_side: str = config.DEFAULT_SHADOW_SIDE,
    screen_half_width: float = config.SCREEN_HALF_WIDTH_MM.to_si(
        config.SCREEN_HALF_WIDTH_MM.default
    ),
    resolution: int = int(config.RESOLUTION.default),
    zoom: float = config.ZOOM.default,
    near_field_limit: float = config.NEAR_FIELD_LIMIT.default,
) -> FresnelSimulationResult:
    """Calculate a slit or edge pattern on a centered square camera grid."""
    selected_case = engine.FresnelCase(case)
    selected_orientation = _orientation(orientation)
    selected_shadow = _shadow_side(shadow_side)

    # Validate all propagation inputs before allocating the observation grid.
    report = engine.evaluate_fresnel_regime(
        selected_case,
        wavelength_vacuum,
        distance,
        slit_width=slit_width,
        n=n,
        illumination=illumination,
        source_distance=source_distance,
        near_field_limit=near_field_limit,
    )
    X, Y = shared_screen.observation_grid(
        screen_half_width, resolution, zoom
    )
    coordinate = X if selected_orientation == "vertical" else Y

    if selected_case == engine.FresnelCase.SLIT:
        intensity = engine.slit_intensity(
            coordinate,
            slit_width,
            wavelength_vacuum,
            distance,
            n=n,
            illumination=illumination,
            source_distance=source_distance,
        )
    else:
        signed_coordinate = (
            coordinate if selected_shadow == "positive" else -coordinate
        )
        intensity = engine.edge_intensity(
            signed_coordinate,
            wavelength_vacuum,
            distance,
            n=n,
            illumination=illumination,
            source_distance=source_distance,
        )

    return FresnelSimulationResult(
        X=np.asarray(X, dtype=float),
        Y=np.asarray(Y, dtype=float),
        intensity=np.asarray(intensity, dtype=float),
        report=report,
    )


def extract_central_profiles(
    result: FresnelSimulationResult,
) -> shared_screen.CentralProfiles:
    """Return horizontal and vertical cuts through a Fresnel result."""
    return shared_screen.extract_central_profiles(
        result.X, result.Y, result.intensity
    )
