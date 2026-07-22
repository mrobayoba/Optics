"""Analytical Fresnel diffraction by a slit or a straight edge.

The implementation follows the normalized Fresnel integrals used in
``Difraccion_Fresnel_Clotoide.pdf``:

``C(u) = integral(cos(pi*t**2/2), 0, u)``
``S(u) = integral(sin(pi*t**2/2), 0, u)``

SciPy returns these values in ``(S, C)`` order. All physical lengths passed to
this module are expressed in metres.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.special import fresnel

import diffraction_engine as shared_engine
import fresnel_config as config

__all__ = [
    "FresnelCase",
    "IlluminationKind",
    "FresnelReport",
    "effective_distance",
    "geometric_magnification",
    "screen_fresnel_scale",
    "screen_fresnel_coordinate",
    "slit_boundary_coordinates",
    "cornu_point",
    "slit_field",
    "slit_intensity",
    "edge_field",
    "edge_intensity",
    "fresnel_number",
    "characteristic_screen_length",
    "evaluate_fresnel_regime",
]


class FresnelCase(str, Enum):
    """Supported analytical Fresnel obstacles."""

    SLIT = "slit"
    EDGE = "edge"


class IlluminationKind(str, Enum):
    """Wavefront incident on the aperture plane."""

    PLANE = "plane"
    POINT = "point"


@dataclass(frozen=True)
class FresnelReport:
    """Geometry and regime information accompanying a simulation."""

    case: FresnelCase
    illumination: IlluminationKind
    wavelength_medium: float
    distance: float
    source_distance: float | None
    effective_distance: float
    magnification: float
    characteristic_length: float
    fresnel_number: float | None
    near_field_limit: float
    regime: str

    @property
    def is_near_field(self) -> bool | None:
        """Whether a finite slit satisfies the configured near-field limit."""
        if self.fresnel_number is None:
            return None
        return self.fresnel_number >= self.near_field_limit


def _positive_finite(value: float | None, name: str) -> float:
    if value is None:
        raise ValueError(f"{name} is required.")
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite value.")
    return value


def _illumination(value: IlluminationKind | str) -> IlluminationKind:
    try:
        return IlluminationKind(value)
    except ValueError as exc:
        choices = ", ".join(item.value for item in IlluminationKind)
        raise ValueError(
            f"Unknown illumination {value!r}; use one of: {choices}."
        ) from exc


def _case(value: FresnelCase | str) -> FresnelCase:
    try:
        return FresnelCase(value)
    except ValueError as exc:
        choices = ", ".join(item.value for item in FresnelCase)
        raise ValueError(
            f"Unknown Fresnel case {value!r}; use one of: {choices}."
        ) from exc


def effective_distance(
    distance: float,
    illumination: IlluminationKind | str = IlluminationKind.PLANE,
    source_distance: float | None = None,
) -> float:
    """Return ``D`` for a plane wave or ``dD/(d+D)`` for a point source."""
    distance = _positive_finite(distance, "distance")
    kind = _illumination(illumination)
    if kind == IlluminationKind.PLANE:
        return distance
    source = _positive_finite(source_distance, "source_distance")
    return source * distance / (source + distance)


def geometric_magnification(
    distance: float,
    illumination: IlluminationKind | str = IlluminationKind.PLANE,
    source_distance: float | None = None,
) -> float:
    """Return geometrical aperture-to-screen magnification."""
    distance = _positive_finite(distance, "distance")
    kind = _illumination(illumination)
    if kind == IlluminationKind.PLANE:
        return 1.0
    source = _positive_finite(source_distance, "source_distance")
    return (source + distance) / source


def screen_fresnel_scale(
    wavelength_vacuum: float,
    distance: float,
    *,
    n: float = 1.0,
    illumination: IlluminationKind | str = IlluminationKind.PLANE,
    source_distance: float | None = None,
) -> float:
    """Return the multiplier mapping a screen coordinate to Fresnel ``u``."""
    wavelength = shared_engine.medium_wavelength(wavelength_vacuum, n)
    effective = effective_distance(
        distance, illumination, source_distance
    )
    magnification = geometric_magnification(
        distance, illumination, source_distance
    )
    return float(np.sqrt(2.0 / (wavelength * effective)) / magnification)


def screen_fresnel_coordinate(
    coordinate: np.ndarray | float,
    wavelength_vacuum: float,
    distance: float,
    *,
    n: float = 1.0,
    illumination: IlluminationKind | str = IlluminationKind.PLANE,
    source_distance: float | None = None,
) -> np.ndarray:
    """Map observation-plane coordinates to dimensionless Fresnel ``u``."""
    values = np.asarray(coordinate, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("coordinate contains non-finite values.")
    scale = screen_fresnel_scale(
        wavelength_vacuum,
        distance,
        n=n,
        illumination=illumination,
        source_distance=source_distance,
    )
    return values * scale


def slit_boundary_coordinates(
    coordinate: np.ndarray | float,
    slit_width: float,
    wavelength_vacuum: float,
    distance: float,
    *,
    n: float = 1.0,
    illumination: IlluminationKind | str = IlluminationKind.PLANE,
    source_distance: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the two Cornu coordinates delimiting a centered slit."""
    width = _positive_finite(slit_width, "slit_width")
    center_u = screen_fresnel_coordinate(
        coordinate,
        wavelength_vacuum,
        distance,
        n=n,
        illumination=illumination,
        source_distance=source_distance,
    )
    wavelength = shared_engine.medium_wavelength(wavelength_vacuum, n)
    effective = effective_distance(
        distance, illumination, source_distance
    )
    aperture_scale = np.sqrt(2.0 / (wavelength * effective))
    dimensionless_width = width * aperture_scale
    return (
        center_u - dimensionless_width / 2.0,
        center_u + dimensionless_width / 2.0,
    )


def cornu_point(u: np.ndarray | float) -> np.ndarray:
    """Return the complex Cornu coordinate ``C(u) + i*S(u)``."""
    values = np.asarray(u, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("u contains non-finite values.")
    sine, cosine = fresnel(values)
    return np.asarray(cosine + 1j * sine, dtype=complex)


def slit_field(
    coordinate: np.ndarray | float,
    slit_width: float,
    wavelength_vacuum: float,
    distance: float,
    *,
    n: float = 1.0,
    illumination: IlluminationKind | str = IlluminationKind.PLANE,
    source_distance: float | None = None,
) -> np.ndarray:
    """Return slit field normalized to the unobstructed incident field."""
    lower, upper = slit_boundary_coordinates(
        coordinate,
        slit_width,
        wavelength_vacuum,
        distance,
        n=n,
        illumination=illumination,
        source_distance=source_distance,
    )
    return (cornu_point(upper) - cornu_point(lower)) / np.sqrt(2.0)


def slit_intensity(
    coordinate: np.ndarray | float,
    slit_width: float,
    wavelength_vacuum: float,
    distance: float,
    *,
    n: float = 1.0,
    illumination: IlluminationKind | str = IlluminationKind.PLANE,
    source_distance: float | None = None,
) -> np.ndarray:
    """Return slit intensity ``I/I0`` from the Cornu chord."""
    field = slit_field(
        coordinate,
        slit_width,
        wavelength_vacuum,
        distance,
        n=n,
        illumination=illumination,
        source_distance=source_distance,
    )
    return np.asarray(np.abs(field) ** 2, dtype=float)


def edge_field(
    coordinate: np.ndarray | float,
    wavelength_vacuum: float,
    distance: float,
    *,
    n: float = 1.0,
    illumination: IlluminationKind | str = IlluminationKind.PLANE,
    source_distance: float | None = None,
) -> np.ndarray:
    """Return edge field with positive coordinates inside the shadow."""
    u = screen_fresnel_coordinate(
        coordinate,
        wavelength_vacuum,
        distance,
        n=n,
        illumination=illumination,
        source_distance=source_distance,
    )
    return (complex(0.5, 0.5) - cornu_point(u)) / np.sqrt(2.0)


def edge_intensity(
    coordinate: np.ndarray | float,
    wavelength_vacuum: float,
    distance: float,
    *,
    n: float = 1.0,
    illumination: IlluminationKind | str = IlluminationKind.PLANE,
    source_distance: float | None = None,
) -> np.ndarray:
    """Return straight-edge intensity ``I/I0``."""
    field = edge_field(
        coordinate,
        wavelength_vacuum,
        distance,
        n=n,
        illumination=illumination,
        source_distance=source_distance,
    )
    return np.asarray(np.abs(field) ** 2, dtype=float)


def fresnel_number(
    slit_width: float,
    wavelength_vacuum: float,
    distance: float,
    *,
    n: float = 1.0,
    illumination: IlluminationKind | str = IlluminationKind.PLANE,
    source_distance: float | None = None,
) -> float:
    """Return ``N_F = slit_width**2 / (4*lambda_medium*D_effective)``."""
    width = _positive_finite(slit_width, "slit_width")
    wavelength = shared_engine.medium_wavelength(wavelength_vacuum, n)
    effective = effective_distance(
        distance, illumination, source_distance
    )
    return width**2 / (4.0 * wavelength * effective)


def characteristic_screen_length(
    wavelength_vacuum: float,
    distance: float,
    *,
    n: float = 1.0,
    illumination: IlluminationKind | str = IlluminationKind.PLANE,
    source_distance: float | None = None,
) -> float:
    """Return the screen displacement corresponding to ``Delta u = 1``."""
    return 1.0 / screen_fresnel_scale(
        wavelength_vacuum,
        distance,
        n=n,
        illumination=illumination,
        source_distance=source_distance,
    )


def evaluate_fresnel_regime(
    case: FresnelCase | str,
    wavelength_vacuum: float,
    distance: float,
    *,
    slit_width: float | None = None,
    n: float = 1.0,
    illumination: IlluminationKind | str = IlluminationKind.PLANE,
    source_distance: float | None = None,
    near_field_limit: float = config.NEAR_FIELD_LIMIT.default,
) -> FresnelReport:
    """Validate geometry and return an informational Fresnel report."""
    selected_case = _case(case)
    selected_illumination = _illumination(illumination)
    selected_limit = _positive_finite(
        near_field_limit, "near_field_limit"
    )
    wavelength = shared_engine.medium_wavelength(wavelength_vacuum, n)
    distance = _positive_finite(distance, "distance")
    effective = effective_distance(
        distance, selected_illumination, source_distance
    )
    magnification = geometric_magnification(
        distance, selected_illumination, source_distance
    )
    source = (
        _positive_finite(source_distance, "source_distance")
        if selected_illumination == IlluminationKind.POINT
        else None
    )
    characteristic = characteristic_screen_length(
        wavelength_vacuum,
        distance,
        n=n,
        illumination=selected_illumination,
        source_distance=source,
    )

    number = None
    regime = "edge diffraction"
    if selected_case == FresnelCase.SLIT:
        number = fresnel_number(
            _positive_finite(slit_width, "slit_width"),
            wavelength_vacuum,
            distance,
            n=n,
            illumination=selected_illumination,
            source_distance=source,
        )
        if number < config.FAR_FIELD_LIMIT:
            regime = "far-field limit"
        elif number < config.FRESNEL_TRANSITION_LIMIT:
            regime = "Fresnel transition"
        else:
            regime = "Fresnel near field"

    return FresnelReport(
        case=selected_case,
        illumination=selected_illumination,
        wavelength_medium=wavelength,
        distance=distance,
        source_distance=source,
        effective_distance=effective,
        magnification=magnification,
        characteristic_length=characteristic,
        fresnel_number=number,
        near_field_limit=selected_limit,
        regime=regime,
    )
