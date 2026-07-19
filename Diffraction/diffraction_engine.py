"""Analytical Fraunhofer diffraction physics.

All lengths are expressed in metres.  The input wavelength is the vacuum
wavelength; propagation uses ``wavelength_medium = wavelength_vacuum / n``.
The aperture is illuminated by a uniform, coherent plane wave.

The module deliberately separates analytical Fourier-domain amplitudes from
screen sampling.  A high-level screen calculation always checks the far-field
criterion before evaluating the field.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np
from scipy.special import j1

import diffraction_config as config

__all__ = [
    "ApertureKind",
    "Aperture",
    "FarFieldReport",
    "FarFieldError",
    "medium_wavelength",
    "aperture_support_radius",
    "evaluate_far_field",
    "require_far_field",
    "spatial_frequencies",
    "aperture_amplitude",
    "composite_amplitude",
    "fraunhofer_field",
    "fraunhofer_intensity",
    "slit_first_minimum",
    "rectangle_first_minima",
    "circular_first_minimum",
]


class ApertureKind(str, Enum):
    """Supported analytical aperture primitives."""

    SLIT = "slit"
    RECTANGLE = "rectangle"
    CIRCLE = "circle"


@dataclass(frozen=True)
class Aperture:
    """One axis-aligned aperture primitive.

    A slit is represented as the finite narrow rectangle used in the source
    PDF, with ``width`` across the slit and ``height`` along it.  Keeping the
    slit finite makes both its two-dimensional pattern and far-field support
    well defined.
    """

    kind: ApertureKind | str
    width: float | None = None
    height: float | None = None
    radius: float | None = None
    center_x: float = 0.0
    center_y: float = 0.0

    def __post_init__(self) -> None:
        try:
            kind = ApertureKind(self.kind)
        except ValueError as exc:
            choices = ", ".join(item.value for item in ApertureKind)
            raise ValueError(
                f"Unknown aperture kind {self.kind!r}; use one of: {choices}."
            ) from exc
        object.__setattr__(self, "kind", kind)

        if not np.isfinite(self.center_x) or not np.isfinite(self.center_y):
            raise ValueError("Aperture center coordinates must be finite.")

        if kind in (ApertureKind.SLIT, ApertureKind.RECTANGLE):
            _require_positive_finite(self.width, "width")
            _require_positive_finite(self.height, "height")
        else:
            _require_positive_finite(self.radius, "radius")

    @classmethod
    def slit(
        cls,
        width: float,
        length: float,
        center_x: float = 0.0,
        center_y: float = 0.0,
    ) -> "Aperture":
        """Create a finite slit of ``width`` by ``length``."""
        return cls(ApertureKind.SLIT, width, length, None, center_x, center_y)

    @classmethod
    def rectangle(
        cls,
        width: float,
        height: float,
        center_x: float = 0.0,
        center_y: float = 0.0,
    ) -> "Aperture":
        """Create a rectangular opening."""
        return cls(
            ApertureKind.RECTANGLE, width, height, None, center_x, center_y
        )

    @classmethod
    def circle(
        cls,
        radius: float,
        center_x: float = 0.0,
        center_y: float = 0.0,
    ) -> "Aperture":
        """Create a circular opening."""
        return cls(ApertureKind.CIRCLE, None, None, radius, center_x, center_y)

    @property
    def area(self) -> float:
        """Geometrical area of the opening."""
        if self.kind == ApertureKind.CIRCLE:
            assert self.radius is not None
            return float(np.pi * self.radius**2)
        assert self.width is not None and self.height is not None
        return float(self.width * self.height)

    @property
    def support_radius(self) -> float:
        """Largest distance from the optical axis to this aperture."""
        if self.kind == ApertureKind.CIRCLE:
            assert self.radius is not None
            return float(np.hypot(self.center_x, self.center_y) + self.radius)
        assert self.width is not None and self.height is not None
        far_x = abs(self.center_x) + self.width / 2.0
        far_y = abs(self.center_y) + self.height / 2.0
        return float(np.hypot(far_x, far_y))


@dataclass(frozen=True)
class FarFieldReport:
    """Numerical interpretation of the Fraunhofer validity condition."""

    is_valid: bool
    fresnel_number: float
    max_fresnel_number: float
    required_distance: float
    distance: float
    wavelength_medium: float
    support_radius: float

    @property
    def safety_ratio(self) -> float:
        """Allowed Fresnel number divided by the actual value."""
        if self.fresnel_number == 0.0:
            return np.inf
        return self.max_fresnel_number / self.fresnel_number


class FarFieldError(ValueError):
    """Raised when a requested configuration is outside the far field."""

    def __init__(self, report: FarFieldReport):
        self.report = report
        super().__init__(
            "Fraunhofer condition failed: "
            f"N_F={report.fresnel_number:.4g} exceeds "
            f"{report.max_fresnel_number:.4g}. "
            f"Use distance >= {report.required_distance:.4g} m."
        )


def _require_positive_finite(value: float | None, name: str) -> float:
    if value is None or not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite value.")
    return float(value)


def _aperture_tuple(apertures: Sequence[Aperture]) -> tuple[Aperture, ...]:
    result = tuple(apertures)
    if not result:
        raise ValueError("At least one aperture is required.")
    if not all(isinstance(aperture, Aperture) for aperture in result):
        raise TypeError("Every aperture must be an Aperture instance.")
    return result


def medium_wavelength(wavelength_vacuum: float, n: float = 1.0) -> float:
    """Return wavelength in the propagation medium, ``lambda_0 / n``."""
    wavelength_vacuum = _require_positive_finite(
        wavelength_vacuum, "wavelength_vacuum"
    )
    n = _require_positive_finite(n, "n")
    return wavelength_vacuum / n


def aperture_support_radius(apertures: Sequence[Aperture]) -> float:
    """Return the maximum support radius of a composite aperture."""
    items = _aperture_tuple(apertures)
    return max(aperture.support_radius for aperture in items)


def evaluate_far_field(
    apertures: Sequence[Aperture],
    wavelength_vacuum: float,
    distance: float,
    n: float = 1.0,
    max_fresnel_number: float = config.MAX_FRESNEL_NUMBER.default,
) -> FarFieldReport:
    """Evaluate ``N_F = R_max^2 / (lambda_medium * distance)``.

    ``N_F <= max_fresnel_number`` is the enforced interpretation of the
    far-field ``much greater than`` condition.  No diffraction field is
    evaluated by this function.
    """
    wavelength = medium_wavelength(wavelength_vacuum, n)
    distance = _require_positive_finite(distance, "distance")
    threshold = _require_positive_finite(
        max_fresnel_number, "max_fresnel_number"
    )
    radius = aperture_support_radius(apertures)
    fresnel_number = radius**2 / (wavelength * distance)
    required_distance = radius**2 / (wavelength * threshold)
    return FarFieldReport(
        is_valid=fresnel_number <= threshold,
        fresnel_number=fresnel_number,
        max_fresnel_number=threshold,
        required_distance=required_distance,
        distance=distance,
        wavelength_medium=wavelength,
        support_radius=radius,
    )


def require_far_field(
    apertures: Sequence[Aperture],
    wavelength_vacuum: float,
    distance: float,
    n: float = 1.0,
    max_fresnel_number: float = config.MAX_FRESNEL_NUMBER.default,
) -> FarFieldReport:
    """Return a validity report or raise :class:`FarFieldError`."""
    report = evaluate_far_field(
        apertures, wavelength_vacuum, distance, n, max_fresnel_number
    )
    if not report.is_valid:
        raise FarFieldError(report)
    return report


def spatial_frequencies(
    X: np.ndarray,
    Y: np.ndarray,
    wavelength_vacuum: float,
    distance: float,
    n: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Map observation-plane coordinates to paraxial spatial frequencies."""
    wavelength = medium_wavelength(wavelength_vacuum, n)
    distance = _require_positive_finite(distance, "distance")
    X_array, Y_array = np.broadcast_arrays(
        np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
    )
    return X_array / (wavelength * distance), Y_array / (
        wavelength * distance
    )


def aperture_amplitude(
    fx: np.ndarray,
    fy: np.ndarray,
    aperture: Aperture,
) -> np.ndarray:
    """Analytical Fourier amplitude of one aperture primitive."""
    fx_array, fy_array = np.broadcast_arrays(
        np.asarray(fx, dtype=float), np.asarray(fy, dtype=float)
    )

    if aperture.kind in (ApertureKind.SLIT, ApertureKind.RECTANGLE):
        assert aperture.width is not None and aperture.height is not None
        envelope = (
            aperture.area
            * np.sinc(aperture.width * fx_array)
            * np.sinc(aperture.height * fy_array)
        )
    else:
        assert aperture.radius is not None
        radial_frequency = np.hypot(fx_array, fy_array)
        argument = 2.0 * np.pi * aperture.radius * radial_frequency
        airy_amplitude = np.ones_like(argument)
        np.divide(
            2.0 * j1(argument),
            argument,
            out=airy_amplitude,
            where=argument != 0.0,
        )
        envelope = aperture.area * airy_amplitude

    shift_phase = np.exp(
        -2j
        * np.pi
        * (fx_array * aperture.center_x + fy_array * aperture.center_y)
    )
    return np.asarray(envelope * shift_phase, dtype=complex)


def composite_amplitude(
    fx: np.ndarray,
    fy: np.ndarray,
    apertures: Sequence[Aperture],
) -> np.ndarray:
    """Coherently sum the analytical amplitudes of all openings."""
    items = _aperture_tuple(apertures)
    fx_array, fy_array = np.broadcast_arrays(
        np.asarray(fx, dtype=float), np.asarray(fy, dtype=float)
    )
    total = np.zeros(fx_array.shape, dtype=complex)
    for aperture in items:
        total += aperture_amplitude(fx_array, fy_array, aperture)
    return total


def fraunhofer_field(
    X: np.ndarray,
    Y: np.ndarray,
    apertures: Sequence[Aperture],
    wavelength_vacuum: float,
    distance: float,
    n: float = 1.0,
    max_fresnel_number: float = config.MAX_FRESNEL_NUMBER.default,
) -> tuple[np.ndarray, FarFieldReport]:
    """Evaluate the screen field after enforcing the far-field criterion."""
    items = _aperture_tuple(apertures)
    report = require_far_field(
        items, wavelength_vacuum, distance, n, max_fresnel_number
    )
    fx, fy = spatial_frequencies(X, Y, wavelength_vacuum, distance, n)
    return composite_amplitude(fx, fy, items), report


def fraunhofer_intensity(
    X: np.ndarray,
    Y: np.ndarray,
    apertures: Sequence[Aperture],
    wavelength_vacuum: float,
    distance: float,
    n: float = 1.0,
    max_fresnel_number: float = config.MAX_FRESNEL_NUMBER.default,
    normalize: bool = True,
) -> tuple[np.ndarray, FarFieldReport]:
    """Evaluate Fraunhofer intensity after the mandatory validity check."""
    items = _aperture_tuple(apertures)
    field, report = fraunhofer_field(
        X,
        Y,
        items,
        wavelength_vacuum,
        distance,
        n,
        max_fresnel_number,
    )
    intensity = np.abs(field) ** 2
    if normalize:
        on_axis_intensity = sum(aperture.area for aperture in items) ** 2
        intensity = intensity / on_axis_intensity
    return np.asarray(intensity, dtype=float), report


def slit_first_minimum(
    wavelength_vacuum: float,
    distance: float,
    width: float,
    n: float = 1.0,
) -> float:
    """First paraxial slit minimum, ``x = lambda_medium*z/width``."""
    return (
        medium_wavelength(wavelength_vacuum, n)
        * _require_positive_finite(distance, "distance")
        / _require_positive_finite(width, "width")
    )


def rectangle_first_minima(
    wavelength_vacuum: float,
    distance: float,
    width: float,
    height: float,
    n: float = 1.0,
) -> tuple[float, float]:
    """First minima along x and y for a rectangular opening."""
    wavelength = medium_wavelength(wavelength_vacuum, n)
    distance = _require_positive_finite(distance, "distance")
    return (
        wavelength * distance / _require_positive_finite(width, "width"),
        wavelength * distance / _require_positive_finite(height, "height"),
    )


def circular_first_minimum(
    wavelength_vacuum: float,
    distance: float,
    radius: float,
    n: float = 1.0,
) -> float:
    """First Airy minimum radius, ``0.61*lambda_medium*z/radius``."""
    return (
        0.61
        * medium_wavelength(wavelength_vacuum, n)
        * _require_positive_finite(distance, "distance")
        / _require_positive_finite(radius, "radius")
    )
