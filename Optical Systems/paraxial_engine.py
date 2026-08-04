"""Paraxial matrix optics using the course ``(n*alpha, x)`` convention.

The matrices in this module intentionally match the MATLAB source material:

``T(n, d) = [[1, 0], [d/n, 1]]``
``Ra(P)   = [[1, -P], [0, 1]]``
``Re(n,R) = [[1, 2*n/R], [0, 1]]``

System factors in a report are ordered left-to-right as the written product.
The rightmost factor multiplies the input ray first.
All lengths use metres, refractive indices are dimensionless, and optical
powers use inverse metres.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "AfocalSystemError",
    "ConjugateAtInfinityError",
    "MatrixFactor",
    "SystemMatrixReport",
    "CardinalPoints",
    "ConjugateReport",
    "interface_power",
    "translation_matrix",
    "refraction_matrix",
    "refraction_surface_matrix",
    "reflection_matrix",
    "thick_lens_matrix",
    "thin_lens_matrix",
    "cascade",
    "system_report",
    "assert_unit_determinant",
    "principal_planes",
    "solve_image_distance",
    "conjugate_planes",
]

Matrix = np.ndarray
DEFAULT_TOLERANCE = 1e-10


class AfocalSystemError(ValueError):
    """Raised when finite principal planes cannot be assigned."""


class ConjugateAtInfinityError(ValueError):
    """Raised when the selected object plane is conjugate to infinity."""


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _positive(value: float, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be greater than zero.")
    return result


def _nonzero(value: float, name: str) -> float:
    result = _finite(value, name)
    if result == 0.0:
        raise ValueError(f"{name} must be non-zero.")
    return result


def _matrix(value: Matrix, name: str = "matrix") -> Matrix:
    result = np.asarray(value, dtype=float)
    if result.shape != (2, 2):
        raise ValueError(f"{name} must be a 2x2 matrix.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    return result


def _readonly_matrix(value: Matrix) -> Matrix:
    result = np.array(value, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class MatrixFactor:
    """One named 2x2 factor in a displayed matrix product."""

    label: str
    matrix: Matrix
    description: str = ""

    def __post_init__(self) -> None:
        label = str(self.label).strip()
        if not label:
            raise ValueError("A matrix factor label cannot be empty.")
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "matrix", _readonly_matrix(_matrix(self.matrix)))


@dataclass(frozen=True)
class SystemMatrixReport:
    """Ordered factors and their vertex-to-vertex product."""

    factors: tuple[MatrixFactor, ...]
    matrix: Matrix
    determinant: float
    is_unit_determinant: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "factors", tuple(self.factors))
        object.__setattr__(self, "matrix", _readonly_matrix(_matrix(self.matrix)))

    @property
    def product_expression(self) -> str:
        """Human-readable multiplication expression."""
        return " @ ".join(factor.label for factor in self.factors)


@dataclass(frozen=True)
class CardinalPoints:
    """Principal-plane reduction of a vertex-to-vertex system matrix."""

    power: float
    object_principal_offset: float
    image_principal_offset: float
    front_focal_length: float
    back_focal_length: float
    equivalent_matrix: Matrix
    reduction_factors: tuple[MatrixFactor, ...]
    reduction_matrix: Matrix
    reduction_residual: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "equivalent_matrix", _readonly_matrix(_matrix(self.equivalent_matrix))
        )
        object.__setattr__(self, "reduction_factors", tuple(self.reduction_factors))
        object.__setattr__(
            self, "reduction_matrix", _readonly_matrix(_matrix(self.reduction_matrix))
        )

    @property
    def principal_product_expression(self) -> str:
        return " @ ".join(factor.label for factor in self.reduction_factors)


@dataclass(frozen=True)
class ConjugateReport:
    """Object/image conjugate solution and complete transfer matrix."""

    object_distance: float
    image_distance: float
    lateral_magnification: float
    angular_magnification: float
    lagrange_invariant: float
    matrix: Matrix
    factors: tuple[MatrixFactor, ...]
    conjugacy_residual: float
    is_conjugate: bool
    is_real: bool
    is_erect: bool
    size_classification: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "matrix", _readonly_matrix(_matrix(self.matrix)))
        object.__setattr__(self, "factors", tuple(self.factors))

    @property
    def product_expression(self) -> str:
        return " @ ".join(factor.label for factor in self.factors)

## Optical System Functions
def interface_power(
    incident_index: float,
    transmitted_index: float,
    radius: float,
) -> float:
    """Return spherical-interface power ``(n_t - n_i) / R``."""
    n_i = _positive(incident_index, "incident_index")
    n_t = _positive(transmitted_index, "transmitted_index")
    radius = _nonzero(radius, "radius")
    return (n_t - n_i) / radius


def translation_matrix(refractive_index: float, distance: float) -> Matrix:
    """Return the course translation matrix through a homogeneous medium."""
    n = _positive(refractive_index, "refractive_index")
    d = _finite(distance, "distance")
    return np.array([[1.0, 0.0], [d / n, 1.0]], dtype=float)


def refraction_matrix(power: float) -> Matrix:
    """Return a refracting-surface matrix from its optical power."""
    p = _finite(power, "power")
    return np.array([[1.0, -p], [0.0, 1.0]], dtype=float)


def refraction_surface_matrix(
    incident_index: float,
    transmitted_index: float,
    radius: float,
) -> Matrix:
    """Return a refracting-surface matrix from indices and radius."""
    return refraction_matrix(
        interface_power(incident_index, transmitted_index, radius)
    )


def reflection_matrix(incident_index: float, radius: float) -> Matrix:
    """Return the spherical-reflection matrix from the course notes."""
    n_i = _positive(incident_index, "incident_index")
    radius = _nonzero(radius, "radius")
    return np.array([[1.0, 2.0 * n_i / radius], [0.0, 1.0]], dtype=float)


def thick_lens_matrix(
    lens_index: float,
    thickness: float,
    first_surface_power: float,
    second_surface_power: float,
) -> Matrix:
    """Return the thick-lens matrix with the first surface acting first.

    Factors are listed in optical order ``(Ra(P_s), T, Ra(P_f))``, then
    reversed once before :func:`cascade` so the product is

    ``Ra(P_f) @ T(n_l, d) @ Ra(P_s)``

    matching a dashboard stack built from separate ``Ra``, ``T``, ``Ra``
    components (first component rightmost). ``cascade`` itself never reverses.
    """
    optical_order = (
        refraction_matrix(first_surface_power),
        translation_matrix(lens_index, thickness),
        refraction_matrix(second_surface_power),
    )
    return cascade(reversed(optical_order))


def thin_lens_matrix(
    first_surface_power: float,
    second_surface_power: float = 0.0,
) -> Matrix:
    """Return a zero-thickness lens built from its two surface powers."""
    return thick_lens_matrix(
        1.0,
        0.0,
        first_surface_power,
        second_surface_power,
    )


def cascade(matrices: Iterable[Matrix]) -> Matrix:
    """Multiply the supplied sequence left-to-right without reordering it.

    ``cascade((M1, M2, Mn))`` returns ``M1 @ M2 @ Mn``.  This function never
    reverses its input.  A caller starting from optical order
    ``(first, ..., last)`` must explicitly pass ``reversed(optical_order)`` so
    the first component is rightmost and acts first on the input column ray.
    """
    values = tuple(_matrix(value, f"matrices[{index}]") for index, value in enumerate(matrices))
    if not values:
        return np.eye(2, dtype=float)
    result = np.eye(2, dtype=float)
    for value in values:
        result = result @ value
    return result


def system_report(
    factors: Sequence[MatrixFactor],
    tolerance: float = DEFAULT_TOLERANCE,
) -> SystemMatrixReport:
    """Resolve ordered factors into the base vertex-to-vertex matrix.

    ``factors`` must already be in written product order
    ``(M_n, ..., M_1)``, with the first optical component last in the tuple.
    """
    items = tuple(factors)
    matrix = cascade(factor.matrix for factor in items)
    determinant = float(np.linalg.det(matrix))
    return SystemMatrixReport(
        factors=items,
        matrix=matrix,
        determinant=determinant,
        is_unit_determinant=bool(
            np.isclose(determinant, 1.0, rtol=tolerance, atol=tolerance)
        ),
    )


def assert_unit_determinant(
    matrix: Matrix,
    tolerance: float = DEFAULT_TOLERANCE,
) -> float:
    """Return the determinant or raise if it differs materially from one."""
    value = float(np.linalg.det(_matrix(matrix)))
    if not np.isclose(value, 1.0, rtol=tolerance, atol=tolerance):
        raise ValueError(
            f"Expected a unit determinant, got {value:.12g} "
            f"(tolerance {tolerance:.3g})."
        )
    return value

## Plane Reduction Functions
def principal_planes(
    matrix: Matrix,
    input_index: float = 1.0,
    output_index: float = 1.0,
    tolerance: float = DEFAULT_TOLERANCE,
) -> CardinalPoints:
    """Reduce ``M_VV'`` to the equivalent matrix between H and H'.

    The distances follow the source notation:

    ``D = n * (1 - M11) / M12`` and
    ``D' = n' * (1 - M22) / M12``.
    """
    base = _matrix(matrix)
    assert_unit_determinant(base, tolerance)
    n = _positive(input_index, "input_index")
    n_prime = _positive(output_index, "output_index")
    a, b = float(base[0, 0]), float(base[0, 1])
    d = float(base[1, 1])
    if np.isclose(b, 0.0, rtol=tolerance, atol=tolerance):
        raise AfocalSystemError(
            "M12 is zero: the system is afocal and has no finite principal "
            "plane reduction."
        )

    power = -b
    object_offset = n * (1.0 - a) / b
    image_offset = n_prime * (1.0 - d) / b
    equivalent = refraction_matrix(power)

    factors = (
        MatrixFactor(
            "T(V'→H')",
            translation_matrix(n_prime, image_offset),
            f"D'={image_offset:.6g} m",
        ),
        MatrixFactor("M_VV'", base, "vertex-to-vertex system"),
        MatrixFactor(
            "T(H→V)",
            translation_matrix(n, object_offset),
            f"D={object_offset:.6g} m",
        ),
    )
    reduced = cascade(factor.matrix for factor in factors)
    residual = float(np.max(np.abs(reduced - equivalent)))

    return CardinalPoints(
        power=power,
        object_principal_offset=object_offset,
        image_principal_offset=image_offset,
        front_focal_length=n / power,
        back_focal_length=n_prime / power,
        equivalent_matrix=equivalent,
        reduction_factors=factors,
        reduction_matrix=reduced,
        reduction_residual=residual,
    )


def solve_image_distance(
    power: float,
    object_distance: float,
    input_index: float = 1.0,
    output_index: float = 1.0,
    tolerance: float = DEFAULT_TOLERANCE,
) -> float:
    """Solve ``n'/s' + n/s = P`` for the image distance."""
    p = _finite(power, "power")
    s = _nonzero(object_distance, "object_distance")
    n = _positive(input_index, "input_index")
    n_prime = _positive(output_index, "output_index")
    denominator = p - n / s
    if np.isclose(denominator, 0.0, rtol=tolerance, atol=tolerance):
        raise ConjugateAtInfinityError(
            "The object lies at the front focal plane; its conjugate is at infinity."
        )
    return n_prime / denominator


def conjugate_planes(
    cardinal: CardinalPoints,
    object_distance: float,
    input_index: float = 1.0,
    output_index: float = 1.0,
    tolerance: float = DEFAULT_TOLERANCE,
) -> ConjugateReport:
    """Resolve the object/image conjugate transfer around ``M_HH'``.

    In the course matrix convention the resolved operator is displayed as
    ``T(H'→image) @ M_HH' @ T(object→H)``.  With the Gaussian image
    distance this makes ``M21 = 0`` and ``M22`` the lateral magnification
    quoted in the source notes.
    """
    s = _nonzero(object_distance, "object_distance")
    n = _positive(input_index, "input_index")
    n_prime = _positive(output_index, "output_index")
    s_prime = solve_image_distance(
        cardinal.power,
        s,
        n,
        n_prime,
        tolerance,
    )

    factors = (
        MatrixFactor(
            "T(H'→image)",
            translation_matrix(n_prime, s_prime),
            f"s'={s_prime:.6g} m",
        ),
        MatrixFactor("M_HH'", cardinal.equivalent_matrix, "equivalent system"),
        MatrixFactor(
            "T(object→H)",
            translation_matrix(n, s),
            f"s={s:.6g} m",
        ),
    )
    matrix = cascade(factor.matrix for factor in factors)
    residual = float(matrix[1, 0])
    lateral = float(matrix[1, 1])
    angular = float(matrix[0, 0] * n / n_prime)
    lagrange = float(lateral * angular * n_prime / n)

    if np.isclose(abs(lateral), 1.0, rtol=tolerance, atol=tolerance):
        size = "same size"
    elif abs(lateral) > 1.0:
        size = "magnified"
    else:
        size = "reduced"

    return ConjugateReport(
        object_distance=s,
        image_distance=s_prime,
        lateral_magnification=lateral,
        angular_magnification=angular,
        lagrange_invariant=lagrange,
        matrix=matrix,
        factors=factors,
        conjugacy_residual=residual,
        is_conjugate=bool(np.isclose(residual, 0.0, atol=tolerance)),
        is_real=s_prime > 0.0,
        is_erect=lateral > 0.0,
        size_classification=size,
    )
## End of Plane Reduction Functions