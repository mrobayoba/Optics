"""System-building, matrix-formatting, and ray-sampling helpers.

Each stack component contributes exactly one matrix to the product. Thin and
thick lenses enter as their composite lens matrices, not as expanded surface
factors.

The element list is optical / addition order: the first component is the first
to act on the input ray. Because matrices act on the left of a column state,
that first component is the **rightmost** factor in the written product, and
later components pile on the left:

``M = M_n @ ... @ M_2 @ M_1``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import paraxial_config as config
import paraxial_engine as engine

__all__ = [
    "ElementKind",
    "OpticalElement",
    "ElementPlacement",
    "SystemGeometry",
    "RayTrace",
    "preset_elements",
    "expand_elements",
    "multiply_order_factors",
    "build_system",
    "system_geometry",
    "trace_rays",
    "sample_ray_fan",
    "format_matrix",
    "element_to_mapping",
    "safe_system_filename",
    "save_system",
    "load_system",
    "list_saved_systems",
]


class ElementKind(str, Enum):
    """Logical components accepted by the interactive system builder."""

    TRANSLATION = "translation"
    REFRACTION = "refraction"
    REFLECTION = "reflection"
    THIN_LENS = "thin_lens"
    THICK_LENS = "thick_lens"
    SYSTEM_MATRIX = "system_matrix"


@dataclass(frozen=True)
class OpticalElement:
    """One logical component and the parameters needed to expand it."""

    kind: ElementKind | str
    label: str
    refractive_index: float = 1.0
    distance: float = 0.0
    incident_index: float = 1.0
    transmitted_index: float = 1.0
    radius: float = 1.0
    lens_index: float = 1.0
    first_power: float = 0.0
    second_power: float = 0.0
    m11: float = 1.0
    m12: float = 0.0
    m21: float = 0.0
    m22: float = 1.0

    def __post_init__(self) -> None:
        try:
            kind = ElementKind(self.kind)
        except ValueError as exc:
            choices = ", ".join(item.value for item in ElementKind)
            raise ValueError(
                f"Unknown optical element kind {self.kind!r}; use: {choices}."
            ) from exc
        label = str(self.label).strip()
        if not label:
            raise ValueError("An optical element label cannot be empty.")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "label", label)

        # Calling the elemental constructors here applies the engine's finite,
        # positive-index, and non-zero-radius validation at creation time.
        self.matrix_factors()

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "OpticalElement":
        """Create an element from a preset/UI parameter mapping."""
        return cls(**dict(values))

    @classmethod
    def translation(
        cls,
        label: str,
        refractive_index: float,
        distance: float,
    ) -> "OpticalElement":
        return cls(
            ElementKind.TRANSLATION,
            label,
            refractive_index=refractive_index,
            distance=distance,
        )

    @classmethod
    def refraction(
        cls,
        label: str,
        incident_index: float,
        transmitted_index: float,
        radius: float,
    ) -> "OpticalElement":
        return cls(
            ElementKind.REFRACTION,
            label,
            incident_index=incident_index,
            transmitted_index=transmitted_index,
            radius=radius,
        )

    @classmethod
    def reflection(
        cls,
        label: str,
        incident_index: float,
        radius: float,
    ) -> "OpticalElement":
        return cls(
            ElementKind.REFLECTION,
            label,
            incident_index=incident_index,
            radius=radius,
        )

    @classmethod
    def thin_lens(
        cls,
        label: str,
        first_power: float,
        second_power: float = 0.0,
        lens_index: float = 1.5,
    ) -> "OpticalElement":
        return cls(
            ElementKind.THIN_LENS,
            label,
            lens_index=lens_index,
            first_power=first_power,
            second_power=second_power,
        )

    @classmethod
    def thick_lens(
        cls,
        label: str,
        lens_index: float,
        distance: float,
        first_power: float,
        second_power: float,
    ) -> "OpticalElement":
        return cls(
            ElementKind.THICK_LENS,
            label,
            lens_index=lens_index,
            distance=distance,
            first_power=first_power,
            second_power=second_power,
        )

    @classmethod
    def system_matrix(
        cls,
        label: str,
        matrix: Sequence[Sequence[float]] | np.ndarray,
    ) -> "OpticalElement":
        """Insert a complete unit-determinant 2×2 system matrix (SI units)."""
        values = np.asarray(matrix, dtype=float)
        if values.shape != (2, 2):
            raise ValueError("system matrix must be 2x2.")
        return cls(
            ElementKind.SYSTEM_MATRIX,
            label,
            m11=float(values[0, 0]),
            m12=float(values[0, 1]),
            m21=float(values[1, 0]),
            m22=float(values[1, 1]),
        )

    def matrix_factors(self) -> tuple[engine.MatrixFactor, ...]:
        """Return the single matrix contributed by this stack component."""
        if self.kind == ElementKind.TRANSLATION:
            matrix = engine.translation_matrix(self.refractive_index, self.distance)
            return (
                engine.MatrixFactor(
                    f"T[{self.label}]",
                    matrix,
                    f"n={self.refractive_index:g}, d={self.distance:g} m",
                    style_key="translation",
                ),
            )

        if self.kind == ElementKind.REFRACTION:
            power = engine.interface_power(
                self.incident_index,
                self.transmitted_index,
                self.radius,
            )
            return (
                engine.MatrixFactor(
                    f"Ra[{self.label}]",
                    engine.refraction_matrix(power),
                    (
                        f"P={power:.6g} m⁻¹; nᵢ={self.incident_index:g}, "
                        f"nₜ={self.transmitted_index:g}, R={self.radius:g} m"
                    ),
                    style_key="refraction",
                ),
            )

        if self.kind == ElementKind.REFLECTION:
            return (
                engine.MatrixFactor(
                    f"Re[{self.label}]",
                    engine.reflection_matrix(self.incident_index, self.radius),
                    f"nᵢ={self.incident_index:g}, R={self.radius:g} m",
                    style_key="reflection",
                ),
            )

        if self.kind == ElementKind.THIN_LENS:
            power = self.first_power + self.second_power
            return (
                engine.MatrixFactor(
                    f"Mtn[{self.label}]",
                    engine.thin_lens_matrix(self.first_power, self.second_power),
                    (
                        f"P={power:.6g} m⁻¹ "
                        f"(P₁={self.first_power:.6g}, P₂={self.second_power:.6g})"
                    ),
                    style_key="thin_lens",
                ),
            )

        if self.kind == ElementKind.SYSTEM_MATRIX:
            matrix = np.array(
                [[self.m11, self.m12], [self.m21, self.m22]],
                dtype=float,
            )
            determinant = engine.assert_unit_determinant(matrix)
            return (
                engine.MatrixFactor(
                    f"M[{self.label}]",
                    matrix,
                    f"custom M_VV′; det={determinant:.6g}",
                    style_key="system_matrix",
                ),
            )

        return (
            engine.MatrixFactor(
                f"Mtk[{self.label}]",
                engine.thick_lens_matrix(
                    self.lens_index,
                    self.distance,
                    self.first_power,
                    self.second_power,
                ),
                (
                    f"n={self.lens_index:g}, d={self.distance:g} m, "
                    f"P₁={self.first_power:.6g} m⁻¹, P₂={self.second_power:.6g} m⁻¹"
                ),
                style_key="thick_lens",
            ),
        )

    @property
    def matrix(self) -> np.ndarray:
        """Resolved matrix of this logical component."""
        return np.array(self.matrix_factors()[0].matrix, dtype=float, copy=True)

    @property
    def axial_length(self) -> float:
        """Signed distance added to the system path coordinate."""
        if self.kind in (ElementKind.TRANSLATION, ElementKind.THICK_LENS):
            return float(self.distance)
        return 0.0


@dataclass(frozen=True)
class ElementPlacement:
    """Logical component location along the one-dimensional system path."""

    label: str
    kind: ElementKind
    start: float
    finish: float


@dataclass(frozen=True)
class SystemGeometry:
    """Start/finish vertices and logical element placements."""

    start_vertex: float
    finish_vertex: float
    placements: tuple[ElementPlacement, ...]


@dataclass(frozen=True)
class RayTrace:
    """Sampled row-state rays ``[x, n*alpha]`` through elemental factors."""

    path_positions: np.ndarray
    heights: np.ndarray
    reduced_angles: np.ndarray
    factor_labels: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("path_positions", "heights", "reduced_angles"):
            value = np.array(getattr(self, name), dtype=float, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "factor_labels", tuple(self.factor_labels))


def preset_elements(name: str) -> list[OpticalElement]:
    """Instantiate one named configuration preset."""
    try:
        definitions = config.PRESET_DEFINITIONS[name]
    except KeyError as exc:
        choices = ", ".join(config.PRESET_DEFINITIONS)
        raise ValueError(f"Unknown preset {name!r}; use: {choices}.") from exc
    return [OpticalElement.from_mapping(values) for values in definitions]


def expand_elements(
    elements: Sequence[OpticalElement],
) -> tuple[engine.MatrixFactor, ...]:
    """Return one matrix per component in optical / addition order.

    Index 0 is the first component added (first to act on the input ray).
    """
    return tuple(
        factor
        for element in elements
        for factor in element.matrix_factors()
    )


def multiply_order_factors(
    elements: Sequence[OpticalElement],
) -> tuple[engine.MatrixFactor, ...]:
    """Return factors in written product order ``M_n @ ... @ M_1``.

    The first optical component is rightmost so it multiplies the input vector
    first; later components pile on the left.
    """
    return tuple(reversed(expand_elements(elements)))


def build_system(
    elements: Sequence[OpticalElement],
    tolerance: float = config.MATRIX_TOLERANCE,
) -> engine.SystemMatrixReport:
    """Resolve an element stack into its base ``M_VV'`` report."""
    return engine.system_report(multiply_order_factors(elements), tolerance)


def system_geometry(elements: Sequence[OpticalElement]) -> SystemGeometry:
    """Lay out logical components on a signed optical path coordinate."""
    z = 0.0
    placements: list[ElementPlacement] = []
    for element in elements:
        finish = z + element.axial_length
        placements.append(
            ElementPlacement(element.label, element.kind, z, finish)
        )
        z = finish
    return SystemGeometry(0.0, z, tuple(placements))


def _factor_operations(
    elements: Sequence[OpticalElement],
) -> tuple[tuple[engine.MatrixFactor, float], ...]:
    operations: list[tuple[engine.MatrixFactor, float]] = []
    for element in elements:
        factor = element.matrix_factors()[0]
        operations.append((factor, element.axial_length))
    return tuple(operations)


def trace_rays(
    elements: Sequence[OpticalElement],
    initial_heights: Sequence[float] | np.ndarray,
    initial_reduced_angles: Sequence[float] | np.ndarray,
) -> RayTrace:
    """Propagate row-state rays through components in optical order.

    Each component matrix ``M`` is applied as ``state @ M`` with
    ``state = [x, n*alpha]``, which matches the column action ``M @ r`` for the
    course state ``r = [n*alpha, x]``. Components are applied first-to-last,
    consistent with the system product ``M_n @ ... @ M_1``.
    """
    heights = np.asarray(initial_heights, dtype=float)
    angles = np.asarray(initial_reduced_angles, dtype=float)
    if heights.ndim != 1 or angles.ndim != 1 or heights.shape != angles.shape:
        raise ValueError(
            "initial_heights and initial_reduced_angles must be equal 1D arrays."
        )
    if heights.size == 0:
        raise ValueError("At least one ray is required.")
    if not np.all(np.isfinite(heights)) or not np.all(np.isfinite(angles)):
        raise ValueError("Initial ray states must be finite.")

    states = np.column_stack((heights, angles))
    positions = [0.0]
    sampled_heights = [states[:, 0].copy()]
    sampled_angles = [states[:, 1].copy()]
    labels = ["V"]
    z = 0.0

    for factor, advance in _factor_operations(elements):
        states = states @ factor.matrix
        z += advance
        positions.append(z)
        sampled_heights.append(states[:, 0].copy())
        sampled_angles.append(states[:, 1].copy())
        labels.append(factor.label)

    return RayTrace(
        path_positions=np.asarray(positions),
        heights=np.asarray(sampled_heights).T,
        reduced_angles=np.asarray(sampled_angles).T,
        factor_labels=tuple(labels),
    )


def sample_ray_fan(
    elements: Sequence[OpticalElement],
    height: float,
    half_angle: float,
    ray_count: int,
    input_index: float = 1.0,
) -> RayTrace:
    """Trace an evenly spaced fan launched at one height on the first vertex."""
    n = float(input_index)
    if not np.isfinite(n) or n <= 0.0:
        raise ValueError("input_index must be positive and finite.")
    count = int(ray_count)
    if count < 2:
        raise ValueError("ray_count must be at least two.")
    angles = np.linspace(-float(half_angle), float(half_angle), count)
    return trace_rays(
        elements,
        np.full(count, float(height)),
        n * angles,
    )


def format_matrix(matrix: np.ndarray, precision: int = config.MATRIX_PRECISION) -> str:
    """Return a compact aligned numeric representation of a 2x2 matrix."""
    values = np.asarray(matrix, dtype=float)
    if values.shape != (2, 2):
        raise ValueError("matrix must be 2x2.")
    return np.array2string(
        values,
        precision=int(precision),
        suppress_small=False,
        floatmode="maxprec_equal",
    )


def element_to_mapping(element: OpticalElement) -> dict[str, Any]:
    """Serialize one stack component to a preset-style mapping (SI units)."""
    kind = ElementKind(element.kind)
    payload: dict[str, Any] = {"kind": kind.value, "label": element.label}
    if kind == ElementKind.TRANSLATION:
        payload["refractive_index"] = float(element.refractive_index)
        payload["distance"] = float(element.distance)
        return payload
    if kind == ElementKind.REFRACTION:
        payload["incident_index"] = float(element.incident_index)
        payload["transmitted_index"] = float(element.transmitted_index)
        payload["radius"] = float(element.radius)
        return payload
    if kind == ElementKind.REFLECTION:
        payload["incident_index"] = float(element.incident_index)
        payload["radius"] = float(element.radius)
        return payload
    if kind == ElementKind.THIN_LENS:
        payload["lens_index"] = float(element.lens_index)
        payload["first_power"] = float(element.first_power)
        payload["second_power"] = float(element.second_power)
        return payload
    if kind == ElementKind.THICK_LENS:
        payload["lens_index"] = float(element.lens_index)
        payload["distance"] = float(element.distance)
        payload["first_power"] = float(element.first_power)
        payload["second_power"] = float(element.second_power)
        return payload
    payload["m11"] = float(element.m11)
    payload["m12"] = float(element.m12)
    payload["m21"] = float(element.m21)
    payload["m22"] = float(element.m22)
    return payload


def safe_system_filename(name: str) -> str:
    """Return a filesystem-safe stem for a saved optical system."""
    cleaned = re.sub(r"[^\w.\-]+", "_", str(name).strip(), flags=re.UNICODE)
    cleaned = cleaned.strip("._")
    if not cleaned:
        raise ValueError("Save name must contain at least one letter or digit.")
    return cleaned


def list_saved_systems(directory: Path | None = None) -> list[str]:
    """Return saved system stems (sorted) under ``directory``."""
    root = Path(directory) if directory is not None else config.SAVES_DIR
    if not root.is_dir():
        return []
    return sorted(path.stem for path in root.glob("*.json") if path.is_file())


def save_system(
    name: str,
    elements: Sequence[OpticalElement],
    analysis: Mapping[str, Any],
    *,
    length_unit: str = "m",
    directory: Path | None = None,
) -> Path:
    """Write stack + analysis JSON under ``saves/`` (SI metres / m⁻¹)."""
    root = Path(directory) if directory is not None else config.SAVES_DIR
    root.mkdir(parents=True, exist_ok=True)
    stem = safe_system_filename(name)
    path = root / f"{stem}.json"
    required = (
        "input_index",
        "output_index",
        "conjugate_mode",
        "vertex_distance_m",
        "object_height_m",
        "ray_half_angle",
        "ray_count",
        "display_decimals",
    )
    missing = [key for key in required if key not in analysis]
    if missing:
        raise ValueError(f"analysis is missing keys: {', '.join(missing)}")
    if length_unit not in ("m", "mm"):
        raise ValueError("length_unit must be 'm' or 'mm'.")
    payload = {
        "version": config.SYSTEM_SAVE_VERSION,
        "name": stem,
        "length_unit": length_unit,
        "analysis": {
            "input_index": float(analysis["input_index"]),
            "output_index": float(analysis["output_index"]),
            "conjugate_mode": str(analysis["conjugate_mode"]),
            "vertex_distance_m": float(analysis["vertex_distance_m"]),
            "object_height_m": float(analysis["object_height_m"]),
            "ray_half_angle": float(analysis["ray_half_angle"]),
            "ray_count": int(analysis["ray_count"]),
            "display_decimals": int(analysis["display_decimals"]),
        },
        "elements": [element_to_mapping(element) for element in elements],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_system(
    name: str,
    *,
    directory: Path | None = None,
) -> dict[str, Any]:
    """Load a saved system; elements are ``OpticalElement`` instances (SI)."""
    root = Path(directory) if directory is not None else config.SAVES_DIR
    stem = safe_system_filename(name)
    path = root / f"{stem}.json"
    if not path.is_file():
        raise FileNotFoundError(f"No saved system named {stem!r} in {root}.")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Saved system file must contain a JSON object.")
    version = int(raw.get("version", 0))
    if version != config.SYSTEM_SAVE_VERSION:
        raise ValueError(
            f"Unsupported save version {version}; "
            f"expected {config.SYSTEM_SAVE_VERSION}."
        )
    length_unit = str(raw.get("length_unit", "m"))
    if length_unit not in ("m", "mm"):
        raise ValueError("Saved length_unit must be 'm' or 'mm'.")
    analysis_raw = raw.get("analysis")
    if not isinstance(analysis_raw, Mapping):
        raise ValueError("Saved system must include an analysis object.")
    elements_raw = raw.get("elements")
    if not isinstance(elements_raw, list):
        raise ValueError("Saved system must include an elements list.")
    elements = [OpticalElement.from_mapping(entry) for entry in elements_raw]
    return {
        "version": version,
        "name": str(raw.get("name", stem)),
        "length_unit": length_unit,
        "analysis": dict(analysis_raw),
        "elements": elements,
        "path": path,
    }
