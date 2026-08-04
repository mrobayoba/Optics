"""Configuration for the interactive paraxial-optics notebook.

Physics uses SI units.  Widget ranges, display precision, labels, colors, and
preset values live here so the front end does not duplicate numerical values.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "NumericControl",
    "ELEMENT_OPTIONS",
    "PRESET_OPTIONS",
    "PRESET_DEFINITIONS",
    "LENGTH_UNIT_OPTIONS",
    "DEFAULT_LENGTH_UNIT",
    "MM_PER_METRE",
    "REFRACTIVE_INDEX",
    "DISTANCE_M",
    "RADIUS_M",
    "SURFACE_POWER",
    "OBJECT_DISTANCE_M",
    "OBJECT_HEIGHT_M",
    "RAY_ANGLE",
    "RAY_COUNT",
    "DEFAULT_ELEMENT_KIND",
    "DEFAULT_PRESET",
    "DEFAULT_AUTO_UPDATE",
    "MATRIX_PRECISION",
    "MATRIX_TOLERANCE",
    "DEFAULT_DISPLAY_DECIMALS",
    "MAX_DISPLAY_DECIMALS",
    "MIN_DISPLAY_DECIMALS",
    "FIGURE_SIZE",
    "SCHEMATIC_RAY_COUNT",
    "COLORS",
    "DASHBOARD_CSS",
    "length_scale",
]


@dataclass(frozen=True)
class NumericControl:
    """Range and default shared by a slider/text control."""

    minimum: float
    maximum: float
    step: float
    default: float


ELEMENT_OPTIONS = (
    ("Translation T", "translation"),
    ("Refraction Ra", "refraction"),
    ("Reflection Re", "reflection"),
    ("Thin lens", "thin_lens"),
    ("Thick lens", "thick_lens"),
)

PRESET_OPTIONS = (
    ("Empty system", "empty"),
    ("Single thin lens", "thin_lens"),
    ("Single thick lens", "thick_lens"),
)

LENGTH_UNIT_OPTIONS = (
    ("Metres (m)", "m"),
    ("Millimetres (mm)", "mm"),
)
DEFAULT_LENGTH_UNIT = "m"
MM_PER_METRE = 1000.0

REFRACTIVE_INDEX = NumericControl(1.0, 3.0, 1.0e-12, 1.5)
DISTANCE_M = NumericControl(-50.0, 50.0, 1.0e-12, 0.10)
RADIUS_M = NumericControl(-1000.0, 1000.0, 1.0e-12, 0.10)
SURFACE_POWER = NumericControl(-1.0e6, 1.0e6, 1.0e-12, 5.0)
OBJECT_DISTANCE_M = NumericControl(-100.0, 100.0, 1.0e-12, 0.20)
OBJECT_HEIGHT_M = NumericControl(-10.0, 10.0, 1.0e-12, 0.02)
RAY_ANGLE = NumericControl(-0.5, 0.5, 1.0e-12, 0.08)
RAY_COUNT = NumericControl(3, 11, 2, 5)

DEFAULT_ELEMENT_KIND = "translation"
DEFAULT_PRESET = "empty"
DEFAULT_AUTO_UPDATE = True
MATRIX_PRECISION = 12
MATRIX_TOLERANCE = 1e-9
DEFAULT_DISPLAY_DECIMALS = 6
MIN_DISPLAY_DECIMALS = 0
MAX_DISPLAY_DECIMALS = 16
FIGURE_SIZE = (11.0, 5.2)
SCHEMATIC_RAY_COUNT = 5

COLORS = {
    "axis": "#475569",
    "translation": "#0891b2",
    "refraction": "#2563eb",
    "reflection": "#d97706",
    "lens": "#0284c7",
    "principal": "#dc2626",
    "object": "#166534",
    "image": "#9f1239",
    "ray": "#ea580c",
    "virtual_ray": "#94a3b8",
    "panel": "#f8fafc",
}

# Presets list components in optical / addition order (first acts first on the
# input ray). Thin and thick lenses each contribute one composite matrix.
PRESET_DEFINITIONS = {
    "thin_lens": (
        {
            "kind": "thin_lens",
            "label": "L1",
            "lens_index": 1.5,
            "first_power": 5.0,
            "second_power": 5.0,
        },
    ),
    "thick_lens": (
        {
            "kind": "thick_lens",
            "label": "L1",
            "lens_index": 1.5,
            "distance": 0.01,
            "first_power": 5.0,
            "second_power": 5.0,
        },
    ),
    "course_exercise": (
        {
            "kind": "thick_lens",
            "label": "L1",
            "lens_index": 1.812,
            "distance": 5.0,
            # Surface powers from the source radii R = 11.5 and -127.
            "first_power": (1.812 - 1.0) / 11.5,
            "second_power": (1.0 - 1.812) / -127.0,
        },
        {
            "kind": "translation",
            "label": "Gap1",
            "refractive_index": 1.0,
            "distance": 1.25,
        },
        {
            "kind": "thick_lens",
            "label": "L2",
            "lens_index": 1.695,
            "distance": 1.55,
            "first_power": (1.695 - 1.0) / -23.5,
            "second_power": (1.0 - 1.695) / 10.2,
        },
        {
            "kind": "translation",
            "label": "Gap2",
            "refractive_index": 1.0,
            "distance": 2.50,
        },
        {
            "kind": "thick_lens",
            "label": "L3",
            "lens_index": 1.812,
            "distance": 5.0,
            "first_power": (1.812 - 1.0) / 30.0,
            "second_power": (1.0 - 1.812) / -15.0,
        },
    ),
    "empty": (),
}

DASHBOARD_CSS = """
<style>
.paraxial-title {
  color: #0f172a;
  font-family: "Palatino Linotype", "Book Antiqua", Georgia, serif;
  letter-spacing: -0.02em;
}
.paraxial-note {
  border-left: 3px solid #0891b2;
  color: #334155;
  padding: 0.35rem 0.7rem;
}
.paraxial-matrix-panel {
  background: linear-gradient(135deg, #f8fafc, #eef6f8);
  border: 1px solid #cbd5e1;
  padding: 0.65rem;
}
.paraxial-error {
  color: #991b1b;
  border-left: 3px solid #dc2626;
  padding-left: 0.7rem;
}
.paraxial-unit-badge {
  display: inline-block;
  background: #0f766e;
  color: #ecfdf5;
  font-weight: 600;
  padding: 0.2rem 0.55rem;
  border-radius: 0.25rem;
  margin-left: 0.35rem;
}
</style>
"""


def length_scale(unit: str) -> float:
    """Return the factor that converts metres into the selected length unit."""
    if unit == "m":
        return 1.0
    if unit == "mm":
        return MM_PER_METRE
    raise ValueError(f"Unknown length unit {unit!r}; use 'm' or 'mm'.")
