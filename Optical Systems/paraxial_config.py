"""Configuration for the interactive paraxial-optics notebook.

Physics uses SI units.  Widget ranges, display precision, labels, colors, and
preset values live here so the front end does not duplicate numerical values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "NumericControl",
    "ELEMENT_OPTIONS",
    "PRESET_OPTIONS",
    "PRESET_DEFINITIONS",
    "LENGTH_UNIT_OPTIONS",
    "DEFAULT_LENGTH_UNIT",
    "MM_PER_METRE",
    "PACKAGE_DIR",
    "SAVES_DIR",
    "SYSTEM_SAVE_VERSION",
    "REFRACTIVE_INDEX",
    "DISTANCE_M",
    "RADIUS_M",
    "SURFACE_POWER",
    "MATRIX_ENTRY",
    "OBJECT_DISTANCE_M",
    "OBJECT_HEIGHT_M",
    "VERTEX_DISTANCE_M",
    "CONJUGATE_MODE_OPTIONS",
    "DEFAULT_CONJUGATE_MODE",
    "ZOOM_SCALE",
    "ZOOM_PAN",
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
    "FIGURE_WIDTH_MAX",
    "FIGURE_WIDTH_PER_EXTRA_ELEMENT",
    "SCHEMATIC_RAY_COUNT",
    "COLORS",
    "FACTOR_CARD_STYLES",
    "DASHBOARD_CSS",
    "length_scale",
    "figure_size_for_elements",
    "factor_card_style",
    "zoom_window",
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
    ("System matrix M", "system_matrix"),
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

PACKAGE_DIR = Path(__file__).resolve().parent
SAVES_DIR = PACKAGE_DIR / "saves"
SYSTEM_SAVE_VERSION = 1

REFRACTIVE_INDEX = NumericControl(1.0, 3.0, 1.0e-12, 1.5)
DISTANCE_M = NumericControl(-50.0, 50.0, 1.0e-12, 0.10)
RADIUS_M = NumericControl(-1000.0, 1000.0, 1.0e-12, 0.10)
SURFACE_POWER = NumericControl(-1.0e6, 1.0e6, 1.0e-12, 5.0)
MATRIX_ENTRY = NumericControl(-1.0e12, 1.0e12, 1.0e-12, 0.0)
OBJECT_DISTANCE_M = NumericControl(-100.0, 100.0, 1.0e-12, 0.20)
OBJECT_HEIGHT_M = NumericControl(-10.0, 10.0, 1.0e-12, 0.02)
VERTEX_DISTANCE_M = NumericControl(-100.0, 100.0, 1.0e-12, 0.20)
CONJUGATE_MODE_OPTIONS = (
    ("Object → V (x)", "object_to_v"),
    ("V′ → image (x′)", "v_prime_to_image"),
)
DEFAULT_CONJUGATE_MODE = "object_to_v"
ZOOM_SCALE = NumericControl(1.0, 8.0, 0.1, 1.0)
ZOOM_PAN = NumericControl(-1.0, 1.0, 0.01, 0.0)
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
FIGURE_WIDTH_MAX = 22.0
FIGURE_WIDTH_PER_EXTRA_ELEMENT = 0.55
SCHEMATIC_RAY_COUNT = 5

COLORS = {
    "axis": "#475569",
    "translation": "#16a34a",
    "refraction": "#2563eb",
    "reflection": "#ca8a04",
    "thin_lens": "#c026d3",
    "thick_lens": "#ffffff",
    "system_matrix": "#475569",
    "result": "#334155",
    # Kept for older schematic call sites; prefer thin_lens / thick_lens.
    "lens": "#c026d3",
    "principal": "#dc2626",
    "object": "#166534",
    "image": "#9f1239",
    "ray": "#ea580c",
    "virtual_ray": "#94a3b8",
    "panel": "#f8fafc",
}

# Border + fill for matrix factor cards and element-stack rows.
FACTOR_CARD_STYLES = {
    "translation": {"border": "#16a34a", "background": "#dcfce7"},
    "refraction": {"border": "#2563eb", "background": "#dbeafe"},
    "reflection": {"border": "#ca8a04", "background": "#fef9c3"},
    "thin_lens": {"border": "#c026d3", "background": "#fae8ff"},
    "thick_lens": {"border": "#334155", "background": "#ffffff"},
    "system_matrix": {"border": "#475569", "background": "#e2e8f0"},
    "result": {"border": "#334155", "background": "#f1f5f9"},
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
.paraxial-factor-card {
  display: inline-flex;
  flex-direction: column;
  align-items: center;
  gap: 0.18rem;
  padding: 0.35rem 0.45rem;
  border-radius: 0.35rem;
  border: 2px solid #334155;
  background: #f1f5f9;
  box-sizing: border-box;
}
.paraxial-product-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.35rem;
  overflow-x: auto;
  max-width: 100%;
  padding-bottom: 0.45rem;
}
.paraxial-figure-scroll {
  width: 100%;
  overflow-x: auto;
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


def factor_card_style(style_key: str) -> dict[str, str]:
    """Return border/background colors for a matrix factor card."""
    key = str(style_key or "result").strip() or "result"
    return dict(FACTOR_CARD_STYLES.get(key, FACTOR_CARD_STYLES["result"]))


def figure_size_for_elements(count: int) -> tuple[float, float]:
    """Grow the schematic figure when many components would crush labels."""
    base_w, base_h = FIGURE_SIZE
    n = max(0, int(count))
    width = min(
        FIGURE_WIDTH_MAX,
        base_w + FIGURE_WIDTH_PER_EXTRA_ELEMENT * max(0, n - 4),
    )
    height = base_h + (0.35 if n >= 8 else 0.0)
    return (float(width), float(height))


def zoom_window(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    zoom: float = 1.0,
    pan: float = 0.0,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return cropped axis limits for schematic zoom/pan.

    ``zoom`` >= 1 shrinks the visible span. ``pan`` in [-1, 1] shifts the
    window as a fraction of half the full span.
    """
    z = max(float(zoom), 1.0e-9)
    p = float(pan)
    if z != z or p != p or abs(z) == float("inf") or abs(p) == float("inf"):
        raise ValueError("zoom and pan must be finite.")
    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    x_half = 0.5 * (x_max - x_min)
    y_half = 0.5 * (y_max - y_min)
    x_center = x_mid + p * x_half
    y_center = y_mid
    return (
        (x_center - x_half / z, x_center + x_half / z),
        (y_center - y_half / z, y_center + y_half / z),
    )
