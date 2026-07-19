"""Central configuration for the interactive Fraunhofer simulator.

The physics modules use SI units.  This file also defines the display-unit
scales and widget ranges used by ``fraunhofer_simulator.ipynb`` so those values
are not duplicated throughout the notebook.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "NumericControl",
    "CASE_OPTIONS",
    "SHAPE_OPTIONS",
    "SLIT_ORIENTATION_OPTIONS",
    "VIEW_OPTIONS",
    "DEFAULT_ACTIVE_VIEWS",
    "DASHBOARD_CASE_OPTIONS",
    "EXTRA_VIEW_OPTIONS",
    "DEFAULT_AUTO_UPDATE",
    "WAVELENGTH_NM",
    "REFRACTIVE_INDEX",
    "DISTANCE_M",
    "SCREEN_HALF_WIDTH_MM",
    "ZOOM",
    "RESOLUTION",
    "MAX_FRESNEL_NUMBER",
    "SECONDARY_MAXIMA_BOOST",
    "APERTURE_WIDTH_UM",
    "APERTURE_HEIGHT_UM",
    "CIRCLE_RADIUS_UM",
    "CENTER_MM",
    "DEFAULT_SLIT_WIDTH_M",
    "DEFAULT_SLIT_LENGTH_M",
    "DEFAULT_RECTANGLE_WIDTH_M",
    "DEFAULT_RECTANGLE_HEIGHT_M",
    "DEFAULT_CIRCLE_RADIUS_M",
    "DEFAULT_MULTIPLE_CENTERS_MM",
    "SLIDER_WIDTH",
    "TEXT_WIDTH",
    "FIGURE_SIZE",
    "FIGURE_WIDTH_RATIOS",
    "GEOMETRY_FIGURE_SIZE",
    "APERTURE_FIGURE_SIZE",
    "APERTURE_VIEW_MARGIN",
    "GEOMETRY_APERTURE_HALF_HEIGHT",
    "GEOMETRY_SCREEN_HALF_HEIGHT",
    "GEOMETRY_VERTICAL_LIMIT",
    "COLOR_MAP",
    "PROFILE_COLORS",
    "VISIBLE_WAVELENGTH_MIN_NM",
    "VISIBLE_WAVELENGTH_MAX_NM",
    "WAVELENGTH_COLOR_GAMMA",
    "INTENSITY_DISPLAY_GAMMA",
    "DASHBOARD_BACKGROUND",
    "PANEL_BACKGROUND",
    "APERTURE_COLOR",
    "SCHEMATIC_BEAM_ALPHA",
    "SCALE_BAR_FRACTION",
    "SCALE_BAR_LINEWIDTH",
    "SCHEMATIC_FIGURE_SIZE",
    "DASHBOARD_PREVIEW_FIGURE_SIZE",
    "DASHBOARD_PROFILE_FIGURE_SIZE",
    "DASHBOARD_CONTROL_WIDTH",
    "DASHBOARD_PREVIEW_WIDTH",
    "DASHBOARD_SLIDER_WIDTH",
    "DASHBOARD_TEXT_WIDTH",
    "DASHBOARD_MAIN_COLUMNS",
    "DASHBOARD_COLUMN_GAP",
    "DASHBOARD_MAX_WIDTH",
    "DASHBOARD_CARD_BORDER",
    "DASHBOARD_CARD_PADDING",
    "DASHBOARD_CARD_MARGIN",
    "WAVELENGTH_GRADIENT_CSS",
    "STATUS_PASS_COLOR",
    "STATUS_FAIL_COLOR",
    "STATUS_NEUTRAL_COLOR",
]


@dataclass(frozen=True)
class NumericControl:
    """Definition of one bounded numeric notebook control."""

    minimum: float
    maximum: float
    step: float
    default: float
    display_to_si: float = 1.0

    def to_si(self, value: float) -> float:
        """Convert a value in the control's display unit to SI."""
        return float(value) * self.display_to_si


CASE_OPTIONS = (
    ("Slit", "slit"),
    ("Rectangular aperture", "rectangle"),
    ("Circular aperture", "circle"),
    ("Multiple / mixed apertures", "multiple"),
)

SHAPE_OPTIONS = (
    ("Slit", "slit"),
    ("Rectangle", "rectangle"),
    ("Circle", "circle"),
)

SLIT_ORIENTATION_OPTIONS = (
    ("Vertical slit", "vertical"),
    ("Horizontal slit", "horizontal"),
)

VIEW_OPTIONS = (
    ("Simulation", "simulation", "image"),
    ("Graphs", "profiles", "chart-line"),
    ("Aperture-objective", "geometry", "arrows-left-right"),
    ("Aperture shape", "aperture", "shapes"),
)
DEFAULT_ACTIVE_VIEWS = ("simulation", "profiles")

DASHBOARD_CASE_OPTIONS = (
    ("Slit", "slit"),
    ("Rectangle", "rectangle"),
    ("Circle", "circle"),
    ("Multiple", "multiple"),
)
EXTRA_VIEW_OPTIONS = (
    ("Profiles", "profiles"),
    ("Detailed geometry", "geometry"),
)
DEFAULT_AUTO_UPDATE = True

WAVELENGTH_NM = NumericControl(380.0, 1_000.0, 0.01, 633.0, 1e-9)
REFRACTIVE_INDEX = NumericControl(1.0, 2.5, 0.01, 1.0)
DISTANCE_M = NumericControl(0.05, 100.0, 0.05, 2.0)
SCREEN_HALF_WIDTH_MM = NumericControl(0.1, 100.0, 0.1, 40.0, 1e-3)
ZOOM = NumericControl(1.0, 20.0, 0.5, 1.0)
RESOLUTION = NumericControl(101.0, 801.0, 50.0, 401.0)

# A common quantitative interpretation of the Fraunhofer requirement is
# N_F = R_max^2 / (lambda*z) <= 0.1.  The notebook exposes this threshold.
MAX_FRESNEL_NUMBER = NumericControl(0.001, 0.25, 0.001, 0.1)

# Display-only tone mapping. A value greater than one applies gamma=1/boost,
# revealing weak secondary maxima without changing calculated intensity.
SECONDARY_MAXIMA_BOOST = NumericControl(1.0, 8.0, 0.25, 4.0)

APERTURE_WIDTH_UM = NumericControl(5.0, 2_000.0, 5.0, 80.0, 1e-6)
APERTURE_HEIGHT_UM = NumericControl(5.0, 2_000.0, 5.0, 400.0, 1e-6)
CIRCLE_RADIUS_UM = NumericControl(5.0, 1_000.0, 5.0, 150.0, 1e-6)
CENTER_MM = NumericControl(-5.0, 5.0, 0.01, 0.0, 1e-3)

DEFAULT_SLIT_WIDTH_M = APERTURE_WIDTH_UM.to_si(APERTURE_WIDTH_UM.default)
DEFAULT_SLIT_LENGTH_M = APERTURE_HEIGHT_UM.to_si(APERTURE_HEIGHT_UM.default)
DEFAULT_RECTANGLE_WIDTH_M = APERTURE_WIDTH_UM.to_si(200.0)
DEFAULT_RECTANGLE_HEIGHT_M = APERTURE_HEIGHT_UM.to_si(
    APERTURE_HEIGHT_UM.default
)
DEFAULT_CIRCLE_RADIUS_M = CIRCLE_RADIUS_UM.to_si(CIRCLE_RADIUS_UM.default)

# Initial mixed-mode example and presentation settings used by the notebook.
DEFAULT_MULTIPLE_CENTERS_MM = (-0.15, 0.15)
SLIDER_WIDTH = "440px"
TEXT_WIDTH = "100px"
FIGURE_SIZE = (14.0, 5.5)
FIGURE_WIDTH_RATIOS = (1.15, 1.0)
GEOMETRY_FIGURE_SIZE = (12.0, 4.0)
APERTURE_FIGURE_SIZE = (6.5, 6.0)
APERTURE_VIEW_MARGIN = 1.2
GEOMETRY_APERTURE_HALF_HEIGHT = 0.35
GEOMETRY_SCREEN_HALF_HEIGHT = 0.8
GEOMETRY_VERTICAL_LIMIT = 1.0
COLOR_MAP = "inferno"
PROFILE_COLORS = ("tab:blue", "tab:orange")

# Friendly dashboard appearance.  All visual values live here so the notebook
# only wires controls and callbacks.
VISIBLE_WAVELENGTH_MIN_NM = 380.0
VISIBLE_WAVELENGTH_MAX_NM = 780.0
WAVELENGTH_COLOR_GAMMA = 0.8
INTENSITY_DISPLAY_GAMMA = 1.0 / SECONDARY_MAXIMA_BOOST.default
DASHBOARD_BACKGROUND = "#0b1020"
PANEL_BACKGROUND = "#02040a"
APERTURE_COLOR = "#f8fafc"
SCHEMATIC_BEAM_ALPHA = 0.28
SCALE_BAR_FRACTION = 0.22
SCALE_BAR_LINEWIDTH = 3.0
SCHEMATIC_FIGURE_SIZE = (13.0, 2.6)
DASHBOARD_PREVIEW_FIGURE_SIZE = (5.0, 4.6)
DASHBOARD_PROFILE_FIGURE_SIZE = (11.0, 3.6)
DASHBOARD_CONTROL_WIDTH = "330px"
DASHBOARD_PREVIEW_WIDTH = "460px"
DASHBOARD_SLIDER_WIDTH = "210px"
DASHBOARD_TEXT_WIDTH = "88px"
DASHBOARD_MAIN_COLUMNS = f"{DASHBOARD_CONTROL_WIDTH} minmax(0, 1fr)"
DASHBOARD_COLUMN_GAP = "12px"
DASHBOARD_MAX_WIDTH = "1500px"
DASHBOARD_CARD_BORDER = "1px solid #cbd5e1"
DASHBOARD_CARD_PADDING = "12px"
DASHBOARD_CARD_MARGIN = "6px"
WAVELENGTH_GRADIENT_CSS = (
    "linear-gradient(90deg, #6f00ff 0%, #004cff 18%, #00b7ff 31%, "
    "#00d05a 45%, #ffe600 62%, #ff7a00 78%, #e00000 100%)"
)
STATUS_PASS_COLOR = "#16794b"
STATUS_FAIL_COLOR = "#b42318"
STATUS_NEUTRAL_COLOR = "#475467"
