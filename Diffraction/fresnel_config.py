"""Configuration for the slit and straight-edge Fresnel simulator."""

from __future__ import annotations

import diffraction_config as shared

__all__ = [
    "CASE_OPTIONS",
    "ILLUMINATION_OPTIONS",
    "ORIENTATION_OPTIONS",
    "SHADOW_SIDE_OPTIONS",
    "EXTRA_VIEW_OPTIONS",
    "DEFAULT_AUTO_UPDATE",
    "DEFAULT_CASE",
    "DEFAULT_ILLUMINATION",
    "DEFAULT_ORIENTATION",
    "DEFAULT_SHADOW_SIDE",
    "SOURCE_DISTANCE_M",
    "DISTANCE_M",
    "SLIT_WIDTH_MM",
    "SCREEN_HALF_WIDTH_MM",
    "RESOLUTION",
    "ZOOM",
    "SECONDARY_MAXIMA_BOOST",
    "CORNU_PROBE_MM",
    "CORNU_U_LIMIT",
    "NEAR_FIELD_LIMIT",
    "NEAR_FIELD_LOG_BASE",
    "NEAR_FIELD_LOG_STEP",
    "FAR_FIELD_LIMIT",
    "FRESNEL_TRANSITION_LIMIT",
]


CASE_OPTIONS = (("Slit", "slit"), ("Edge", "edge"))
ILLUMINATION_OPTIONS = (
    ("Plane wave", "plane"),
    ("Point source", "point"),
)
ORIENTATION_OPTIONS = shared.SLIT_ORIENTATION_OPTIONS
SHADOW_SIDE_OPTIONS = (
    ("Positive side", "positive"),
    ("Negative side", "negative"),
)
EXTRA_VIEW_OPTIONS = (
    ("Profiles", "profiles"),
    ("Cornu spiral", "cornu"),
)

DEFAULT_AUTO_UPDATE = True
DEFAULT_CASE = "slit"
DEFAULT_ILLUMINATION = "plane"
DEFAULT_ORIENTATION = "vertical"
DEFAULT_SHADOW_SIDE = "positive"

# Fresnel distances and openings are commonly shorter/larger than the
# Fraunhofer dashboard defaults. Display units remain explicit.
SOURCE_DISTANCE_M = shared.NumericControl(0.02, 100.0, 0.01, 1.0)
DISTANCE_M = shared.NumericControl(0.01, 20.0, 0.01, 0.5)
SLIT_WIDTH_MM = shared.NumericControl(0.02, 5.0, 0.01, 1.0, 1e-3)
SCREEN_HALF_WIDTH_MM = shared.NumericControl(0.1, 25.0, 0.1, 5.0, 1e-3)
RESOLUTION = shared.NumericControl(101.0, 1_001.0, 50.0, 401.0)
ZOOM = shared.NumericControl(1.0, 20.0, 0.5, 1.0)
SECONDARY_MAXIMA_BOOST = shared.SECONDARY_MAXIMA_BOOST
CORNU_PROBE_MM = shared.NumericControl(-25.0, 25.0, 0.05, 0.0, 1e-3)

CORNU_U_LIMIT = 5.0
# User-selectable teaching criterion. The dashboard renders this on a
# logarithmic scale; the fixed constants below retain conventional labels.
NEAR_FIELD_LIMIT = shared.NumericControl(1e-4, 10.0, 1e-4, 0.1)
NEAR_FIELD_LOG_BASE = 10.0
NEAR_FIELD_LOG_STEP = 0.01
FAR_FIELD_LIMIT = 0.1
FRESNEL_TRANSITION_LIMIT = 1.0
