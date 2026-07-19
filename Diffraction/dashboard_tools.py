"""Visual helpers for the friendly Fraunhofer notebook dashboard.

The functions in this module only render presentation views.  They do not
change aperture definitions, the Fraunhofer validity criterion, or calculated
intensity values.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle

import diffraction_config as config
import diffraction_engine as engine

__all__ = [
    "wavelength_to_rgb",
    "intensity_to_rgb",
    "nice_scale_length",
    "add_scale_bar",
    "draw_optical_schematic",
    "draw_aperture_preview",
    "draw_diffraction_preview",
    "draw_far_field_unavailable",
]


def wavelength_to_rgb(wavelength_nm: float) -> tuple[float, float, float]:
    """Approximate a visible display color for a wavelength in nanometres.

    Values outside the visible interval are clamped to its nearest endpoint so
    infrared and ultraviolet inputs still have a stable teaching-display color.
    """
    wavelength_nm = float(wavelength_nm)
    if not np.isfinite(wavelength_nm) or wavelength_nm <= 0.0:
        raise ValueError("wavelength_nm must be a positive finite value.")

    wavelength = float(
        np.clip(
            wavelength_nm,
            config.VISIBLE_WAVELENGTH_MIN_NM,
            config.VISIBLE_WAVELENGTH_MAX_NM,
        )
    )

    if wavelength < 440.0:
        red = -(wavelength - 440.0) / 60.0
        green = 0.0
        blue = 1.0
    elif wavelength < 490.0:
        red = 0.0
        green = (wavelength - 440.0) / 50.0
        blue = 1.0
    elif wavelength < 510.0:
        red = 0.0
        green = 1.0
        blue = -(wavelength - 510.0) / 20.0
    elif wavelength < 580.0:
        red = (wavelength - 510.0) / 70.0
        green = 1.0
        blue = 0.0
    elif wavelength < 645.0:
        red = 1.0
        green = -(wavelength - 645.0) / 65.0
        blue = 0.0
    else:
        red = 1.0
        green = 0.0
        blue = 0.0

    edge = 1.0
    if wavelength < 420.0:
        edge = 0.3 + 0.7 * (wavelength - 380.0) / 40.0
    elif wavelength > 700.0:
        edge = 0.3 + 0.7 * (780.0 - wavelength) / 80.0
    edge = float(np.clip(edge, 0.3, 1.0))
    gamma = config.WAVELENGTH_COLOR_GAMMA
    return tuple(
        float((component * edge) ** gamma)
        for component in (red, green, blue)
    )


def intensity_to_rgb(
    intensity: np.ndarray,
    wavelength_nm: float,
    gamma: float = config.INTENSITY_DISPLAY_GAMMA,
) -> np.ndarray:
    """Map a nonnegative intensity array to a wavelength-tinted RGB image."""
    values = np.asarray(intensity, dtype=float)
    if values.ndim != 2:
        raise ValueError("intensity must be a two-dimensional array.")
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("gamma must be a positive finite value.")

    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.clip(values, 0.0, None)
    maximum = float(values.max(initial=0.0))
    if maximum > 0.0:
        values = values / maximum
    brightness = values**gamma
    color = np.asarray(wavelength_to_rgb(wavelength_nm), dtype=float)
    return np.clip(brightness[..., np.newaxis] * color, 0.0, 1.0)


def nice_scale_length(span: float, fraction: float = config.SCALE_BAR_FRACTION) -> float:
    """Choose a readable 1/2/5 scale-bar length for a displayed span."""
    span = float(span)
    fraction = float(fraction)
    if not np.isfinite(span) or span <= 0.0:
        raise ValueError("span must be a positive finite value.")
    if not np.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be between zero and one.")

    target = span * fraction
    exponent = np.floor(np.log10(target))
    base = 10.0**exponent
    normalized = target / base
    if normalized >= 5.0:
        multiplier = 5.0
    elif normalized >= 2.0:
        multiplier = 2.0
    else:
        multiplier = 1.0
    return float(multiplier * base)


def add_scale_bar(
    axis: Axes,
    extent: tuple[float, float, float, float],
    unit: str,
    color: str = "white",
) -> float:
    """Draw a lower-left physical scale bar and return its length."""
    x_min, x_max, y_min, y_max = extent
    x_span = x_max - x_min
    y_span = y_max - y_min
    length = nice_scale_length(x_span)
    x_start = x_min + 0.08 * x_span
    y_start = y_min + 0.09 * y_span
    axis.plot(
        [x_start, x_start + length],
        [y_start, y_start],
        color=color,
        linewidth=config.SCALE_BAR_LINEWIDTH,
        solid_capstyle="butt",
    )
    axis.text(
        x_start + length / 2.0,
        y_start + 0.035 * y_span,
        f"{length:g} {unit}",
        color=color,
        ha="center",
        va="bottom",
        fontsize=9,
    )
    return length


def draw_optical_schematic(
    axis: Axes,
    apertures: Sequence[engine.Aperture],
    distance: float,
    wavelength_nm: float,
    report: engine.FarFieldReport,
) -> None:
    """Draw a compact source-to-aperture-to-screen teaching schematic."""
    if not apertures:
        raise ValueError("At least one aperture is required.")
    color = wavelength_to_rgb(wavelength_nm)
    source_x, aperture_x, screen_x = 0.12, 0.48, 0.90
    source_width, source_height = 0.17, 0.22

    axis.set_facecolor("#f8fafc")
    source = FancyBboxPatch(
        (source_x - source_width / 2.0, 0.5 - source_height / 2.0),
        source_width,
        source_height,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        facecolor="#d0d5dd",
        edgecolor="#344054",
        linewidth=1.2,
    )
    axis.add_patch(source)
    axis.add_patch(
        Circle((source_x, 0.5), source_height * 0.27, color=color, alpha=0.95)
    )

    aperture_plane = Rectangle(
        (aperture_x - 0.012, 0.37),
        0.024,
        0.26,
        facecolor="#101828",
        edgecolor="#344054",
    )
    screen = Polygon(
        [
            (screen_x - 0.018, 0.30),
            (screen_x + 0.018, 0.33),
            (screen_x + 0.018, 0.67),
            (screen_x - 0.018, 0.70),
        ],
        closed=True,
        facecolor="#101828",
        edgecolor="#344054",
    )
    axis.add_patch(aperture_plane)
    axis.add_patch(screen)

    beam_start = source_x + source_width / 2.0
    axis.fill(
        [beam_start, aperture_x, aperture_x, beam_start],
        [0.47, 0.485, 0.515, 0.53],
        color=color,
        alpha=config.SCHEMATIC_BEAM_ALPHA,
        linewidth=0.0,
    )
    axis.fill(
        [aperture_x, screen_x, screen_x, aperture_x],
        [0.485, 0.33, 0.67, 0.515],
        color=color,
        alpha=config.SCHEMATIC_BEAM_ALPHA,
        linewidth=0.0,
    )
    axis.plot([beam_start, screen_x], [0.5, 0.5], color=color, linewidth=1.4)
    axis.scatter([screen_x], [0.5], s=180, color=color, alpha=0.55)

    axis.annotate(
        "",
        xy=(screen_x, 0.20),
        xytext=(aperture_x, 0.20),
        arrowprops={"arrowstyle": "<->", "color": "#475467"},
    )
    axis.text(
        (aperture_x + screen_x) / 2.0,
        0.14,
        f"z = {distance:g} m",
        ha="center",
        va="center",
        color="#344054",
    )
    axis.text(source_x, 0.76, f"Source\n{wavelength_nm:g} nm", ha="center")
    axis.text(
        aperture_x,
        0.76,
        f"Aperture\n{len(apertures)} opening(s)",
        ha="center",
    )
    axis.text(screen_x, 0.76, "Observation\nplane", ha="center")
    state = "FAR FIELD" if report.is_valid else "NEAR FIELD"
    state_color = (
        config.STATUS_PASS_COLOR if report.is_valid else config.STATUS_FAIL_COLOR
    )
    axis.text(
        0.99,
        0.06,
        f"{state}  |  N_F={report.fresnel_number:.3g}",
        ha="right",
        va="bottom",
        color=state_color,
        fontweight="bold",
    )
    axis.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    axis.axis("off")


def draw_aperture_preview(
    axis: Axes,
    apertures: Sequence[engine.Aperture],
) -> tuple[float, float, float, float]:
    """Render all openings as white shapes on a dark aperture plane."""
    if not apertures:
        raise ValueError("At least one aperture is required.")
    millimetres = config.CENTER_MM.display_to_si
    support = engine.aperture_support_radius(apertures)
    limit = config.APERTURE_VIEW_MARGIN * support / millimetres
    extent = (-limit, limit, -limit, limit)

    axis.set_facecolor(config.PANEL_BACKGROUND)
    for aperture in apertures:
        center_x = aperture.center_x / millimetres
        center_y = aperture.center_y / millimetres
        if aperture.kind == engine.ApertureKind.CIRCLE:
            assert aperture.radius is not None
            patch = Circle(
                (center_x, center_y),
                aperture.radius / millimetres,
                facecolor=config.APERTURE_COLOR,
                edgecolor=config.APERTURE_COLOR,
            )
        else:
            assert aperture.width is not None and aperture.height is not None
            patch = Rectangle(
                (
                    center_x - aperture.width / (2.0 * millimetres),
                    center_y - aperture.height / (2.0 * millimetres),
                ),
                aperture.width / millimetres,
                aperture.height / millimetres,
                facecolor=config.APERTURE_COLOR,
                edgecolor=config.APERTURE_COLOR,
            )
        axis.add_patch(patch)

    axis.set(xlim=extent[:2], ylim=extent[2:], aspect="equal")
    axis.set_title("APERTURE PLANE", color="white", fontsize=11, pad=10)
    axis.axis("off")
    add_scale_bar(axis, extent, "mm")
    return extent


def draw_diffraction_preview(
    axis: Axes,
    X: np.ndarray,
    Y: np.ndarray,
    intensity: np.ndarray,
    wavelength_nm: float,
    gamma: float = config.INTENSITY_DISPLAY_GAMMA,
) -> tuple[float, float, float, float]:
    """Render a wavelength-colored preview with display-only tone mapping."""
    millimetres = config.SCREEN_HALF_WIDTH_MM.display_to_si
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    extent = (
        float(X.min() / millimetres),
        float(X.max() / millimetres),
        float(Y.min() / millimetres),
        float(Y.max() / millimetres),
    )
    rgb = intensity_to_rgb(intensity, wavelength_nm, gamma=gamma)
    axis.set_facecolor(config.PANEL_BACKGROUND)
    axis.imshow(rgb, origin="lower", extent=extent, aspect="equal")
    axis.set(xlim=extent[:2], ylim=extent[2:])
    axis.set_title("FRAUNHOFER PATTERN", color="white", fontsize=11, pad=10)
    axis.axis("off")
    add_scale_bar(axis, extent, "mm")
    return extent


def draw_far_field_unavailable(
    axis: Axes,
    report: engine.FarFieldReport,
) -> None:
    """Render a friendly warning in place of an invalid diffraction preview."""
    axis.set_facecolor(config.PANEL_BACKGROUND)
    axis.text(
        0.5,
        0.62,
        "Fraunhofer preview unavailable",
        ha="center",
        va="center",
        color="white",
        fontsize=13,
        fontweight="bold",
        transform=axis.transAxes,
    )
    axis.text(
        0.5,
        0.43,
        (
            f"N_F = {report.fresnel_number:.3g} exceeds "
            f"{report.max_fresnel_number:.3g}\n"
            f"Increase distance to at least {report.required_distance:.3g} m"
        ),
        ha="center",
        va="center",
        color="#fda29b",
        fontsize=10,
        transform=axis.transAxes,
    )
    axis.set_title("OBSERVATION PLANE", color="white", fontsize=11, pad=10)
    axis.axis("off")
