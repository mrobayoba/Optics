"""Visual helpers for the Fresnel slit and straight-edge dashboard."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

import dashboard_tools as shared_visuals
import diffraction_config as shared_config
import fresnel_config as config
import fresnel_engine as engine

__all__ = [
    "draw_optical_schematic",
    "draw_aperture_preview",
    "draw_fresnel_preview",
    "draw_cornu_construction",
]


def draw_optical_schematic(
    axis: Axes,
    case: engine.FresnelCase | str,
    illumination: engine.IlluminationKind | str,
    distance: float,
    source_distance: float | None,
    wavelength_nm: float,
    report: engine.FresnelReport,
) -> None:
    """Draw source, aperture, and screen with Fresnel-regime information."""
    selected_case = engine.FresnelCase(case)
    source_kind = engine.IlluminationKind(illumination)
    color = shared_visuals.wavelength_to_rgb(wavelength_nm)
    source_x, aperture_x, screen_x = 0.12, 0.48, 0.90

    axis.set_facecolor("#f8fafc")
    if source_kind == engine.IlluminationKind.POINT:
        axis.scatter([source_x], [0.5], s=420, color=color, alpha=0.18)
        axis.scatter([source_x], [0.5], s=95, color=color, edgecolor="none")
        axis.fill(
            [source_x, aperture_x, aperture_x],
            [0.5, 0.35, 0.65],
            color=color,
            alpha=shared_config.SCHEMATIC_BEAM_ALPHA,
            linewidth=0.0,
        )
        source_label = f"Point source\nd = {source_distance:g} m"
    else:
        for offset in (-0.07, 0.0, 0.07):
            axis.plot(
                [source_x - 0.012, source_x + 0.012],
                [0.5 + offset, 0.5 + offset],
                color=color,
                linewidth=2.0,
            )
        axis.fill(
            [source_x, aperture_x, aperture_x, source_x],
            [0.36, 0.36, 0.64, 0.64],
            color=color,
            alpha=shared_config.SCHEMATIC_BEAM_ALPHA,
            linewidth=0.0,
        )
        source_label = "Plane wave"

    axis.add_patch(
        Rectangle(
            (aperture_x - 0.012, 0.36),
            0.024,
            0.28,
            facecolor="#101828",
            edgecolor="#344054",
        )
    )
    axis.add_patch(
        Rectangle(
            (screen_x - 0.01, 0.30),
            0.02,
            0.40,
            facecolor="#101828",
            edgecolor="#344054",
        )
    )
    axis.fill(
        [aperture_x, screen_x, screen_x, aperture_x],
        [0.45, 0.32, 0.68, 0.55],
        color=color,
        alpha=shared_config.SCHEMATIC_BEAM_ALPHA,
        linewidth=0.0,
    )
    axis.plot([source_x, screen_x], [0.5, 0.5], color=color, linewidth=1.2)
    axis.scatter([screen_x], [0.5], s=170, color=color, alpha=0.55)

    axis.annotate(
        "",
        xy=(screen_x, 0.20),
        xytext=(aperture_x, 0.20),
        arrowprops={"arrowstyle": "<->", "color": "#475467"},
    )
    axis.text(
        (aperture_x + screen_x) / 2.0,
        0.14,
        f"D = {distance:g} m",
        ha="center",
        va="center",
        color="#344054",
    )
    axis.text(
        source_x,
        0.78,
        f"{source_label}\n{wavelength_nm:.2f} nm",
        ha="center",
    )
    axis.text(
        aperture_x,
        0.78,
        "Single slit"
        if selected_case == engine.FresnelCase.SLIT
        else "Straight edge",
        ha="center",
    )
    axis.text(screen_x, 0.78, "Observation\nplane", ha="center")
    regime_text = report.regime.upper()
    if report.fresnel_number is not None:
        regime_text += f"  |  N_F={report.fresnel_number:.3g}"
    else:
        regime_text += (
            f"  |  l_F={report.characteristic_length * 1e3:.3g} mm"
        )
    axis.text(
        0.99,
        0.05,
        regime_text,
        ha="right",
        va="bottom",
        color=shared_config.STATUS_PASS_COLOR,
        fontweight="bold",
    )
    axis.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    axis.axis("off")


def draw_aperture_preview(
    axis: Axes,
    case: engine.FresnelCase | str,
    *,
    slit_width: float | None = None,
    orientation: str = config.DEFAULT_ORIENTATION,
    shadow_side: str = config.DEFAULT_SHADOW_SIDE,
) -> tuple[float, float, float, float]:
    """Draw a slit or opaque half-plane on the dark aperture card."""
    selected_case = engine.FresnelCase(case)
    if orientation not in ("vertical", "horizontal"):
        raise ValueError("orientation must be 'vertical' or 'horizontal'.")
    if shadow_side not in ("positive", "negative"):
        raise ValueError("shadow_side must be 'positive' or 'negative'.")

    millimetres = 1e-3
    if selected_case == engine.FresnelCase.SLIT:
        if slit_width is None or slit_width <= 0.0:
            raise ValueError("slit_width must be positive for a slit preview.")
        width_mm = slit_width / millimetres
        limit = max(0.75, 1.5 * width_mm)
    else:
        limit = 1.5
    extent = (-limit, limit, -limit, limit)

    axis.set_facecolor(shared_config.PANEL_BACKGROUND)
    if selected_case == engine.FresnelCase.SLIT:
        width_mm = float(slit_width / millimetres)
        if orientation == "vertical":
            opening = Rectangle(
                (-width_mm / 2.0, -0.85 * limit),
                width_mm,
                1.7 * limit,
                facecolor=shared_config.APERTURE_COLOR,
                edgecolor=shared_config.APERTURE_COLOR,
            )
        else:
            opening = Rectangle(
                (-0.85 * limit, -width_mm / 2.0),
                1.7 * limit,
                width_mm,
                facecolor=shared_config.APERTURE_COLOR,
                edgecolor=shared_config.APERTURE_COLOR,
            )
        axis.add_patch(opening)
        title = "SLIT APERTURE"
    else:
        open_negative = shadow_side == "positive"
        if orientation == "vertical":
            x_start = -limit if open_negative else 0.0
            opening = Rectangle(
                (x_start, -limit),
                limit,
                2.0 * limit,
                facecolor=shared_config.APERTURE_COLOR,
                edgecolor=shared_config.APERTURE_COLOR,
            )
            axis.axvline(0.0, color="#98a2b3", linewidth=2.0)
        else:
            y_start = -limit if open_negative else 0.0
            opening = Rectangle(
                (-limit, y_start),
                2.0 * limit,
                limit,
                facecolor=shared_config.APERTURE_COLOR,
                edgecolor=shared_config.APERTURE_COLOR,
            )
            axis.axhline(0.0, color="#98a2b3", linewidth=2.0)
        axis.add_patch(opening)
        title = "OPAQUE STRAIGHT EDGE"

    axis.set(xlim=extent[:2], ylim=extent[2:], aspect="equal")
    axis.set_title(title, color="white", fontsize=11, pad=10)
    axis.axis("off")
    shared_visuals.add_scale_bar(axis, extent, "mm")
    return extent


def draw_fresnel_preview(
    axis: Axes,
    X: np.ndarray,
    Y: np.ndarray,
    intensity: np.ndarray,
    wavelength_nm: float,
    *,
    gamma: float = shared_config.INTENSITY_DISPLAY_GAMMA,
) -> tuple[float, float, float, float]:
    """Draw a wavelength-colored Fresnel camera image."""
    millimetres = 1e-3
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    extent = (
        float(X.min() / millimetres),
        float(X.max() / millimetres),
        float(Y.min() / millimetres),
        float(Y.max() / millimetres),
    )
    rgb = shared_visuals.intensity_to_rgb(
        intensity, wavelength_nm, gamma=gamma
    )
    axis.set_facecolor(shared_config.PANEL_BACKGROUND)
    axis.imshow(rgb, origin="lower", extent=extent, aspect="equal")
    axis.set(xlim=extent[:2], ylim=extent[2:])
    axis.set_title("FRESNEL PATTERN", color="white", fontsize=11, pad=10)
    axis.axis("off")
    shared_visuals.add_scale_bar(axis, extent, "mm")
    return extent


def draw_cornu_construction(
    axis: Axes,
    case: engine.FresnelCase | str,
    probe_coordinate: float,
    wavelength_vacuum: float,
    distance: float,
    *,
    slit_width: float | None = None,
    n: float = 1.0,
    illumination: engine.IlluminationKind | str = (
        engine.IlluminationKind.PLANE
    ),
    source_distance: float | None = None,
    shadow_side: str = config.DEFAULT_SHADOW_SIDE,
    u_limit: float = config.CORNU_U_LIMIT,
) -> tuple[complex, complex]:
    """Plot the Cornu spiral and return the active chord endpoints."""
    selected_case = engine.FresnelCase(case)
    u_values = np.linspace(-u_limit, u_limit, 2_001)
    spiral = engine.cornu_point(u_values)
    axis.plot(spiral.real, spiral.imag, color="#667085", linewidth=1.5)

    if selected_case == engine.FresnelCase.SLIT:
        lower, upper = engine.slit_boundary_coordinates(
            probe_coordinate,
            slit_width,
            wavelength_vacuum,
            distance,
            n=n,
            illumination=illumination,
            source_distance=source_distance,
        )
        start = complex(engine.cornu_point(lower))
        finish = complex(engine.cornu_point(upper))
        label = f"slit chord: u1={float(lower):.3g}, u2={float(upper):.3g}"
    else:
        signed_probe = (
            probe_coordinate
            if shadow_side == "positive"
            else -probe_coordinate
        )
        u = engine.screen_fresnel_coordinate(
            signed_probe,
            wavelength_vacuum,
            distance,
            n=n,
            illumination=illumination,
            source_distance=source_distance,
        )
        start = complex(engine.cornu_point(u))
        finish = complex(0.5, 0.5)
        label = f"edge chord: u={float(u):.3g} to +infinity"

    axis.plot(
        [start.real, finish.real],
        [start.imag, finish.imag],
        color="#d92d20",
        linewidth=2.4,
        marker="o",
    )
    axis.text(
        0.02,
        0.98,
        label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    axis.set(
        title="Cornu (clothoid) construction",
        xlabel="C(u)",
        ylabel="S(u)",
        aspect="equal",
    )
    axis.grid(alpha=0.25)
    return start, finish
