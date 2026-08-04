"""Presentation helpers for the paraxial-optics notebook dashboard."""

from __future__ import annotations

from html import escape
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

import paraxial_config as config
import paraxial_engine as engine
import paraxial_tools as tools

__all__ = [
    "matrix_html",
    "factor_product_html",
    "values_html",
    "draw_optical_schematic",
]


def matrix_html(
    matrix: np.ndarray,
    decimals: int = config.DEFAULT_DISPLAY_DECIMALS,
) -> str:
    """Render a 2x2 numeric matrix as a compact HTML table."""
    values = np.asarray(matrix, dtype=float)
    if values.shape != (2, 2):
        raise ValueError("matrix must be 2x2.")
    digits = int(decimals)
    if digits < config.MIN_DISPLAY_DECIMALS or digits > config.MAX_DISPLAY_DECIMALS:
        raise ValueError(
            f"decimals must be between {config.MIN_DISPLAY_DECIMALS} and "
            f"{config.MAX_DISPLAY_DECIMALS}."
        )
    rows = []
    for row in values:
        cells = "".join(
            (
                "<td style='padding:0.12rem 0.42rem;text-align:right;"
                "font-family:Consolas,monospace'>"
                f"{value:.{digits}f}</td>"
            )
            for value in row
        )
        rows.append(f"<tr>{cells}</tr>")
    return (
        "<span style='display:inline-block;border-left:2px solid #334155;"
        "border-right:2px solid #334155;padding:0.08rem 0.15rem;"
        "vertical-align:middle'><table style='border-collapse:collapse'>"
        f"{''.join(rows)}</table></span>"
    )


def factor_product_html(
    title: str,
    factors: Sequence[engine.MatrixFactor],
    result: np.ndarray,
    result_label: str,
    checks: Sequence[tuple[str, str]] = (),
    decimals: int = config.DEFAULT_DISPLAY_DECIMALS,
) -> str:
    """Render every factor, operator, result, and numeric check in order."""
    factor_blocks = []
    for factor in factors:
        details = (
            f"<div style='font-size:0.76rem;color:#475569'>{escape(factor.description)}</div>"
            if factor.description
            else ""
        )
        factor_blocks.append(
            "<div style='display:inline-flex;flex-direction:column;"
            "align-items:center;gap:0.18rem'>"
            f"<strong>{escape(factor.label)}</strong>"
            f"{matrix_html(factor.matrix, decimals)}{details}</div>"
        )
    product = (
        "<span style='font-size:1.25rem;padding:0 0.3rem'>×</span>"
    ).join(factor_blocks)
    if not product:
        product = "<em>Identity (empty factor list)</em>"
    checks_html = "".join(
        f"<span style='margin-right:1rem'><strong>{escape(label)}:</strong> "
        f"{escape(value)}</span>"
        for label, value in checks
    )
    return (
        "<div class='paraxial-matrix-panel'>"
        f"<h4 style='margin:0 0 0.55rem'>{escape(title)}</h4>"
        "<div style='display:flex;align-items:center;gap:0.25rem;"
        "overflow-x:auto;padding-bottom:0.45rem'>"
        f"{product}"
        "<span style='font-size:1.25rem;padding:0 0.3rem'>=</span>"
        "<div style='display:inline-flex;flex-direction:column;"
        "align-items:center;gap:0.18rem'>"
        f"<strong>{escape(result_label)}</strong>"
        f"{matrix_html(result, decimals)}</div>"
        "</div>"
        f"<div style='font-size:0.84rem;color:#334155'>{checks_html}</div>"
        "</div>"
    )


def values_html(
    title: str,
    values: Sequence[tuple[str, str]],
    note: str = "",
) -> str:
    """Render named scalar results without hiding their units."""
    items = "".join(
        "<div style='min-width:8.5rem'>"
        f"<div style='font-size:0.75rem;color:#64748b'>{escape(label)}</div>"
        f"<strong>{escape(value)}</strong></div>"
        for label, value in values
    )
    note_html = (
        f"<div class='paraxial-note' style='margin-top:0.55rem'>{escape(note)}</div>"
        if note
        else ""
    )
    return (
        f"<h4 style='margin-bottom:0.45rem'>{escape(title)}</h4>"
        "<div style='display:flex;flex-wrap:wrap;gap:0.65rem 1rem'>"
        f"{items}</div>{note_html}"
    )


def _draw_arrow(
    ax: Axes,
    z: float,
    height: float,
    color: str,
    label: str,
) -> None:
    ax.annotate(
        "",
        xy=(z, height),
        xytext=(z, 0.0),
        arrowprops={"arrowstyle": "-|>", "color": color, "lw": 2.2},
    )
    ax.text(
        z,
        height + np.sign(height or 1.0) * 0.04 * max(abs(height), 1e-4),
        label,
        color=color,
        ha="center",
        va="bottom" if height >= 0.0 else "top",
        fontweight="bold",
    )


def draw_optical_schematic(
    ax: Axes,
    elements: Sequence[tools.OpticalElement],
    cardinal: engine.CardinalPoints | None = None,
    conjugate: engine.ConjugateReport | None = None,
    object_height: float = 0.02,
    trace: tools.RayTrace | None = None,
    length_scale: float = 1.0,
    length_unit: str = "m",
) -> Axes:
    """Draw system vertices, elements, principal planes, object/image, and rays.

    Geometry inputs are SI metres. ``length_scale`` converts plotted coordinates
    into the dashboard display unit (1 for m, 1000 for mm).
    """
    scale = float(length_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("length_scale must be a positive finite value.")

    geometry = tools.system_geometry(elements)
    colors = config.COLORS

    coordinates = [geometry.start_vertex, geometry.finish_vertex]
    h_position = None
    h_prime_position = None
    object_position = None
    image_position = None
    image_height = None

    if cardinal is not None:
        h_position = geometry.start_vertex - cardinal.object_principal_offset
        h_prime_position = (
            geometry.finish_vertex + cardinal.image_principal_offset
        )
        coordinates.extend((h_position, h_prime_position))
    if cardinal is not None and conjugate is not None:
        object_position = h_position - conjugate.object_distance
        image_position = h_prime_position + conjugate.image_distance
        image_height = object_height * conjugate.lateral_magnification
        coordinates.extend((object_position, image_position))

    def _x(value: float) -> float:
        return float(value) * scale

    def _y(value: float) -> float:
        return float(value) * scale

    x_min, x_max = min(coordinates), max(coordinates)
    span = max(x_max - x_min, 1.0 / scale)
    padding = 0.08 * span
    ax.axhline(0.0, color=colors["axis"], lw=1.0, alpha=0.8)

    for placement in geometry.placements:
        if placement.kind == tools.ElementKind.TRANSLATION:
            ax.plot(
                [_x(placement.start), _x(placement.finish)],
                [0.0, 0.0],
                color=colors["translation"],
                lw=3.0,
                alpha=0.45,
            )
            continue
        if placement.kind == tools.ElementKind.REFLECTION:
            color = colors["reflection"]
            linestyle = "-."
        elif placement.kind in (
            tools.ElementKind.THIN_LENS,
            tools.ElementKind.THICK_LENS,
        ):
            color = colors["lens"]
            linestyle = "-"
        else:
            color = colors["refraction"]
            linestyle = "-"
        ax.axvline(_x(placement.start), color=color, lw=2.0, ls=linestyle, alpha=0.9)
        if placement.finish != placement.start:
            ax.axvline(_x(placement.finish), color=color, lw=2.0, alpha=0.9)
        ax.text(
            _x((placement.start + placement.finish) / 2.0),
            0.02,
            placement.label,
            color=color,
            ha="center",
            va="bottom",
            transform=ax.get_xaxis_transform(),
            fontsize=8,
        )

    ax.axvline(_x(geometry.start_vertex), color=colors["axis"], lw=1.0, ls=":")
    ax.axvline(_x(geometry.finish_vertex), color=colors["axis"], lw=1.0, ls=":")
    ax.text(
        _x(geometry.start_vertex),
        0.96,
        "V",
        transform=ax.get_xaxis_transform(),
        ha="center",
    )
    ax.text(
        _x(geometry.finish_vertex),
        0.96,
        "V′",
        transform=ax.get_xaxis_transform(),
        ha="center",
    )

    if h_position is not None and h_prime_position is not None:
        ax.axvline(
            _x(h_position),
            color=colors["principal"],
            lw=1.4,
            ls="--",
        )
        ax.axvline(
            _x(h_prime_position),
            color=colors["principal"],
            lw=1.4,
            ls="--",
        )
        ax.text(
            _x(h_position),
            0.90,
            "H",
            color=colors["principal"],
            transform=ax.get_xaxis_transform(),
            ha="center",
        )
        ax.text(
            _x(h_prime_position),
            0.90,
            "H′",
            color=colors["principal"],
            transform=ax.get_xaxis_transform(),
            ha="center",
        )

    plotted_heights = [abs(_y(float(object_height)))]
    if trace is not None:
        for ray in trace.heights:
            ax.plot(
                [_x(z) for z in trace.path_positions],
                [_y(height) for height in ray],
                color=colors["translation"],
                lw=0.9,
                alpha=0.35,
            )
        plotted_heights.append(float(np.max(np.abs(trace.heights))) * scale)

    if (
        object_position is not None
        and image_position is not None
        and image_height is not None
    ):
        _draw_arrow(
            ax,
            _x(object_position),
            _y(object_height),
            colors["object"],
            "Object",
        )
        _draw_arrow(
            ax,
            _x(image_position),
            _y(image_height),
            colors["image"],
            "Image" if conjugate.is_real else "Virtual image",
        )
        plotted_heights.append(abs(_y(image_height)))

        ray_scale = max(abs(_y(object_height)), abs(_y(image_height)), 1e-3)
        intercepts = np.linspace(-0.6, 1.35, config.SCHEMATIC_RAY_COUNT) * ray_scale
        ray_style = "-" if conjugate.is_real else "--"
        for intercept in intercepts:
            ax.plot(
                [_x(object_position), _x(h_position)],
                [_y(object_height), intercept],
                color=colors["ray"],
                lw=1.0,
                alpha=0.7,
            )
            ax.plot(
                [_x(h_position), _x(h_prime_position)],
                [intercept, intercept],
                color=colors["ray"],
                lw=0.8,
                alpha=0.5,
            )
            ax.plot(
                [_x(h_prime_position), _x(image_position)],
                [intercept, _y(image_height)],
                color=(
                    colors["ray"]
                    if conjugate.is_real
                    else colors["virtual_ray"]
                ),
                lw=1.0,
                ls=ray_style,
                alpha=0.7,
            )

    y_extent = max(plotted_heights + [1e-3]) * 1.8
    ax.set_xlim(_x(x_min - padding), _x(x_max + padding))
    ax.set_ylim(-y_extent, y_extent)
    ax.set_xlabel(f"Optical path coordinate ({length_unit})")
    ax.set_ylabel(f"Ray height x ({length_unit})")
    ax.set_title("Paraxial system, principal planes, and conjugate image")
    ax.grid(axis="x", alpha=0.15)
    return ax
