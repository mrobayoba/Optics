"""Sensor / camera utilities for the Michelson simulator.

These helpers turn the physics in :mod:`michelson_engine` into something a
camera would record: a sampled 2-D intensity image, an intensity profile taken
perpendicular to the fringes, and the contrast (visibility) measurement used to
estimate the coherence length and compare it with theory.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

import michelson_engine as eng

__all__ = [
    "camera_grid",
    "simulate_pattern",
    "extract_perpendicular_profile",
    "visibility_from_profile",
    "sweep_visibility",
    "measure_coherence_length",
]


def camera_grid(sensor_size=0.01, resolution=400, zoom=1.0):
    """Square sensor coordinate grid centered on the optical axis.

    Parameters
    ----------
    sensor_size : float
        Full physical width/height of the sensor at ``zoom = 1`` (meters).
    resolution : int
        Number of pixels per side.
    zoom : float
        Magnification; the visible half-extent is ``sensor_size / (2 * zoom)``.

    Returns
    -------
    X, Y : ndarray
        ``np.meshgrid`` coordinate arrays of shape ``(resolution, resolution)``.
    """
    half = sensor_size / (2.0 * zoom)
    axis = np.linspace(-half, half, resolution)
    return np.meshgrid(axis, axis)


def simulate_pattern(
    wavelength,
    dL0,
    tilt_x,
    tilt_y,
    n=1.0,
    I0=1.0,
    source="mono",
    dlambda=None,
    profile="rect",
    sensor_size=0.01,
    resolution=400,
    zoom=1.0,
):
    """Build the camera image for the given configuration.

    ``source`` is ``'mono'`` (monochromatic) or ``'broad'`` (broad spectrum,
    which requires ``dlambda`` and uses ``profile``).

    Returns
    -------
    X, Y, I : ndarray
        Sensor coordinates and the intensity image.
    """
    X, Y = camera_grid(sensor_size, resolution, zoom)
    opd = eng.opd_map(X, Y, dL0, tilt_x, tilt_y)
    if source == "mono":
        I = eng.intensity_monochromatic(opd, wavelength, n=n, I0=I0)
    elif source == "broad":
        if dlambda is None:
            raise ValueError("Broad-spectrum source requires 'dlambda'.")
        I = eng.intensity_broadband(opd, wavelength, dlambda, n=n, I0=I0, profile=profile)
    else:
        raise ValueError(f"Unknown source: {source!r} (use 'mono' or 'broad')")
    return X, Y, I


def extract_perpendicular_profile(I_map, tilt_x, tilt_y, n_points=400):
    """Sample the intensity along a line perpendicular to the fringes.

    Fringes run perpendicular to the tilt gradient ``(tilt_x, tilt_y)``, so the
    profile is taken through the image center along that gradient direction
    using bilinear interpolation. When there is no tilt the horizontal axis is
    used (the field is uniform anyway).

    Returns
    -------
    positions : ndarray
        Signed distance from center in pixel units.
    profile : ndarray
        Sampled intensity values.
    """
    rows, cols = I_map.shape
    center_r = (rows - 1) / 2.0
    center_c = (cols - 1) / 2.0

    tilt = np.hypot(tilt_x, tilt_y)
    if tilt == 0.0:
        ux, uy = 1.0, 0.0
    else:
        ux, uy = tilt_x / tilt, tilt_y / tilt

    # The grid is square and symmetric, so the index-space direction matches
    # the physical direction. Span the largest centered line that stays inside.
    half = min(center_r, center_c)
    positions = np.linspace(-half, half, n_points)
    sample_cols = center_c + positions * ux
    sample_rows = center_r + positions * uy

    profile = map_coordinates(
        I_map, np.vstack([sample_rows, sample_cols]), order=1, mode="nearest"
    )
    return positions, profile


def visibility_from_profile(profile, center_fraction=1.0):
    """Fringe visibility ``V = (Imax - Imin) / (Imax + Imin)`` of a 1-D profile.

    ``center_fraction`` (0, 1] restricts the contrast measurement to the central
    portion of the profile. This matters with a tilted mirror: the coherence
    envelope varies across the field, so a narrow central window gives the local
    visibility (nearly constant envelope) instead of a field-averaged value.
    """
    profile = np.asarray(profile, dtype=float)
    if center_fraction < 1.0:
        n_keep = max(3, int(round(len(profile) * center_fraction)))
        start = (len(profile) - n_keep) // 2
        profile = profile[start : start + n_keep]
    Imax = float(profile.max())
    Imin = float(profile.min())
    denom = Imax + Imin
    if denom == 0.0:
        return 0.0
    return (Imax - Imin) / denom


def sweep_visibility(
    dL0_array,
    wavelength,
    dlambda,
    tilt_x,
    tilt_y,
    n=1.0,
    profile="rect",
    sensor_size=0.01,
    resolution=400,
    zoom=1.0,
    center_fraction=0.3,
):
    """Measured visibility versus baseline arm difference (the 'experiment').

    For each ``dL0`` the full 2-D broadband pattern is simulated, the
    perpendicular profile extracted, and its contrast measured over the central
    ``center_fraction`` of the profile. A nonzero tilt is required so the
    profile spans several fringes.

    Returns
    -------
    ndarray
        Visibility for each entry of ``dL0_array``.
    """
    if np.hypot(tilt_x, tilt_y) == 0.0:
        raise ValueError("sweep_visibility needs a nonzero tilt so fringes are present.")

    X, Y = camera_grid(sensor_size, resolution, zoom)
    visibilities = np.empty(len(dL0_array), dtype=float)
    for i, dL0 in enumerate(dL0_array):
        opd = eng.opd_map(X, Y, dL0, tilt_x, tilt_y)
        I = eng.intensity_broadband(opd, wavelength, dlambda, n=n, profile=profile)
        _, prof = extract_perpendicular_profile(I, tilt_x, tilt_y, n_points=resolution)
        visibilities[i] = visibility_from_profile(prof, center_fraction=center_fraction)
    return visibilities


def measure_coherence_length(
    dL0_array, visibility_array, n=1.0, method="first_zero", threshold=1.0 / np.e
):
    """Estimate the coherence length (round-trip OPD) from a visibility sweep.

    The arm-difference sweep is converted to round-trip OPD via ``OPD = 2 n dL``.

    Parameters
    ----------
    method : {'first_zero', 'threshold'}
        ``'first_zero'`` locates the first local minimum of the visibility
        envelope (natural for a rectangular spectrum, where fringes vanish).
        ``'threshold'`` finds the first OPD where the visibility drops to
        ``threshold`` (e.g. ``1/e`` or ``0.5`` for a Gaussian spectrum).

    Returns
    -------
    float
        Estimated coherence length as a round-trip OPD (meters), or ``nan`` if
        the criterion is never reached within the sweep.
    """
    d = np.abs(np.asarray(dL0_array, dtype=float))
    V = np.asarray(visibility_array, dtype=float)
    order = np.argsort(d)
    d, V = d[order], V[order]

    if method == "first_zero":
        # A genuine envelope null sits well below the central visibility; the
        # 0.5*V0 floor rejects small ripples near V ~ 1 from coarse sampling.
        floor = 0.5 * V[0] if len(V) else 0.0
        for i in range(1, len(V) - 1):
            if V[i] <= V[i - 1] and V[i] <= V[i + 1] and V[i] < floor:
                return 2.0 * n * d[i]
        return np.nan

    if method == "threshold":
        for i in range(len(V) - 1):
            if V[i] >= threshold > V[i + 1]:
                frac = (threshold - V[i]) / (V[i + 1] - V[i])
                d_cross = d[i] + frac * (d[i + 1] - d[i])
                return 2.0 * n * d_cross
        return np.nan

    raise ValueError(f"Unknown method: {method!r} (use 'first_zero' or 'threshold')")
