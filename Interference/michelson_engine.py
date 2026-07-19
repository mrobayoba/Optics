"""Physics engine for a Michelson interferometer simulator.

This module is intentionally UI-agnostic: it only contains vectorized NumPy
functions describing the interference physics, so it can be unit-tested and
reused from notebooks, scripts or a web app.

Conventions and units (SI)
--------------------------
- ``wavelength`` (lambda) and ``dlambda`` (spectral width) are in meters.
- ``opd`` is the ONE-WAY arm-length difference Delta L of the interferometer
  (the quantity controlled by the movable mirror), in meters. The physical
  round-trip optical path difference is ``OPD = 2 * n * opd``.
- ``n`` is the refractive index of the medium between beam splitter and mirror.
- Angles ``tilt_x`` / ``tilt_y`` are mirror tilts in radians (small-angle wedge).

Core relations (from the lecture slides, cases b-f)
---------------------------------------------------
- Monochromatic:  I = 2 I0 [1 + cos(2 n k0 * dL)],   k0 = 2*pi/lambda.
- Tilted mirror:  dL(x, y) = dL0 + tilt_x*x + tilt_y*y  (air/medium wedge).
- Broad spectrum (rectangular band of width dk = 2*pi*dlambda/lambda^2):
      I = 2 I0 [1 + V(dL) * cos(2 n k0 * dL)],   V(dL) = |sinc(n * dk * dL)|.
- Coherence length (round-trip OPD): rectangular spectrum first zero at
  Lc = lambda^2 / dlambda; Gaussian (FWHM dlambda, 50% visibility) at
  Lc = (2 ln 2 / pi) * lambda^2 / dlambda.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "vacuum_wavenumber",
    "opd_map",
    "intensity_monochromatic",
    "visibility_rect",
    "visibility_gaussian",
    "intensity_broadband",
    "coherence_length_theory",
    "fringe_spacing",
]


def vacuum_wavenumber(wavelength):
    """Return the vacuum wavenumber k0 = 2*pi / lambda."""
    return 2.0 * np.pi / wavelength


def opd_map(X, Y, dL0, tilt_x, tilt_y):
    """One-way arm-difference field across the sensor.

    A tilted mirror introduces a linear wedge so the local arm difference is
    ``dL(x, y) = dL0 + tilt_x * x + tilt_y * y``.

    Parameters
    ----------
    X, Y : ndarray
        Sensor coordinates (meters), e.g. from :func:`camera_tools.camera_grid`.
    dL0 : float
        Baseline one-way arm difference at the optical axis (meters).
    tilt_x, tilt_y : float
        Mirror tilts (radians) about the two axes (azimuthal / zenithal).
    """
    return dL0 + tilt_x * X + tilt_y * Y


def intensity_monochromatic(opd, wavelength, n=1.0, I0=1.0):
    """Intensity for a monochromatic source (slide cases c and d).

    ``I = 2 I0 [1 + cos(2 n k0 * opd)]`` where ``opd`` is the one-way arm
    difference field. With ``tilt = 0`` this is a uniform field (case c); with
    a nonzero tilt it produces straight (Fizeau) fringes (case d).
    """
    k0 = vacuum_wavenumber(wavelength)
    phase = 2.0 * n * k0 * opd
    return 2.0 * I0 * (1.0 + np.cos(phase))


def visibility_rect(opd, wavelength, dlambda, n=1.0):
    """Fringe visibility for a rectangular (top-hat) spectrum.

    ``V(dL) = |sinc(n * dk * dL)|`` with ``dk = 2*pi*dlambda/lambda^2`` and
    the unnormalized sinc ``sin(z)/z``. First zero at ``dL = lambda^2/(2 n dlambda)``
    (round-trip OPD ``lambda^2/dlambda``).
    """
    dk = 2.0 * np.pi * dlambda / wavelength**2
    z = n * dk * np.asarray(opd, dtype=float)
    # np.sinc(y) = sin(pi*y)/(pi*y); we want sin(z)/z, so pass y = z/pi.
    return np.abs(np.sinc(z / np.pi))


def visibility_gaussian(opd, wavelength, dlambda, n=1.0):
    """Fringe visibility for a Gaussian spectrum of FWHM ``dlambda``.

    The visibility is the Fourier transform of the spectral line and equals
    1/2 at the round-trip OPD ``Lc = (2 ln 2 / pi) lambda^2 / dlambda``.
    """
    dsigma = dlambda / wavelength**2  # FWHM in wavenumber sigma = 1/lambda
    round_trip = 2.0 * n * np.asarray(opd, dtype=float)
    return np.exp(-(np.pi * round_trip * dsigma) ** 2 / (4.0 * np.log(2.0)))


def intensity_broadband(opd, wavelength, dlambda, n=1.0, I0=1.0, profile="rect"):
    """Intensity for a broad-spectrum source (slide case f).

    ``I = 2 I0 [1 + V(opd) * cos(2 n k0 * opd)]`` where the coherence envelope
    ``V`` is selected by ``profile`` ('rect' or 'gaussian').
    """
    k0 = vacuum_wavenumber(wavelength)
    phase = 2.0 * n * k0 * opd
    if profile == "rect":
        V = visibility_rect(opd, wavelength, dlambda, n=n)
    elif profile == "gaussian":
        V = visibility_gaussian(opd, wavelength, dlambda, n=n)
    else:
        raise ValueError(f"Unknown spectral profile: {profile!r} (use 'rect' or 'gaussian')")
    return 2.0 * I0 * (1.0 + V * np.cos(phase))


def coherence_length_theory(wavelength, dlambda, profile="rect", n=1.0):
    """Theoretical coherence length as a round-trip OPD (meters).

    - ``rect``: first zero of the sinc envelope, ``Lc = lambda^2 / dlambda``.
    - ``gaussian``: 50% visibility, ``Lc = (2 ln 2 / pi) lambda^2 / dlambda``.

    The round-trip OPD coherence length is independent of ``n`` (it cancels);
    the argument is kept for API symmetry. The corresponding arm-difference
    half-width is ``Lc / (2 n)``.
    """
    if profile == "rect":
        return wavelength**2 / dlambda
    if profile == "gaussian":
        return (2.0 * np.log(2.0) / np.pi) * wavelength**2 / dlambda
    raise ValueError(f"Unknown spectral profile: {profile!r} (use 'rect' or 'gaussian')")


def fringe_spacing(wavelength, tilt_x, tilt_y, n=1.0):
    """Spacing between adjacent fringes for a tilted mirror (meters).

    ``ds = lambda / (2 n |tilt|)`` with ``|tilt| = sqrt(tilt_x^2 + tilt_y^2)``.
    Returns ``inf`` when there is no tilt (uniform field, no fringes).
    """
    tilt = np.hypot(tilt_x, tilt_y)
    if tilt == 0.0:
        return np.inf
    return wavelength / (2.0 * n * tilt)
