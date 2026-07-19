"""Physics engine for a Fourier-transform infrared (FTIR) spectroscopy simulator.

This module is UI-agnostic: it only contains vectorized NumPy/SciPy functions
describing the FTIR measurement chain, so it can be unit-tested and reused from
notebooks, scripts or an app. It is a companion to ``michelson_engine.py`` (the
FTIR interferogram is produced by a Michelson interferometer with a moving
mirror).

Conventions and units
---------------------
- Spectra are expressed versus **wavenumber** ``sigma = 1 / lambda`` in cm^-1,
  the standard FTIR convention (mid-IR is roughly 400-4000 cm^-1).
- The optical path difference (OPD) ``delta = 2 n dL`` is in cm, where ``dL`` is
  the one-way mirror displacement and ``n`` the medium index (``n = 1`` for a
  standard instrument).
- A "single-beam" spectrum is the intensity reaching the detector per wavenumber:
  source envelope optionally multiplied by the sample transmittance.

Core relations (lecture slides g / VII-IX)
------------------------------------------
- Interferogram:  ``I(delta) = integral S(sigma) [1 + cos(2*pi*sigma*delta)] d sigma``.
- Centered part:  ``W(delta) = I(delta) - I_total``  (remove the DC offset).
- Recovery:       ``S(sigma) = F{W(delta)}``  (Fourier transform of the interferogram).
- Resolution:     ``d_sigma = 1 / (2 * delta_max)``; Nyquist ``sigma_max = 1 / (2 * d_delta)``.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

__all__ = [
    "GAS_LIBRARY",
    "default_wavenumber_grid",
    "source_spectrum",
    "gas_transmittance",
    "mixture_transmittance",
    "single_beam_spectrum",
    "make_opd_grid",
    "interferogram",
    "recover_spectrum",
    "transmittance_from_beams",
    "absorbance",
    "find_absorption_bands",
    "identify_gas",
    "spectral_resolution",
    "nyquist_sigma",
]

# Second radiation constant c2 = h c / k_B, in cm*K (Planck law in wavenumber).
_C2_CM_K = 1.438776877

# Synthetic mid-IR absorption library. Each gas maps to a list of bands
# (center [cm^-1], relative strength, Gaussian/Lorentzian half-width [cm^-1]).
# Band centers are representative literature positions of strong fundamentals.
GAS_LIBRARY = {
    "CO2": [(2349.0, 1.0, 8.0), (667.0, 0.6, 6.0)],
    "CO": [(2143.0, 0.9, 5.0)],
    "CH4": [(3019.0, 0.8, 10.0), (1306.0, 0.6, 8.0)],
    "H2O": [(3700.0, 0.7, 30.0), (1595.0, 0.5, 25.0)],
    "N2O": [(2224.0, 0.8, 6.0), (1285.0, 0.5, 7.0)],
    "SO2": [(1361.0, 0.7, 10.0), (1151.0, 0.5, 9.0)],
}


def default_wavenumber_grid(sigma_min=400.0, sigma_max=4000.0, num=2048):
    """Uniform mid-IR wavenumber grid (cm^-1)."""
    return np.linspace(sigma_min, sigma_max, num)


def source_spectrum(sigma, temperature=1200.0):
    """Broadband IR source envelope versus wavenumber (normalized to peak 1).

    Models a thermal emitter (e.g. a globar) with Planck's law expressed per
    unit wavenumber: ``B(sigma) ~ sigma^3 / (exp(c2 sigma / T) - 1)``.

    Parameters
    ----------
    sigma : ndarray
        Wavenumber grid (cm^-1).
    temperature : float
        Source temperature (K).
    """
    sigma = np.asarray(sigma, dtype=float)
    x = _C2_CM_K * sigma / temperature
    # Guard against overflow/zero division at the grid extremes.
    with np.errstate(over="ignore", invalid="ignore"):
        radiance = sigma**3 / np.expm1(x)
    radiance = np.nan_to_num(radiance, nan=0.0, posinf=0.0, neginf=0.0)
    peak = radiance.max()
    return radiance / peak if peak > 0 else radiance


def _band_profile(sigma, center, width, shape):
    """Unit-height band profile centered at ``center`` with half-width ``width``."""
    if shape == "gaussian":
        return np.exp(-0.5 * ((sigma - center) / width) ** 2)
    if shape == "lorentzian":
        return 1.0 / (1.0 + ((sigma - center) / width) ** 2)
    raise ValueError(f"Unknown band shape: {shape!r} (use 'gaussian' or 'lorentzian')")


def gas_transmittance(sigma, gas, concentration=1.0, shape="gaussian"):
    """Transmittance ``T(sigma) = exp(-A(sigma))`` of a single gas.

    The absorbance ``A`` is the sum of the gas's library bands scaled by
    ``concentration``.

    Parameters
    ----------
    sigma : ndarray
        Wavenumber grid (cm^-1).
    gas : str
        Key into :data:`GAS_LIBRARY`.
    concentration : float
        Multiplies the absorbance (path-length x concentration, arbitrary units).
    shape : {'gaussian', 'lorentzian'}
        Band line shape.
    """
    if gas not in GAS_LIBRARY:
        raise KeyError(f"Unknown gas {gas!r}; available: {sorted(GAS_LIBRARY)}")
    sigma = np.asarray(sigma, dtype=float)
    absorb = np.zeros_like(sigma)
    for center, strength, width in GAS_LIBRARY[gas]:
        absorb += strength * _band_profile(sigma, center, width, shape)
    return np.exp(-concentration * absorb)


def mixture_transmittance(sigma, components, shape="gaussian"):
    """Transmittance of a gas mixture: product of per-gas transmittances.

    Parameters
    ----------
    sigma : ndarray
        Wavenumber grid (cm^-1).
    components : dict
        Mapping ``gas -> concentration``.
    shape : {'gaussian', 'lorentzian'}
        Band line shape.
    """
    sigma = np.asarray(sigma, dtype=float)
    T = np.ones_like(sigma)
    for gas, conc in components.items():
        T *= gas_transmittance(sigma, gas, concentration=conc, shape=shape)
    return T


def single_beam_spectrum(sigma, source, transmittance=None):
    """Detector single-beam spectrum: ``source`` optionally x ``transmittance``."""
    source = np.asarray(source, dtype=float)
    if transmittance is None:
        return source
    return source * np.asarray(transmittance, dtype=float)


def make_opd_grid(sigma_max, d_sigma, n=1.0):
    """Symmetric OPD grid sized for a target resolution and Nyquist limit.

    The maximum OPD sets resolution ``delta_max = 1 / (2 d_sigma)``; the OPD step
    sets the Nyquist wavenumber ``sigma_max = 1 / (2 d_delta)``. A safety factor
    of 2 on the step keeps the highest band well below Nyquist.

    Returns
    -------
    ndarray
        OPD samples (cm), symmetric about zero, with an odd length so the grid
        contains ``delta = 0``.
    """
    delta_max = 1.0 / (2.0 * d_sigma)
    d_delta = 1.0 / (2.0 * (2.0 * sigma_max))
    n_side = int(np.ceil(delta_max / d_delta))
    offsets = np.arange(-n_side, n_side + 1)
    return offsets * d_delta


def interferogram(sigma, S, opd):
    """Interferogram ``I(delta) = integral S(sigma)[1 + cos(2 pi sigma delta)] d sigma``.

    Evaluated numerically over the provided wavenumber grid for every OPD value.

    Parameters
    ----------
    sigma : ndarray
        Wavenumber grid (cm^-1), assumed (approximately) uniform.
    S : ndarray
        Single-beam spectrum sampled on ``sigma``.
    opd : ndarray
        OPD samples (cm).

    Returns
    -------
    ndarray
        Interferogram intensity sampled on ``opd``.
    """
    sigma = np.asarray(sigma, dtype=float)
    S = np.asarray(S, dtype=float)
    opd = np.asarray(opd, dtype=float)
    dsig = float(np.mean(np.diff(sigma))) if sigma.size > 1 else 1.0
    # Outer product phase: rows over OPD, columns over wavenumber.
    phase = 2.0 * np.pi * np.outer(opd, sigma)
    integrand = S[None, :] * (1.0 + np.cos(phase))
    return np.trapezoid(integrand, dx=dsig, axis=1)


def recover_spectrum(opd, ifg):
    """Recover the single-beam spectrum from an interferogram by FFT.

    Subtracts the DC offset (``W = I - mean``) and takes the magnitude of the
    real FFT. The returned wavenumber axis uses the OPD sample step.

    Parameters
    ----------
    opd : ndarray
        Uniformly spaced OPD samples (cm).
    ifg : ndarray
        Interferogram values on ``opd``.

    Returns
    -------
    sigma_axis : ndarray
        Recovered wavenumber axis (cm^-1).
    magnitude : ndarray
        Recovered spectral magnitude (arbitrary units).
    """
    opd = np.asarray(opd, dtype=float)
    ifg = np.asarray(ifg, dtype=float)
    d_delta = float(np.mean(np.diff(opd)))
    W = ifg - ifg.mean()
    spectrum = np.fft.rfft(W)
    sigma_axis = np.fft.rfftfreq(W.size, d=d_delta)
    return sigma_axis, np.abs(spectrum)


def transmittance_from_beams(sample_sb, background_sb, eps=1e-12):
    """Transmittance ``T = sample / background`` (clipped to [0, 1])."""
    sample_sb = np.asarray(sample_sb, dtype=float)
    background_sb = np.asarray(background_sb, dtype=float)
    T = sample_sb / (background_sb + eps)
    return np.clip(T, 0.0, 1.0)


def absorbance(T, eps=1e-6):
    """Absorbance ``A = -log10(T)``."""
    T = np.asarray(T, dtype=float)
    return -np.log10(np.clip(T, eps, 1.0))


def find_absorption_bands(sigma, absorb, height=0.05, distance=5):
    """Locate absorption band centers as peaks of an absorbance spectrum.

    Returns
    -------
    ndarray
        Wavenumbers (cm^-1) of detected bands, strongest first.
    """
    sigma = np.asarray(sigma, dtype=float)
    absorb = np.asarray(absorb, dtype=float)
    peaks, props = find_peaks(absorb, height=height, distance=distance)
    if peaks.size == 0:
        return np.array([])
    order = np.argsort(props["peak_heights"])[::-1]
    return sigma[peaks][order]


def _reference_absorbance(sigma, gas, shape="gaussian"):
    """Library absorbance signature of a gas on the given grid."""
    sigma = np.asarray(sigma, dtype=float)
    absorb = np.zeros_like(sigma)
    for center, strength, width in GAS_LIBRARY[gas]:
        absorb += strength * _band_profile(sigma, center, width, shape)
    return absorb


def identify_gas(sigma, absorb, library=None, shape="gaussian"):
    """Rank library gases by similarity to a measured absorbance spectrum.

    Uses cosine similarity between the measured absorbance and each gas's
    synthetic absorbance signature on the same wavenumber grid.

    Parameters
    ----------
    sigma : ndarray
        Wavenumber grid (cm^-1) of ``absorb``.
    absorb : ndarray
        Measured/recovered absorbance spectrum.
    library : dict, optional
        Library to match against (defaults to :data:`GAS_LIBRARY`).
    shape : {'gaussian', 'lorentzian'}
        Line shape used to build the reference signatures.

    Returns
    -------
    list of (str, float)
        ``(gas, score)`` pairs sorted by descending score (score in [0, 1]).
    """
    if library is None:
        library = GAS_LIBRARY
    sigma = np.asarray(sigma, dtype=float)
    measured = np.asarray(absorb, dtype=float)
    m_norm = np.linalg.norm(measured)
    scores = []
    for gas in library:
        ref = _reference_absorbance(sigma, gas, shape=shape)
        denom = m_norm * np.linalg.norm(ref)
        score = float(np.dot(measured, ref) / denom) if denom > 0 else 0.0
        scores.append((gas, score))
    scores.sort(key=lambda item: item[1], reverse=True)
    return scores


def spectral_resolution(opd_max, n=1.0):
    """Spectral resolution (cm^-1) for a maximum OPD ``opd_max`` (cm)."""
    return 1.0 / (2.0 * opd_max) if opd_max > 0 else np.inf


def nyquist_sigma(d_opd, n=1.0):
    """Highest unaliased wavenumber (cm^-1) for OPD step ``d_opd`` (cm)."""
    return 1.0 / (2.0 * d_opd) if d_opd > 0 else np.inf
