# Optics

Python modules for physical optics calculations, featuring two interactive
simulators built on the Michelson interferometer:

1. A **Michelson interferometer simulator** - reproduces the interference
   pattern seen by a camera with a collimated (plane-wave) beam and a mirror
   tilted in two degrees of freedom (azimuthal and zenithal), for monochromatic
   and broad-spectrum sources. Includes zoom, an intensity profile perpendicular
   to the fringes, and a tool to measure the source coherence length and compare
   it with theory (Taller 3, point 8).
2. An **FTIR spectroscopy simulator** - illustrates Fourier-transform infrared
   spectroscopy: it builds the interferogram of an IR source through a gas
   sample, recovers the spectrum by Fourier transform, and identifies the gas
   from its absorption-band fingerprint.

## Features

### Michelson simulator

- Monochromatic and broad-spectrum (rectangular or Gaussian) sources.
- Mirror tilt in two degrees of freedom producing straight (Fizeau) fringes.
- Live, interactive controls via `ipywidgets`: wavelength, spectral width, arm
  difference, both tilt axes, and zoom.
- Camera view plus an intensity profile taken perpendicular to the fringes.
- Coherence-length experiment: sweep the arm difference, measure fringe
  contrast, and contrast the result against the theoretical value.

### FTIR simulator

- Built-in synthetic mid-IR gas library (CO2, CO, CH4, H2O, N2O, SO2) with
  characteristic absorption bands.
- Thermal IR source envelope, gas/mixture transmittance, and the interferogram
  recorded by a moving-mirror interferometer.
- Fourier-transform recovery of the spectrum, transmittance and absorbance.
- Qualitative identification by matching the recovered fingerprint against the
  library (cosine similarity ranking).

## Project structure

| File | Description |
| --- | --- |
| [`michelson_engine.py`](michelson_engine.py) | Pure, vectorized interference physics (OPD, intensity, visibility, coherence length, fringe spacing). |
| [`camera_tools.py`](camera_tools.py) | Sensor grid + zoom, full-pattern wrapper, perpendicular profile, contrast measurement, coherence sweep. |
| [`michelson_simulator.ipynb`](michelson_simulator.ipynb) | Interactive Michelson notebook UI and the coherence-length experiment. |
| [`ftir_engine.py`](ftir_engine.py) | FTIR physics: gas library, IR source, transmittance, interferogram, FFT recovery, identification. |
| [`ftir_simulator.ipynb`](ftir_simulator.ipynb) | Interactive FTIR notebook: interferogram, spectral recovery, and gas identification. |
| [`PLAN.md`](PLAN.md) | Design document (English). |
| [`Original_Plan.md`](Original_Plan.md) | Original Spanish design note, kept for reference. |
| [`Michelson/`](Michelson/) | Lecture slides used to ground the physics. |
| [`requirements.txt`](requirements.txt) | Pinned dependencies. |

## Setup (Windows / PowerShell)

```powershell
# From the repo root
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # if blocked: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
python -m pip install --upgrade pip
pip install -r requirements.txt

# Register a Jupyter kernel bound to this environment
python -m ipykernel install --user --name optics-michelson --display-name "Python (Optics Michelson)"
```

On macOS / Linux, activate with `source .venv/bin/activate` instead.

## Usage

```powershell
.\.venv\Scripts\Activate.ps1
jupyter lab
```

Open either [`michelson_simulator.ipynb`](michelson_simulator.ipynb) or
[`ftir_simulator.ipynb`](ftir_simulator.ipynb) and select the
**Python (Optics Michelson)** kernel.

The Michelson notebook contains:

1. Setup and environment check.
2. Sanity checks (simulated fringe spacing and coherence length vs theory).
3. The interactive simulator (camera view + perpendicular profile).
4. The coherence-length experiment (measured vs theoretical, with percent error).

The FTIR notebook contains:

1. Setup and environment check.
2. Discrete-line sanity check (recovers the lecture worked example by FFT).
3. The interactive simulator (source/sample spectra, interferogram, recovered
   transmittance and absorbance).
4. Qualitative identification of an unknown gas or mixture against the library.

The physics modules can also be used directly:

```python
import numpy as np
import michelson_engine as eng
import camera_tools as cam

lam, dlam, tilt = 633e-9, 10e-9, 0.5e-3  # 633 nm, 10 nm bandwidth, 0.5 mrad tilt

# Simulate a broad-spectrum camera frame
X, Y, I = cam.simulate_pattern(lam, 0.0, tilt, 0.0, source="broad", dlambda=dlam)

# Theoretical coherence length (round-trip OPD)
print(eng.coherence_length_theory(lam, dlam, "rect"))  # lambda^2 / dlambda
```

```python
import ftir_engine as ftir

sigma = ftir.default_wavenumber_grid()                 # mid-IR grid (cm^-1)
source = ftir.source_spectrum(sigma)
sample = ftir.single_beam_spectrum(sigma, source, ftir.gas_transmittance(sigma, "CO2"))

opd = ftir.make_opd_grid(sigma.max(), ftir.spectral_resolution(0.5))
sig_rec, mag = ftir.recover_spectrum(opd, ftir.interferogram(sigma, sample, opd))
print(ftir.identify_gas(sigma, ftir.absorbance(sample / source))[0])  # ('CO2', ~1.0)
```

## Physics summary

### Michelson

Conventions: wavelengths in meters; `opd` is the one-way arm-length difference
`ΔL`; the round-trip optical path difference is `2 n · ΔL`; tilts are small
angles in radians.

- Monochromatic: `I = 2 I0 [1 + cos(2 n k0 · ΔL)]`, `k0 = 2π/λ`.
- Tilted mirror wedge: `ΔL(x, y) = ΔL0 + tilt_x·x + tilt_y·y`.
- Broad spectrum: `I = 2 I0 [1 + V(ΔL) · cos(2 n k0 · ΔL)]`, with rectangular-band
  visibility `V(ΔL) = |sinc(n · Δk · ΔL)|`, `Δk = 2π·Δλ/λ²`.
- Coherence length (round-trip OPD): rectangular `Lc = λ²/Δλ`; Gaussian
  `Lc = (2 ln 2 / π) · λ²/Δλ`. Fringe contrast `V = (Imax − Imin)/(Imax + Imin)`.
- Fringe spacing: `Δs = λ / (2 n |tilt|)`.

### FTIR

Conventions: spectra versus wavenumber `σ = 1/λ` in cm⁻¹ (mid-IR ~400-4000);
OPD `δ = 2 n · ΔL` in cm; standard instrument uses `n = 1`.

- Interferogram: `I(δ) = ∫ S(σ) [1 + cos(2π σ δ)] dσ`.
- Centered part: `W(δ) = I(δ) − I_total`; recovery `S(σ) = F{W(δ)}`.
- Transmittance `T = sample / background`; absorbance `A = −log10(T)`.
- Resolution `Δσ = 1/(2 δmax)`; Nyquist `σmax = 1/(2 Δδ)`.

## Requirements

Python 3.10+ with numpy, scipy, matplotlib, ipywidgets, jupyterlab, notebook,
and ipykernel (see [`requirements.txt`](requirements.txt)).
