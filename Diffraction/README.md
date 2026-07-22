# Fraunhofer Diffraction Simulator

Analytical and numerical 2D Fraunhofer diffraction simulator for Taller 4,
exercise 20.
It supports a finite slit, a rectangular opening, a circular opening, and a
coherent collection containing independently sized and positioned primitives
of all three types. Arbitrary custom openings can also be entered as a safe
mathematical expression, Python callable, or binary/grayscale image.

The interactive interface is
[`fraunhofer_simulator.ipynb`](fraunhofer_simulator.ipynb), built with
Jupyter, `ipywidgets`, NumPy, SciPy, and Matplotlib. The physics remains in
plain Python modules so it can be tested and reused without the notebook.

## Project structure

- [`diffraction_config.py`](diffraction_config.py): widget ranges, defaults,
  unit scales, plot settings, and the default far-field threshold.
- [`diffraction_engine.py`](diffraction_engine.py): aperture definitions,
  analytical Fourier amplitudes, coherent composition, first minima, and the
  mandatory far-field gate.
- [`screen_tools.py`](screen_tools.py): observation grid, guarded 2D
  simulation, and central intensity profiles.
- [`fourier_transform.py`](fourier_transform.py): arbitrary-aperture
  rasterization, restricted expression interpreter, image decoding, centered
  FFT, and observation-plane interpolation.
- [`dashboard_tools.py`](dashboard_tools.py): wavelength-color conversion,
  optical schematic, aperture preview, diffraction camera, and scale bars.
- [`diffraction_style.py`](diffraction_style.py): self-contained ipywidgets
  dashboard, controls, callbacks, layout, and optional analytical views.
- [`fraunhofer_simulator.ipynb`](fraunhofer_simulator.ipynb): interactive
  analytical sanity checks plus a minimal dashboard import/launch cell.
- [`tests/test_diffraction_engine.py`](tests/test_diffraction_engine.py):
  analytical and validity regression tests.
- [`tests/test_dashboard_tools.py`](tests/test_dashboard_tools.py):
  dashboard color and rendering regression tests.
- [`tests/test_fourier_transform.py`](tests/test_fourier_transform.py):
  arbitrary-aperture, input-safety, image, FFT, and scaling tests.
- [`theory and context/`](theory%20and%20context/): the source PDF, exercise
  image, and condensed theory note.

## Physical model

All internal lengths use SI units. The UI accepts the vacuum wavelength
`lambda_0` and refractive index `n`; propagation uses

```text
lambda_medium = lambda_0 / n
fx = x' / (lambda_medium * z)
fy = y' / (lambda_medium * z)
```

NumPy's normalized `sinc(u) = sin(pi*u)/(pi*u)` matches the source PDF.

For a rectangle of width `a`, height `b`, and center `(x0, y0)`, the
analytical Fraunhofer amplitude is

```text
A = a*b*sinc(a*fx)*sinc(b*fy)*exp[-i*2*pi*(fx*x0 + fy*y0)]
```

The slit case uses this same expression with a narrow width and finite length,
as in the PDF's 2D slit example. A finite length is required to define the
vertical pattern and the aperture's full far-field support.

For a circular opening of radius `r`,

```text
q = 2*pi*r*sqrt(fx^2 + fy^2)
A = pi*r^2 * 2*J1(q)/q
```

with the analytical limit `2*J1(q)/q = 1` at `q = 0`.

For multiple openings, the engine coherently adds the complex amplitudes and
then computes `I = |sum(A_j)|^2`. The normalized result has on-axis intensity
one. Components are intended to represent distinct, non-overlapping openings
under uniform coherent plane-wave illumination; overlapping entries would
model additive transmittance rather than the union of two cutouts.

For a sampled custom transmittance `t(x_tilde, y_tilde)`, the numerical path
calculates

```text
A(fx, fy) = FFT2{t} * dx * dy
I = |A|^2
x' = lambda_medium * z * fx
y' = lambda_medium * z * fy
```

The input mask is centered, symmetrically zero-padded, transformed with
`scipy.fft`, and linearly interpolated onto the same camera grid used by the
analytical cases. Normalized intensity is divided by the squared effective
open area, so the on-axis sample is one.

## Mandatory far-field check

The source states that observation distance must be much greater than the
aperture-dependent quadratic phase scale. Because `>>` is not a numerical
test, this simulator enforces the explicit Fresnel-number criterion

```text
N_F = R_max^2 / (lambda_medium * z) <= N_F,max
```

where `R_max` is the largest radius from the optical axis to any point in any
configured opening. The default `N_F,max` is `0.1` and can be made stricter in
the notebook's Advanced controls.

The gate runs before the observation grid or Fourier-transform array is
allocated. A custom input mask must first be sampled to determine its support
radius. Invalid propagation input raises `FarFieldError`. The dashboard continues to
show the selected aperture and optical path, but replaces the diffraction
camera with a warning that reports

```text
z_required = R_max^2 / (lambda_medium * N_F,max)
```

The threshold is an explicit engineering tolerance, not an absolute boundary
between Fresnel and Fraunhofer diffraction.

## Setup and use

From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter lab
```

Open `Diffraction/fraunhofer_simulator.ipynb`, select the repository's Python
kernel, and run all cells. The friendly dashboard then keeps three views
together:

- a live source → aperture → observation-plane schematic;
- a dark aperture-plane preview with all selected openings;
- a wavelength-colored Fraunhofer camera preview with a physical scale bar.

Use the button selector for **Slit**, **Rectangle**, **Circle**, **Multiple**,
or **Custom**. Slits expose width, finite length, and vertical/horizontal
orientation. Multiple mode can add or remove independently sized and
positioned slit, rectangle, and circle openings. Custom mode accepts either a
restricted expression or an uploaded image plus its calibrated physical width
and height, mask resolution, and FFT padding. The shared **Invert white /
black** toggle complements the selected expression or image mask.

Global controls set vacuum wavelength with `0.01 nm` precision, refractive
index, and aperture-to-screen distance. The source beam and Fraunhofer camera
use the display color corresponding to the selected wavelength. Display
controls set screen half-width, zoom, and a display-only
**secondary maxima boost** that reveals weak rings/fringes without modifying
the physical intensity calculation. Advanced controls set sampling resolution
and the maximum accepted Fresnel number. With **Auto update** enabled, previews
refresh when a slider is released or a selector changes. **Refresh** forces an
update and **Reset** restores all configured defaults.

The optional **Profiles** button adds analytical horizontal and vertical line
graphs below the main dashboard.

## Python API example

Run from the `Diffraction` directory:

```python
import diffraction_engine as engine
import screen_tools as screen

apertures = [
    engine.Aperture.slit(50e-6, 300e-6, center_x=-150e-6),
    engine.Aperture.circle(75e-6, center_x=150e-6),
]

result = screen.simulate_pattern(
    apertures,
    wavelength_vacuum=633e-9,
    distance=5.0,
    n=1.0,
    screen_half_width=0.04,
    resolution=401,
)
print(result.far_field)
```

Arbitrary callables receive aperture-plane `X, Y` arrays in metres:

```python
import numpy as np
import fourier_transform as fourier

aperture = fourier.CustomAperture.from_callable(
    lambda X, Y: (
        (np.hypot(X, Y) <= 0.18e-3)
        & ~((X > 0.04e-3) & (np.abs(Y) < 0.03e-3))
    ),
    width=0.5e-3,
    height=0.5e-3,
    resolution=256,
)
result = fourier.simulate_pattern(
    aperture,
    wavelength_vacuum=633e-9,
    distance=5.0,
    screen_half_width=0.04,
    resolution=401,
    padding_factor=4,
)
```

Dashboard expressions use `x`, `y`, and `r` in millimetres. For example:

```python
aperture = fourier.CustomAperture.from_expression(
    "(r <= 0.18) & ~((x > 0.04) & (abs(y) < 0.03))",
    width=0.5e-3,
    height=0.5e-3,
    resolution=256,
    invert=False,
)
```

Expressions are interpreted by a strict AST walker; Python evaluation is not
used. Attributes, imports, subscripts, comprehensions, assignments, lambdas,
unknown names/functions, long inputs, and large exponents are rejected.
Approved vectorized functions are listed directly below the dashboard editor.

The expression and image modes both expose **Invert white / black**, which
replaces transmittance `t` with `1 - t`. Images are decoded with Pillow,
converted to grayscale, resized with nearest neighbour sampling, and
thresholded with white=open by default. The selected physical width and height
calibrate the image pixels in the aperture plane; image rows are flipped so
the displayed top corresponds to positive `y`.

Numerical accuracy depends on both mask sampling and FFT sampling. Increase
mask resolution for fine aperture edges and increase padding for denser
frequency samples. The mask must contain the complete physical opening, and
camera coordinates beyond the FFT Nyquist range are filled with zero.

## Tests

From the repository root:

```powershell
python -m unittest discover -s Diffraction/tests -v
```

The suite checks analytical center limits and first zeros, phase translation,
double-slit interference, mixed-shape intensity, refractive-index scaling,
first-minimum helpers, rejection before allocation, custom callable,
expression and image construction, expression safety, FFT normalization and
padding, numerical rectangle/Airy minima, wavelength-color mapping, and
dashboard rendering and custom-control behavior.

## References

The formulas and coordinate conventions follow
[`theory and context/Fraunhofer_Transfomada_Fourier_espacial.pdf`](theory%20and%20context/Fraunhofer_Transfomada_Fourier_espacial.pdf),
especially the rectangular aperture/slit, circular aperture, translated
aperture, and far-field sections. The exercise statement is preserved in
[`theory and context/Screenshot 2026-07-19 121417.png`](theory%20and%20context/Screenshot%202026-07-19%20121417.png).
