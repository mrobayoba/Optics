# Fraunhofer and Fresnel Diffraction Simulators

Analytical and numerical 2D Fraunhofer diffraction simulator for Taller 4,
exercise 20, plus an analytical slit and straight-edge Fresnel simulator for
exercise 22.
It supports a finite slit, a rectangular opening, a circular opening, and a
coherent collection containing independently sized and positioned primitives
of all three types. Arbitrary custom openings can also be entered as a safe
mathematical expression, Python callable, or binary/grayscale image.

The interactive interfaces are
[`fraunhofer_simulator.ipynb`](fraunhofer_simulator.ipynb) and
[`fresnel_simulator.ipynb`](fresnel_simulator.ipynb), built with Jupyter,
`ipywidgets`, NumPy, SciPy, Pillow, and Matplotlib. Physics remains in plain
Python modules so it can be tested and reused without either notebook.

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
- [`fresnel_engine.py`](fresnel_engine.py): analytical Cornu/Fresnel-integral
  fields for slit and straight-edge diffraction.
- [`fresnel_screen_tools.py`](fresnel_screen_tools.py): Fresnel camera
  sampling, orientation, and shadow-side mapping.
- [`fresnel_config.py`](fresnel_config.py): Fresnel-specific control ranges and
  regime thresholds.
- [`fresnel_dashboard_tools.py`](fresnel_dashboard_tools.py): Fresnel optical
  schematic, aperture/edge preview, camera, and Cornu construction.
- [`fresnel_style.py`](fresnel_style.py): self-contained slit/edge dashboard.
- [`fresnel_simulator.ipynb`](fresnel_simulator.ipynb): source-document sanity
  checks and minimal Fresnel dashboard launcher.
- [`tests/test_diffraction_engine.py`](tests/test_diffraction_engine.py):
  analytical and validity regression tests.
- [`tests/test_dashboard_tools.py`](tests/test_dashboard_tools.py):
  dashboard color and rendering regression tests.
- [`tests/test_fourier_transform.py`](tests/test_fourier_transform.py):
  arbitrary-aperture, input-safety, image, FFT, and scaling tests.
- [`tests/test_fresnel_engine.py`](tests/test_fresnel_engine.py): Fresnel
  formulas, geometry, regimes, orientation, and shadow-side tests.
- [`tests/test_fresnel_dashboard.py`](tests/test_fresnel_dashboard.py):
  Fresnel visual, Cornu, and widget regressions.
- [`theory and context/`](theory%20and%20context/): the source PDF, exercise
  image, and condensed theory note.

## Fraunhofer physical model

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

## Fresnel physical model

The Fresnel notebook uses SciPy's normalized integrals:

```text
C(u) = integral(cos(pi*t^2/2), 0, u)
S(u) = integral(sin(pi*t^2/2), 0, u)
```

For a centered slit, the field is the Cornu chord between its dimensionless
edge coordinates:

```text
field = [(C(u2)-C(u1)) + i*(S(u2)-S(u1))] / sqrt(2)
I/I0 = |field|^2
```

For an edge, positive `u` is defined as the geometrical shadow:

```text
field = [(1/2-C(u)) + i*(1/2-S(u))] / sqrt(2)
```

This gives `I/I0 = 1` deep in the illuminated region, `1/4` at the
geometrical boundary, and zero deep in the shadow.

Plane-wave illumination uses `u = x*sqrt(2/(lambda_medium*D))`. Point-source
illumination adds source distance `d`, geometrical magnification
`(d+D)/d`, and effective distance

```text
D_effective = d*D/(d+D)
```

For a slit of full width `b`, the reported value is
`N_F = b^2/(4*lambda_medium*D_effective)`. It is informational in the Fresnel
notebook. The Advanced panel exposes a logarithmic selected condition
`N_F >= N_F,min` across `0.0001`–`10`, defaulting to `0.1`. This adjustable
criterion controls only the PASS/NOT MET teaching indicator. Conventional
labels remain fixed: values below `0.1` identify the Fraunhofer limit, values
from `0.1` to `1` are the Fresnel transition, and values at least `1` are
developed Fresnel near field. The criterion never blocks the more general
Fresnel solution. A straight edge has no finite support radius, so the control
is disabled and its status reports the characteristic screen length.

## Mandatory Fraunhofer far-field check

The source states that observation distance must be much greater than the
aperture-dependent quadratic phase scale. Because `>>` is not a numerical
test, this simulator enforces the explicit Fresnel-number criterion

```text
N_F = R_max^2 / (lambda_medium * z) <= N_F,max
```

where `R_max` is the largest radius from the optical axis to any point in any
configured opening. The default `N_F,max` is `0.1` and can be made stricter in
the notebook's Advanced controls.

The Fraunhofer gate runs before the observation grid or Fourier-transform
array is allocated. A custom input mask must first be sampled to determine its
support radius. Invalid propagation input raises `FarFieldError`. The
Fraunhofer dashboard continues to show the selected aperture and optical path,
but replaces the diffraction camera with a warning that reports

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

Open either notebook, select the repository's Python kernel, and run all
cells. Both dashboards keep three views together:

- a live source → aperture → observation-plane schematic;
- a dark aperture-plane preview with all selected openings;
- a wavelength-colored diffraction camera preview with a physical scale bar.

In the Fraunhofer notebook, use the selector for **Slit**, **Rectangle**,
**Circle**, **Multiple**,
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

In the Fresnel notebook, select **Slit** or **Edge**, then choose plane-wave or
finite point-source illumination. A slit exposes width and orientation. An
edge exposes orientation and which signed side is shadowed. **Profiles** plots
physical `I/I0`; **Cornu spiral** exposes a screen-coordinate probe and draws
the chord whose squared length produces that intensity. Fresnel number is
shown as regime information and does not disable the calculation.

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

The Fresnel API supports the same vacuum-wavelength and refractive-index
conventions:

```python
import fresnel_screen_tools as fresnel

result = fresnel.simulate_pattern(
    "slit",
    wavelength_vacuum=633e-9,
    distance=0.5,
    slit_width=1e-3,
    illumination="point",
    source_distance=1.0,
    orientation="vertical",
    screen_half_width=5e-3,
    resolution=401,
)
print(result.report.fresnel_number)
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
dashboard rendering and custom-control behavior. Fresnel regressions also
cover the PDF's slit/edge values, edge limits, Cornu chord intensity,
point-source geometry, sinc-squared convergence, orientation, shadow side,
regime reporting, and dashboard controls.

## References

The formulas and coordinate conventions follow
[`theory and context/Fraunhofer_Transfomada_Fourier_espacial.pdf`](theory%20and%20context/Fraunhofer_Transfomada_Fourier_espacial.pdf),
especially the rectangular aperture/slit, circular aperture, translated
aperture, and far-field sections. The exercise statement is preserved in
[`theory and context/Screenshot 2026-07-19 121417.png`](theory%20and%20context/Screenshot%202026-07-19%20121417.png).

The Fresnel formulas, source geometry, and Cornu construction follow
[`theory and context/Difraccion_Fresnel_Clotoide.pdf`](theory%20and%20context/Difraccion_Fresnel_Clotoide.pdf).
`scipy.special.fresnel` uses the same normalized `S(u), C(u)` convention and
returns them in that order.
