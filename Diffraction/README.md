# Fraunhofer Diffraction Simulator

Analytical 2D Fraunhofer diffraction simulator for Taller 4, exercise 20.
It supports a finite slit, a rectangular opening, a circular opening, and a
coherent collection containing independently sized and positioned primitives
of all three types.

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
- [`dashboard_tools.py`](dashboard_tools.py): wavelength-color conversion,
  optical schematic, aperture preview, diffraction camera, and scale bars.
- [`fraunhofer_simulator.ipynb`](fraunhofer_simulator.ipynb): interactive
  optical-workbench dashboard and analytical sanity checks.
- [`tests/test_diffraction_engine.py`](tests/test_diffraction_engine.py):
  analytical and validity regression tests.
- [`tests/test_dashboard_tools.py`](tests/test_dashboard_tools.py):
  dashboard color and rendering regression tests.
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

The gate runs before the observation grid is allocated or a field is
evaluated. Invalid input raises `FarFieldError`. The dashboard continues to
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

Use the button selector for **Slit**, **Rectangle**, **Circle**, or
**Multiple**. Slits expose width, finite length, and vertical/horizontal
orientation. Multiple mode can add or remove independently sized and
positioned slit, rectangle, and circle openings.

Global controls set vacuum wavelength, refractive index, and aperture-to-screen
distance. Display controls set screen half-width, zoom, and a display-only
**secondary maxima boost** that reveals weak rings/fringes without modifying
the physical intensity calculation. Advanced controls set sampling resolution
and the maximum accepted Fresnel number. With **Auto update** enabled, previews
refresh when a slider is released or a selector changes. **Refresh** forces an
update and **Reset** restores all configured defaults.

Optional **Profiles** and **Detailed geometry** buttons add analytical line
graphs and the expanded aperture-to-observation side view below the main
dashboard.

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

## Tests

From the repository root:

```powershell
python -m unittest discover -s Diffraction/tests -v
```

The suite checks analytical center limits and first zeros, phase translation,
double-slit interference, mixed-shape intensity, refractive-index scaling,
first-minimum helpers, rejection before screen-grid creation, wavelength-color
mapping, intensity RGB normalization, and dashboard rendering helpers.

## References

The formulas and coordinate conventions follow
[`theory and context/Fraunhofer_Transfomada_Fourier_espacial.pdf`](theory%20and%20context/Fraunhofer_Transfomada_Fourier_espacial.pdf),
especially the rectangular aperture/slit, circular aperture, translated
aperture, and far-field sections. The exercise statement is preserved in
[`theory and context/Screenshot 2026-07-19 121417.png`](theory%20and%20context/Screenshot%202026-07-19%20121417.png).
