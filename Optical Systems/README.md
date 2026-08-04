# Paraxial Optical Systems

Python port of the course MATLAB matrix-optics scripts, extended with
principal-plane reduction, conjugate-plane imaging, ordered matrix resolution,
ray sampling, and an interactive Jupyter system builder.

The interface is [`paraxial_simulator.ipynb`](paraxial_simulator.ipynb).
Physics remains in plain Python modules so it can be tested and reused without
the notebook.

## Project structure

- [`paraxial_engine.py`](paraxial_engine.py): elemental matrices, ordered
  cascade, vertex-to-vertex reports, principal planes, and conjugate imaging.
- [`paraxial_config.py`](paraxial_config.py): widget ranges, display settings,
  and system presets.
- [`paraxial_tools.py`](paraxial_tools.py): logical components, one matrix per
  stack entry, system construction, geometry, and ray-fan sampling.
- [`paraxial_dashboard_tools.py`](paraxial_dashboard_tools.py): matrix HTML and
  Matplotlib optical schematics.
- [`paraxial_style.py`](paraxial_style.py): interactive element stack and live
  matrix/principal/conjugate panels.
- [`tests/test_paraxial_engine.py`](tests/test_paraxial_engine.py): MATLAB
  exercise, ordering, cardinal-plane, conjugate-plane, and ray tests.
- [`context.md`](context.md): supporting course theory and sign conventions.

## Course matrix convention

All lengths are in metres and powers are in inverse metres. The implementation
deliberately preserves the source matrices:

```text
ray = (n*alpha, x)

T(n,d) = [[1,   0],       Ra(P) = [[1, -P],
          [d/n, 1]]                [0,  1]]

Re(n,R) = [[1, 2*n/R],
           [0,     1]]
```

Factors are written left-to-right, but the **first optical component is
rightmost** so it multiplies the input ray first. Later components pile on
the left:

```text
M_VV' = M_n @ ... @ M_2 @ M_1
```

They are not commutative. For example, adding a translation then a thin lens
displays `Mtn @ T`, not `T @ Mtn`. The dashboard shows every component's
label, numeric 2×2 matrix, multiplication sign, and resolved product.

Thin and thick lenses each contribute **one** matrix to that product. Their
factors are listed in optical order `(Ra(P1), T, Ra(P2))`, then reversed
once before left-to-right multiply (`cascade` never reverses on its own):

```text
M_thick = Ra(P2) @ T(n_lens, thickness) @ Ra(P1)
M_thin  = Ra(P2) @ Ra(P1)
```

so a thick-lens component produces the same `M_VV'` as adding `Ra`, `T`,
`Ra` separately. The row-state representation `[x, n*alpha] @ M` is used
only for plotting ray fans; it matches the column action of these matrices.

## Analysis chain

### 1. Base system

The user-created elemental stack defines the base matrix only between its
start and finish vertices:

```text
M_VV' = product of stack matrices with first component rightmost
        (T, Ra, Re, M_thin, or M_thick)
det(M_VV') = 1
```

Object/image translations are not included in this base matrix.

### 2. Principal planes

For

```text
M_VV' = [[M11, M12],
         [M21, M22]]
```

the system power and principal offsets are

```text
P  = -M12
D  = n  * (1 - M11) / M12       (H to V)
D' = n' * (1 - M22) / M12       (V' to H')
f  = n/P
f' = n'/P
```

The dashboard explicitly resolves

```text
T(V'→H') @ M_VV' @ T(H→V) = M_HH'
M_HH' = [[1, -P],
         [0,  1]]
```

An afocal system (`M12 = 0`) has no finite version of this reduction and is
reported as such rather than divided by zero.

### 3. Conjugate planes

Lab measurements usually give Object→V (`x`) or V′→image (`x′`), not the
principal-plane distances. With `D` and `D′` from `M_VV′`:

```text
s  = x  - D
s' = x' - D'
```

The dashboard accepts either known distance and solves the other side from

```text
n'/s' + n/s = P
```

where `P = -M12` comes from the principal-plane reduction. The displayed
transfer is still

```text
M_oi = T(H'→image) @ M_HH' @ T(object→H)
```

For conjugate planes, `M21 = 0`. The report shows:

```text
m_x     = M22 = -n*s'/(n'*s)
m_alpha = M11*n/n' = -s/s'
m_x * m_alpha * n'/n = 1
P_check = n'/s' + n/s   (should match P)
```

The sign of `s'` classifies real/virtual images, the sign of `m_x` classifies
erect/inverted images, and `abs(m_x)` classifies their relative size.

If the object is at the front focal plane, its conjugate is at infinity and no
finite object-to-image matrix is displayed.

## Reflection signs

The mirror matrix follows the course source:

```text
Re(n,R) = [[1, 2*n/R],
           [0,     1]]
```

Its equivalent power is `P = -2*n/R`, so the conjugate equation reduces to
`1/s_o + 1/s_i = -2/R` in one medium. Positive path distance follows the
direction of propagation. After a reflection, plots use an **unfolded optical
path coordinate**; this keeps cascaded matrix order readable without implying
that the reflected ray continues to the right in laboratory coordinates.

With the source sign convention, a real image has positive image distance
along the outgoing path; a virtual image has negative distance. Radius signs
must be entered consistently with the course notes in [`context.md`](context.md).

## Use

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jupyter lab
```

Open `Optical Systems/paraxial_simulator.ipynb`, run all cells, then:

1. load a preset or add translation, refraction, reflection, thin-lens, or
   thick-lens components;
2. reorder components with the arrow buttons;
3. inspect the elemental product and `M_VV'`;
4. inspect the principal-plane reduction and values;
5. choose Object→V (`x`) or V′→image (`x′`) and inspect conjugates, magnifications,
   and the zoomable schematic.

Plain Python example:

```python
import paraxial_engine as engine
import paraxial_tools as tools

elements = [
    tools.OpticalElement.translation("T1", 1.0, 0.2),
    tools.OpticalElement.thin_lens("L1", 5.0, 5.0),
]
base = tools.build_system(elements)
cardinal = engine.principal_planes(base.matrix)
image = engine.conjugate_from_vertex_distance(
    cardinal,
    mode="object_to_v",
    distance=0.2,
)
```

## Tests

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s "Optical Systems\tests" -v
```

The source three-lens exercise is checked against:

```text
M_t ≈ [[0.5616, -0.0574],
       [11.9278, 0.5622]]
```
