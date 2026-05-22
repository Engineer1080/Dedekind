# Differenzierbare CFD & Shape-Optimierung in Dedekind

Dieses Dokument beschreibt das CFD-Modul von Dedekind, mit besonderem Fokus
auf die differenzierbare Form-Parametrisierung für aerodynamische und
hydrodynamische Optimierung.

## Überblick

Dedekind kombiniert einen vollständigen Lattice-Boltzmann-Solver (D2Q9) mit
PyTorch-Autograd. Form-Parameter wie Radius, Halbachsen oder Fourier-Koeffizienten
sind native Tensoren — der Gradient des Drag- oder Lift-Werts nach diesen
Parametern fällt automatisch aus dem Forward-Pass. Damit entfällt der separate
Adjoint-Solver, wie er in OpenFOAM, SU2 oder dolfin-adjoint benötigt wird.

Aktivierung: `use fluid_dynamics`

## Form-Parametrisierungen

### 1. Soft-Zylinder — `soft_cylinder_mask(nx, ny, cx, cy, r, alpha)`

Sigmoid-Mask um einen Kreis mit Radius `r`. Differenzierbar in `cx, cy, r`.
Geeignet für klassische Karman-Strömung und einfache Drag-Studien.

### 2. Soft-Ellipse — `soft_ellipse_mask(nx, ny, cx, cy, a, b, alpha)`

Sigmoid-Mask um eine Ellipse mit Halbachsen `a` (x) und `b` (y). Differenzierbar
in `cx, cy, a, b`. Mit Volumen-Constraint `a·b = const` reduziert sich die
Optimierung auf einen einzigen Aspect-Ratio-Parameter — minimaler Suchraum,
schneller Adam-Lauf.

Demo: `examples/dedekind/engineering/lbm_shape_optimization.ddk`. Reduziert
den Drag um ~17 % gegenüber dem Kreis durch Streckung in Strömungsrichtung.

### 3. Soft-Airfoil — `soft_airfoil_mask(nx, ny, t, c, beta, x_start, x_end, y_center, alpha)`

NACA-artige Profilbeschreibung mit Dicken-(`t`), Wölbungs-(`c`) und
Hinterkanten-Profilierungs-(`beta`)-Parametern. Geeignet als Startpunkt für
klassische Tragflächen.

### 4. Fourier-Form — `fourier_shape_mask(nx, ny, cx, cy, r0, a_coeffs, b_coeffs, alpha)`

Universelle 2D-Topologie-Beschreibung über Fourier-Reihe der Randkontur:

```
r(θ) = r0 · (1 + Σ_{k=1..K} a_k·cos(k·θ) + b_k·sin(k·θ))
```

- `a_coeffs`, `b_coeffs`: 1D-Tensoren oder Listen mit K Einträgen — für
  jeden Cos- und Sin-Modus ein Freiheitsgrad.
- Differenzierbar in `cx, cy, r0` und jedem Koeffizienten.
- Methodisch entspricht das der "Class-Shape-Transformation" (CST), wie sie
  in der Aerospace-Industrie für aerodynamische Form-Optimierung etabliert ist.

Glatte, beliebig komplexe Topologien werden bereits mit K=4 Harmonischen
(8 Parameter) abgedeckt: Tropfen-, Tragflächen-, Tear-Drop-, Bohnen-Formen.

Demo: `examples/dedekind/engineering/lbm_fourier_airfoil_optimization.ddk`.
Adam differenziert durch 150 LBM-Schritte und reduziert den Drag um
~15–25 % durch Optimierung aller acht Koeffizienten gleichzeitig.

## Workflow für Shape-Optimierung

```dedekind
use fluid_dynamics

fn drag_of_shape(params) {
    a = [params[0], params[1], params[2], params[3]]
    b = [params[4], params[5], params[6], params[7]]
    M = fourier_shape_mask(80, 40, 25.0, 20.0, 4.0, a, b, 0.4)
    sim = lbm_simulation_full(80, 40, 0.7, M, 0.06)
    simulation_run(sim, 150, 0.06)
    F = simulation_get_drag_lift(sim)
    return [F[0]]
}

// Adam-Optimierer differenziert via Autograd durch die gesamte Pipeline
x0     = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
result = minimize(drag_of_shape, x0, "adam", 0.05, 20)
```

## Validierung

- **Autograd vs. zentrale finite Differenzen**: relativer Fehler < 10⁻⁴
  (siehe `tests/dedekind/fluid_dynamics_test.ddk::test_autograd_validates_against_fd`).
- **Karman-Benchmark**: Strouhal-Zahl St ≈ 0.167 bei Re=100 (Literatur 0.16–0.17).
- **MRT-Stabilität bei Re=2000**: BGK divergiert, MRT bleibt finite.

## Tests

- `tests/dedekind/fluid_dynamics_test.ddk` — 15 Sub-Tests für LBM-Grundfunktionen,
  einheiten-bewusste API, MRT, hard/soft Bounce-Back, Autograd-Validierung.
- `tests/dedekind/lbm_shape_opt_test.ddk` — Ellipsen-Maske + 1-Parameter-Optimierung.
- `tests/dedekind/lbm_fourier_shape_test.ddk` — Fourier-Maske + Multi-Parameter-Optimierung,
  Tropfen-Asymmetrie-Verifikation, Multi-Parameter-Autograd vs. FD.
- `tests/dedekind/ns_ibm_test.ddk` — Chorin-NS-Solver + IBM-Brinkman-Penalisierung.

## Industrielle Relevanz

Die Fourier-Form-Parametrisierung ist die etablierte Methode für:

- **Luftfahrt:** Class-Shape-Transformation in der Aerospace-Industrie für
  Tragflächen- und Karosserieoptimierung.
- **Automobilbau:** Karosserie-Drag-Minimierung; Strömungsabriss-Vermeidung
  an Außenspiegeln und Heckklappen.
- **Schiffbau:** Rumpfprofile, Stevenkontur, Ruderblattgestaltung.
- **Windkraft:** Rotorblattprofile mit variablen Spannweitenstationen.
- **Bauphysik:** Aerodynamik von Hochhäusern, Brückenpfeilern bei
  Kármán-Wirbel-Resonanz.

Dedekind bietet diese Methodik nativ in einer einzigen DSL: keine Kopplung
zwischen Mesher, Solver und Adjoint-Modul, keine Skript-Pipeline aus
Python/MATLAB/OpenFOAM/SU2 — eine Datei, ein `minimize(...)`-Aufruf.
