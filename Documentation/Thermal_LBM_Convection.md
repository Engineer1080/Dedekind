# Thermal LBM in Dedekind — Konvektion und Wärmetauscher

Diese Dokumentation beschreibt den Thermal-LBM-Solver in Dedekind: einen
vollständig differenzierbaren Double-Distribution-Solver für gekoppelte
Strömungs-Wärme-Probleme.

## Mathematische Grundlage

### Boussinesq-Approximation

Inkompressible Strömung mit kleiner thermischer Dichteanomalie:

```
∂u/∂t + (u·∇)u = -∇p/ρ₀ + ν∇²u - g·β·(T - T_ref)·ŷ
∂T/∂t + (u·∇)T = α·∇²T
∇·u = 0
```

mit kinematischer Viskosität ν, thermischer Diffusivität α, thermischem
Ausdehnungskoeffizienten β, Schwerebeschleunigung g, Referenztemperatur T_ref.

### Double-Distribution LBM

- **Strömung:** D2Q9-Verteilung f_i mit BGK-Kollision, Relaxationszeit τ_u.
  ν = (τ_u - 0.5)/3 in Lattice-Einheiten.
- **Temperatur:** D2Q5-Verteilung g_i (rest + 4 Achsenrichtungen) als
  passiver Skalar, Relaxationszeit τ_T. α = (τ_T - 0.5)/3.
- **Kopplung:** Boussinesq-Buoyancy als "Shift-Velocity"-Forcing im
  Strömungs-Equilibrium: u_eq_y = u_y + τ_u · F_y/ρ.

### Dimensionslose Kennzahlen

- **Rayleigh-Zahl:** Ra = g·β·ΔT·H³ / (ν·α)
  - Ra < 1708 (Ra_c): reine Konduktion, Nu = 1
  - Ra > 1708: Konvektionsrollen, Nu > 1 (Hopf-Bifurkation)
- **Prandtl-Zahl:** Pr = ν/α. Für τ_u = τ_T gilt Pr = 1 (Modell-Standard).
- **Nusselt-Zahl:** Nu = 1 + H/(α·ΔT) · ⟨u_y · (T - T_ref)⟩
  Misst den konvektiven Anteil am Gesamtwärmetransport relativ zu reiner
  Konduktion.

## API

Aktivierung: `use fluid_dynamics`

### Konstruktoren

```dedekind
sim = lbm_thermal_simulation(nx, ny, tau_u, tau_T,
                              T_hot, T_cold, gravity_beta)
```

- `(nx, ny)`: Gittergröße in Lattice-Einheiten.
- `tau_u, tau_T`: Relaxationszeiten (beide > 0.5 für Stabilität).
- `T_hot`: Temperatur an der unteren Wand (j=0).
- `T_cold`: Temperatur an der oberen Wand (j=ny-1).
- `gravity_beta`: Produkt g·β. Steuert die Buoyancy-Stärke.

Mit Hindernis (Dirichlet-T):

```dedekind
sim = lbm_thermal_with_obstacle(nx, ny, tau_u, tau_T,
                                 T_hot, T_cold, gravity_beta,
                                 mask, T_obstacle)
```

- `mask`: Soft-Maske (Tensor (nx, ny) in [0, 1]) für Hindernis-Zellen.
- `T_obstacle`: feste Temperatur des Hindernisses (Dirichlet-BC).

### Zeitintegration

```dedekind
thermal_step(sim)            // Ein Zeitschritt
thermal_run(sim, n_steps)    // Mehrere Schritte
```

### Felder & Diagnostik

```dedekind
T  = thermal_temperature(sim)   // (nx, ny) Temperaturfeld
u  = thermal_velocity(sim)      // (2, nx, ny) Geschwindigkeitsfeld
Nu = thermal_nusselt(sim)       // Skalare globale Nusselt-Zahl
Ra = rayleigh_number(tau_u, tau_T, delta_T, gravity_beta, H)  // Helper
```

### Initialisierung mit Perturbation

Die Default-Initial-T-Verteilung ist linear (konduktiv). Float64-Rundungsrauschen
reicht oft nicht zur Symmetriebrechung — eine deterministische Perturbation
beschleunigt den Konvektionseinsatz dramatisch:

```dedekind
T_init = thermal_temperature(sim)
rows = linspace(0.0, nx - 1.0, nx)
cols = linspace(0.0, ny - 1.0, ny)
IX = outer(rows, 1.0 + 0.0 * cols)
IY = outer(1.0 + 0.0 * rows, cols)
pert = 0.05 * sin(2.0 * pi * IX / 25.0) * sin(pi * IY / (ny - 1.0))
thermal_set_temperature(sim, T_init + pert)
```

## Validierung

- **Reine Konduktion (g·β = 0):** linearer T-Verlauf, Nu = 1.000 exakt.
- **Bei g·β = 0:** Geschwindigkeitsfeld bleibt strikt 0 (keine Drift).
- **Konvektionseinsatz:** Ra > 1708 erzeugt Nu > 1, sichtbare Konvektionsrollen.
- **Dirichlet-Hindernis:** T_obstacle wird im Stab-Zentrum erreicht (T ≈ T_obstacle).

Siehe `tests/dedekind/thermal_lbm_test.ddk` (5 Sub-Tests).

## Beispiele

### Rayleigh-Bénard-Konvektion

`examples/dedekind/engineering/lbm_rayleigh_benard.ddk` zeigt:

- Reine Konduktion (Baseline): Nu = 1.000
- Konvektion bei Ra ≈ 19440: Nu ≈ 1.81 (80 % zusätzlicher Wärmetransport)

### Wärmetauscher-Geometrie-Studie

`examples/dedekind/engineering/lbm_heat_exchanger.ddk` zeigt:

- Bénard-Konvektionszelle mit zusätzlichem Kühl-Stab (Dirichlet T_cold)
- Position des Stabs (nahe heißer Wand, Mitte, nahe kalter Wand) beeinflusst
  die globale Nu deutlich.
- Industrierelevant: Reaktor-Brennstab-Anordnung, Server-Raum-Klimatisierung,
  Solar-Pond-Schichtung, Verdampfer-/Kondensator-Design.

## Differenzierbarkeit

Sämtliche Eingabeparameter (cy_position, r, T_obstacle, gravity_beta, τ_u, τ_T)
sind PyTorch-Tensoren und tragen Gradienten. Optimierungsalgorithmen wie
Adam können direkt durch hunderte Zeitschritte hindurch zur optimalen
Geometrie navigieren — ohne separaten Adjoint-Solver.

Beispiel-Pattern (Adam optimiert Stab-Position):

```dedekind
fn nu_objective(params) {
    cy = params[0]
    M  = soft_cylinder_mask(nx, ny, cx_const, cy, r_const, 0.5)
    sim = lbm_thermal_with_obstacle(nx, ny, 0.6, 0.6,
                                     1.0, 0.0, 1e-4, M, 0.0)
    thermal_run(sim, 2000)
    return [-thermal_nusselt(sim)]   // negativ → Maximierung
}

result = minimize(nu_objective, [25.0], "adam", 0.5, 30)
```

## Limitierungen & Erweiterungen

- **2D only:** Aktuell nur D2Q9/D2Q5; 3D (D3Q19/D3Q7) als zukünftige
  Erweiterung möglich.
- **Boussinesq nur:** Stark-kompressible thermische Strömungen (z.B. Stoßwellen)
  brauchen ein vollständiges Energie-Modell.
- **Adiabatic boundaries:** Aktuell nur Dirichlet-T an Hindernissen. Neumann/
  adiabatische Wände sind als Erweiterung implementierbar.
