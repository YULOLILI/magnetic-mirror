# Magnetic Mirror Simulation

Charged particle trajectory simulation in a magnetic mirror (magnetic bottle) configuration. Solves the Lorentz force ODE system numerically and analyzes the adiabatic invariant (magnetic moment μ).

## Physics

- **Magnetic mirror field**: Bz = B₀(1 + αz²), with ∇·B = 0 preserved via Bx, By components
- **Lorentz force**: F = q(v × B), integrated via SciPy RK45 (rtol=1e-9)
- **Adiabatic invariant**: μ = mv⊥²/(2B) — tracked over 3000 timesteps to verify conservation

## Key Findings

- Magnetic moment μ conserved to within ~0.01% over the full trajectory
- Kinetic energy remains constant (as expected — B does no work)
- Particle oscillates between magnetic mirror reflection points
- μ conservation degrades where magnetic field gradients are steepest

## Requirements

```bash
pip install numpy scipy matplotlib pillow
```

## Usage

```bash
python magnetic_mirror.py
```

Outputs:
- `output/01_3d_trajectory.png` — Full 3D trajectory
- `output/02_z_position.png` — Position along magnetic axis
- `output/03_B_magnitude.png` — |B| along trajectory
- `output/04_magnetic_moment.png` — Adiabatic invariant μ(t)/μ(0)
- `output/05_kinetic_energy.png` — Energy conservation check
- `output/06_xy_projection.png` — Gyration radius
- `output/particle_trajectory.gif` — 3D animated trajectory

## Course

PHY 286 — Computational Physics, Miami University
