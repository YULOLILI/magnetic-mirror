"""
Magnetic Mirror Simulation — Charged Particle Trajectory in a Non-uniform Magnetic Field

Simulates a charged particle moving in a magnetic mirror (magnetic bottle) configuration.
Solves the Lorentz force ODE system using SciPy's RK45, analyzes the adiabatic invariant
(magnetic moment μ), and generates 3D trajectory visualization + animation.

Based on: PHY 286 Computational Physics Course Project
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import os

# ── Output directory ────────────────────────────────────────────
OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Parameters ──────────────────────────────────────────────────
q, m = 1.0, 1.0          # charge and mass
B0, alpha = 1.0, 0.01    # magnetic mirror field parameters
r0 = [1.0, 0.0, -2.0]    # initial position
v0 = [1.0, 1.5, 1.5]     # initial velocity
y0 = r0 + v0              # initial state [x, y, z, vx, vy, vz]

# ── Magnetic Field Model ────────────────────────────────────────
def B_field(r):
    """
    Magnetic mirror field:
      Bz = B0 * (1 + α*z²)
      Bx = -0.5 * B0 * α * z * x
      By = -0.5 * B0 * α * z * y

    Satisfies ∇·B = 0 by construction.
    """
    x, y, z = r
    Bz = B0 * (1 + alpha * z**2)
    Bx = -0.5 * B0 * alpha * z * x
    By = -0.5 * B0 * alpha * z * y
    return np.array([Bx, By, Bz])

# ── Lorentz Force ODE ───────────────────────────────────────────
def lorentz(t, state):
    x, y, z, vx, vy, vz = state
    r = np.array([x, y, z])
    v = np.array([vx, vy, vz])
    B = B_field(r)
    a = (q / m) * np.cross(v, B)
    return [vx, vy, vz, a[0], a[1], a[2]]

# ── Numerical Solution ──────────────────────────────────────────
t_span = (0, 100)
t_eval = np.linspace(0, 100, 3000)

print("Solving ODE (RK45, rtol=1e-9, atol=1e-12)...")
sol = solve_ivp(lorentz, t_span, y0, t_eval=t_eval, rtol=1e-9, atol=1e-12)
print("Solution complete!")

# Extract data
x, y, z = sol.y[0], sol.y[1], sol.y[2]
vx, vy, vz = sol.y[3], sol.y[4], sol.y[5]
t = sol.t

# ── Magnetic Moment & Energy Analysis ───────────────────────────
print("Calculating magnetic moment and energy...")
mu = np.zeros(len(t))
B_mag = np.zeros(len(t))
v_perp_sq = np.zeros(len(t))
kinetic_energy = np.zeros(len(t))

for i in range(len(t)):
    r_i = np.array([x[i], y[i], z[i]])
    v_i = np.array([vx[i], vy[i], vz[i]])
    B_i = B_field(r_i)
    B_mag[i] = np.linalg.norm(B_i)

    v_parallel = np.dot(v_i, B_i) / B_mag[i]
    v_perp_sq[i] = np.dot(v_i, v_i) - v_parallel**2
    mu[i] = (m * v_perp_sq[i]) / (2 * B_mag[i])
    kinetic_energy[i] = 0.5 * m * np.dot(v_i, v_i)

mu_norm = mu / mu[0]

# Print summary statistics
print(f"\n── Results ──")
print(f"Initial μ:        {mu[0]:.6f}")
print(f"Final μ:          {mu[-1]:.6f}")
print(f"μ change:         {(mu[-1]-mu[0])/mu[0]*100:.4f}%")
print(f"Initial KE:       {kinetic_energy[0]:.6f}")
print(f"Final KE:         {kinetic_energy[-1]:.6f}")
print(f"KE change:        {(kinetic_energy[-1]-kinetic_energy[0])/kinetic_energy[0]*100:.4f}%")
print(f"μ mean:           {mu_norm.mean():.8f}")
print(f"μ std:            {mu_norm.std():.8f}")
print(f"μ min:            {mu_norm.min():.8f} @ t={t[np.argmin(mu_norm)]:.4f}")
print(f"μ max:            {mu_norm.max():.8f} @ t={t[np.argmax(mu_norm)]:.4f}")

# ── Static Analysis Plots ───────────────────────────────────────
print("\n── Generating plots ──")

# Plot 1: 3D Trajectory
print("  [1/6] 3D trajectory...")
fig1 = plt.figure(figsize=(10, 8))
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot(x, y, z, linewidth=0.8)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.set_title('3D Trajectory in Magnetic Mirror')
fig1.savefig(f"{OUT_DIR}/01_3d_trajectory.png", dpi=150, bbox_inches='tight')
plt.close(fig1)

# Plot 2: z(t)
print("  [2/6] z(t)...")
fig2 = plt.figure(figsize=(10, 6))
plt.plot(t, z)
plt.xlabel('Time')
plt.ylabel('z')
plt.title('z(t) — Position along Magnetic Axis')
plt.grid(True)
fig2.savefig(f"{OUT_DIR}/02_z_position.png", dpi=150, bbox_inches='tight')
plt.close(fig2)

# Plot 3: B(t)
print("  [3/6] |B|(t)...")
fig3 = plt.figure(figsize=(10, 6))
plt.plot(t, B_mag)
plt.xlabel('Time')
plt.ylabel('|B|')
plt.title('Magnetic Field Strength along Trajectory')
plt.grid(True)
fig3.savefig(f"{OUT_DIR}/03_B_magnitude.png", dpi=150, bbox_inches='tight')
plt.close(fig3)

# Plot 4: μ(t)/μ(0)
print("  [4/6] Normalized magnetic moment...")
fig4 = plt.figure(figsize=(14, 7))
plt.plot(t, mu_norm, 'b-', linewidth=1.5, label='μ(t)/μ(0)')
y_min, y_max = mu_norm.min(), mu_norm.max()
y_margin = (y_max - y_min) * 0.05
plt.ylim(y_min - y_margin, y_max + y_margin)
plt.xlabel('Time', fontsize=12)
plt.ylabel('μ(t)/μ(0)', fontsize=12)
plt.title(f'Adiabatic Invariant μ (range: [{y_min:.4f}, {y_max:.4f}])', fontsize=12)
plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='μ conserved (ideal = 1)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
fig4.savefig(f"{OUT_DIR}/04_magnetic_moment.png", dpi=150, bbox_inches='tight')
plt.close(fig4)

# Plot 5: Kinetic Energy
print("  [5/6] Kinetic energy...")
fig5 = plt.figure(figsize=(10, 6))
plt.plot(t, kinetic_energy)
plt.xlabel('Time')
plt.ylabel('Kinetic Energy')
plt.title('Kinetic Energy (constant — B does no work)')
plt.grid(True)
fig5.savefig(f"{OUT_DIR}/05_kinetic_energy.png", dpi=150, bbox_inches='tight')
plt.close(fig5)

# Plot 6: xy Projection
print("  [6/6] xy projection...")
fig6 = plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('xy Projection (Gyration Motion)')
plt.axis('equal')
plt.grid(True)
fig6.savefig(f"{OUT_DIR}/06_xy_projection.png", dpi=150, bbox_inches='tight')
plt.close(fig6)

print(f"All plots saved to '{OUT_DIR}/'")

# ── 3D Animation ────────────────────────────────────────────────
print("\n── Generating 3D animation ──")

plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

fig_anim = plt.figure(figsize=(10, 8))
ax_anim = fig_anim.add_subplot(111, projection='3d')

ax_anim.set_xlabel(r'$x$', fontsize=14, fontweight='bold')
ax_anim.set_ylabel(r'$y$', fontsize=14, fontweight='bold')
ax_anim.set_zlabel(r'$z$', fontsize=14, fontweight='bold')
ax_anim.set_title('Charged Particle Trajectory in a Magnetic Mirror', fontsize=16, fontweight='bold', pad=20)

x_range, y_range, z_range = max(x)-min(x), max(y)-min(y), max(z)-min(z)
center_x, center_y, center_z = (max(x)+min(x))/2, (max(y)+min(y))/2, (max(z)+min(z))/2
max_range = max(x_range, y_range, z_range) / 2

ax_anim.set_xlim(center_x - max_range, center_x + max_range)
ax_anim.set_ylim(center_y - max_range, center_y + max_range)
ax_anim.set_zlim(center_z - max_range, center_z + max_range)
ax_anim.view_init(elev=25, azim=-60)
ax_anim.grid(True, alpha=0.3)

traj_line, = ax_anim.plot([], [], [], 'r-', linewidth=1.0, alpha=0.8, label='Trajectory')
current_point, = ax_anim.plot([], [], [], 'bo', markersize=8, markeredgecolor='darkblue',
                               markeredgewidth=1.5, label='Current position')
start_point, = ax_anim.plot([x[0]], [y[0]], [z[0]], 'g*', markersize=12, label='Start point')
ax_anim.legend(loc='upper right', fontsize=10, framealpha=0.9)

n_frames = len(t) // 3

def init():
    traj_line.set_data([], [])
    traj_line.set_3d_properties([])
    current_point.set_data([], [])
    current_point.set_3d_properties([])
    return traj_line, current_point

def animate(i):
    idx = int(i * len(t) / n_frames)
    idx = min(idx, len(t) - 1)
    traj_line.set_data(x[:idx], y[:idx])
    traj_line.set_3d_properties(z[:idx])
    current_point.set_data([x[idx]], [y[idx]])
    current_point.set_3d_properties([z[idx]])
    ax_anim.set_title(f'Charged Particle Trajectory in a Magnetic Mirror\n'
                      f'Time = {t[idx]:.1f} | Position = ({x[idx]:.2f}, {y[idx]:.2f}, {z[idx]:.2f})',
                      fontsize=12, fontweight='normal')
    return traj_line, current_point

print(f"  Generating {n_frames} frames...")
ani = animation.FuncAnimation(fig_anim, animate, init_func=init,
                              frames=n_frames, interval=25, blit=True)

try:
    gif_path = f"{OUT_DIR}/particle_trajectory.gif"
    ani.save(gif_path, writer='pillow', fps=30, dpi=100)
    print(f"  Animation saved to {gif_path}")
except Exception as e:
    print(f"  Failed to save GIF: {e}")

plt.close(fig_anim)
print("\nDone! All analysis complete.")
