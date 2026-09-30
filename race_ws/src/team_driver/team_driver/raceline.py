#!/usr/bin/env python3
import os
import csv
import math
import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d

# Vehicle and optimization parameters
MAX_LAT_OFFSET = 0.35   # Allowable lateral deviation from centerline (meters)
MAX_SPEED = 8.0         # Top speed target (m/s)
MAX_LAT_ACCEL = 7.5     # Maximum lateral grip limit (m/s^2)
MAX_ACCEL = 4.5         # Maximum longitudinal acceleration limit (m/s^2)
MAX_DECEL = 6.5         # Maximum braking deceleration limit (m/s^2)
PAD = 15                # Wrap length for closed-loop calculations

def compute_curvature(x, y):
    """Calculates path curvature kappa over a closed loop."""
    x_pad = np.pad(x, PAD, mode='wrap')
    y_pad = np.pad(y, PAD, mode='wrap')

    dx = np.gradient(x_pad)
    dy = np.gradient(y_pad)
    ds = np.hypot(dx, dy)
    ds[ds < 1e-6] = 1e-6

    dx_ds = dx / ds
    dy_ds = dy / ds
    ddx_ds = np.gradient(dx_ds) / ds
    ddy_ds = np.gradient(dy_ds) / ds

    num = dx_ds * ddy_ds - dy_ds * ddx_ds
    den = (dx_ds**2 + dy_ds**2)**1.5
    den[den < 1e-6] = 1e-6

    kappa = num / den
    return kappa[PAD:-PAD]

def main():
    input_csv = "/hackathon/maps/icra26_centerline.csv"
    output_csv = "/hackathon/race_ws/src/team_driver/team_driver/maps/icra26_raceline.csv"

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Centerline CSV not found: {input_csv}")

    # 1. Load raw centerline coordinates
    raw_x, raw_y = [], []
    with open(input_csv, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row and not row[0].startswith('#'):
                raw_x.append(float(row[0]))
                raw_y.append(float(row[1]))

    x_center = np.array(raw_x, dtype=np.float64)
    y_center = np.array(raw_y, dtype=np.float64)
    N = len(x_center)

    # 2. Compute normal vectors perpendicular to the centerline
    x_pad = np.pad(x_center, PAD, mode='wrap')
    y_pad = np.pad(y_center, PAD, mode='wrap')

    dx = np.gradient(x_pad)[PAD:-PAD]
    dy = np.gradient(y_pad)[PAD:-PAD]
    norm = np.hypot(dx, dy)
    norm[norm < 1e-6] = 1e-6

    nx = -dy / norm
    ny = dx / norm

    # 3. Objective function: Minimize sum of squared curvature
    def objective(alpha):
        x_opt = x_center + alpha * nx
        y_opt = y_center + alpha * ny
        kappa = compute_curvature(x_opt, y_opt)
        return np.sum(kappa**2)

    # Solve optimization with lateral offset bounds
    bounds = [(-MAX_LAT_OFFSET, MAX_LAT_OFFSET)] * N
    print(f"Optimizing {N} waypoints for minimum curvature...")
    res = minimize(objective, np.zeros(N), bounds=bounds, method='L-BFGS-B')

    alpha_opt = res.x
    x_opt = x_center + alpha_opt * nx
    y_opt = y_center + alpha_opt * ny

    # 4. Smooth optimized waypoints
    x_opt = gaussian_filter1d(np.pad(x_opt, PAD, mode='wrap'), sigma=1.2)[PAD:-PAD]
    y_opt = gaussian_filter1d(np.pad(y_opt, PAD, mode='wrap'), sigma=1.2)[PAD:-PAD]

    # 5. Compute Arc Length (s), Heading (psi), and Curvature (kappa)
    dx_opt = np.diff(x_opt, prepend=x_opt[-1])
    dy_opt = np.diff(y_opt, prepend=y_opt[-1])
    ds = np.hypot(dx_opt, dy_opt)
    ds[ds < 1e-6] = 1e-6
    s = np.cumsum(ds)

    psi = np.arctan2(np.gradient(y_opt), np.gradient(x_opt))
    kappa = compute_curvature(x_opt, y_opt)

    # 6. Two-Pass Velocity Profile Generation
    v_lat = np.sqrt(MAX_LAT_ACCEL / (np.abs(kappa) + 1e-3))
    v_profile = np.minimum(MAX_SPEED, v_lat)

    v_ext = np.tile(v_profile, 2)
    ds_ext = np.tile(ds, 2)
    N_ext = len(v_ext)

    # Backward pass (Braking limit)
    for i in range(N_ext - 2, -1, -1):
        v_max_brake = np.sqrt(v_ext[i + 1]**2 + 2.0 * MAX_DECEL * ds_ext[i + 1])
        v_ext[i] = min(v_ext[i], v_max_brake)

    # Forward pass (Acceleration limit)
    for i in range(1, N_ext):
        v_max_accel = np.sqrt(v_ext[i - 1]**2 + 2.0 * MAX_ACCEL * ds_ext[i])
        v_ext[i] = min(v_ext[i], v_max_accel)

    vx = np.clip(v_ext[N:2 * N], 2.5, MAX_SPEED)
    ax = np.clip(np.gradient(vx, s), -MAX_DECEL, MAX_ACCEL)

    # 7. Write to CSV: s; x; y; psi; kappa; vx; ax
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        f.write("# s; x; y; psi; kappa; vx; ax\n")
        for i in range(N):
            f.write(f"{s[i]:.4f};{x_opt[i]:.4f};{y_opt[i]:.4f};{psi[i]:.4f};{kappa[i]:.4f};{vx[i]:.4f};{ax[i]:.4f}\n")

    print(f"Successfully generated racing line at: {output_csv}")

if __name__ == '__main__':
    main()
