import numpy as np
import matplotlib.pyplot as plt
from src.collision_operator_HG_1D import CollisionOperatorHG1D
from tqdm import tqdm

import torch
from model.deeponet import *


def get_upwind_flux(i, omega_q, f_q):
    if omega_q >= 0:
        return omega_q * f_q[i - 1]
    return omega_q * f_q[i]


def run_classic_simulation(
    f_init,
    n_x,
    n_q,
    n_t,
    dt,
    dx,
    sigma_s,
    sigma_t,
    source,
    omega,
    Q: CollisionOperatorHG1D,
):
    f_kin = np.copy(f_init)
    fluxes = np.zeros(shape=(n_q, n_x))
    # Perform time-stepping
    for k in tqdm(range(n_t - 1)):
        for i in range(1, n_x - 1):
            for q in range(n_q):
                fluxes[q, i] = get_upwind_flux(i, omega[q], f_kin[q, :])
        for i in range(1, n_x - 1):
            collision = Q.evaluate_Q(f_kin[:, i])

            for q in range(n_q):
                # Calculate streaming term

                # Update angular flux using finite difference method
                f_kin[q, i] = (
                    f_kin[q, i]
                    - dt / dx * (fluxes[q, i + 1] - fluxes[q, i])
                    + dt * collision[q]
                    + dt * source[i]
                )

    return f_kin


def run_neural_simulation(
    f_init, n_x, n_q, n_t, dt, dx, sigma_s, sigma_t, source, omega, model, grid
):
    f_kin = np.copy(f_init)
    fluxes = np.zeros(shape=(n_q, n_x))
    # Perform time-stepping
    for k in range(n_t - 1):
        for i in range(1, n_x - 1):
            for q in range(n_q):
                fluxes[q, i] = get_upwind_flux(i, omega[q], f_kin[q, :])
        for i in range(1, n_x - 1):
            _f = torch.FloatTensor(f_kin[:, i].reshape(1, 1, -1))
            collision = model(_f, grid).squeeze().detach().numpy()

            for q in range(n_q):
                # Calculate streaming term

                # Update angular flux using finite difference method
                f_kin[q, i] = (
                    f_kin[q, i]
                    - dt / dx * (fluxes[q, i + 1] - fluxes[q, i])
                    + dt * collision[q]
                    + dt * source[i]
                )

    return f_kin


def main():
    # Define parameters
    L = 10.0  # Length of the 1D domain
    n_x = 100  # Number of spatial grid points
    dx = L / n_x  # Spatial grid size
    sigma_t = 1.0  # Total cross-section
    sigma_s = 0.5  # Scattering cross-section
    source = np.zeros(n_x)  # Source term

    # Angular discretization (Discrete Ordinates)
    n_q = 100  # Number of discrete angles
    Q = CollisionOperatorHG1D(integrator_order=n_q, anisotropy_param=0.0)
    omega = Q.get_quad_pts()
    w_q = Q.get_quad_weights()
    # Initialize angular flux and boundary conditions
    f_init = np.zeros((n_q, n_x))
    f_init[:, 0] = 1.0  # Left boundary (Dirichlet boundary condition)
    f_init[:, -1] = 0.0  # Right boundary (Dirichlet boundary condition)

    # Define the time step and number of time steps
    dt = 0.4 * dx  # obeys cfl
    n_t = 100

    f_tN = run_classic_simulation(f_init, n_x, n_q, n_t, dt, dx, sigma_s, sigma_t, source, omega, Q)
    return 0


if __name__ == "__main__":
    # test_1d()
    main()
