import numpy as np
import matplotlib.pyplot as plt

from src.collision_operator_HG_1D import CollisionOperatorHG1D


def main():
    # Define parameters
    L = 10.0  # Length of the 1D domain
    n_x = 100  # Number of spatial grid points
    dx = L / n_x  # Spatial grid size
    sigma_t = 1.0  # Total cross-section
    sigma_s = 0.5  # Scattering cross-section
    source = np.zeros(n_x)  # Source term

    # Angular discretization (Discrete Ordinates)
    n_q = 10  # Number of discrete angles
    Q = CollisionOperatorHG1D(integrator_order=n_q, anisotropy_param=0.0)
    omega = Q.get_quad_pts()
    w_q = Q.get_quad_weights()
    # Initialize angular flux and boundary conditions
    f_init = np.zeros((n_q, n_x))
    f_init[:, 0] = 1.0  # Left boundary (Dirichlet boundary condition)
    f_init[:, - 1] = 0.0  # Right boundary (Dirichlet boundary condition)

    # Define the time step and number of time steps
    dt = 0.4 * dx  # obeys cfl
    n_t = 100

    f_tN = run_simulation(f_init, n_x, n_q, n_t, dt, dx, sigma_s, sigma_t, source, omega, Q)

    f_res_tn = w_q @ f_tN / 2.
    f_res_t0 = w_q @ f_init / 2.
    # Plot the angular flux for one angle (e.g., the first angle)
    plt.plot(np.linspace(0, L, n_x), f_res_t0)
    plt.plot(np.linspace(0, L, n_x), f_res_tn)
    plt.xlabel('Position (cm)')
    plt.ylabel('Angular Flux')
    plt.title('1D Discrete Ordinates with Dirichlet Boundary Conditions')
    plt.legend(['t0', 'tN'])
    plt.grid(True)
    plt.show()
    return 0


def get_upwind_flux(i, omega_q, f_q):
    if omega_q >= 0:
        return omega_q * f_q[i - 1]
    return omega_q * f_q[i + 1]


def run_simulation(f_init, n_x, n_q, n_t, dt, dx, sigma_s, sigma_t, source, omega, Q: CollisionOperatorHG1D):
    f_kin = np.copy(f_init)
    fluxes = np.zeros(shape=(n_q, n_x))
    # Perform time-stepping
    for k in range(n_t - 1):
        for i in range(1, n_x - 1):
            for q in range(n_q):
                fluxes[q, i] = get_upwind_flux(i, omega[q], f_kin[q, :])
        for i in range(1, n_x - 1):
            collision = Q.evaluate_Q(f_kin[:, i])

            for q in range(n_q):
                # Calculate streaming term

                # Update angular flux using finite difference method
                f_kin[q, i] = f_kin[q, i] - dt / dx * (
                        fluxes[q, i + 1] - fluxes[q, i]) + dt * collision[q] + dt * source[i]

    return f_kin


if __name__ == '__main__':
    # test_1d()
    main()
