import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.collision_operator_HG_3D import CollisionOperatorHG3D


def setup_lattice_test(n_x, n_y):
    # Define parameters
    L = 7.0  # Length of the 1D domain

    dx = L / n_x  # Spatial grid size
    dy = L / n_y  # Spatial grid size

    source_strength = 1.0
    source = np.zeros((n_x, n_y))  # Source term

    sigma_a = np.zeros((n_x, n_y))
    sigma_s = np.ones((n_x, n_y))

    lbounds = [1, 2, 3, 4, 5]
    ubounds = [2, 3, 4, 5, 6]

    for i in range(n_x):
        for j in range(n_y):
            pos = [i * dx, j * dy]
            for k in range(len(lbounds)):
                for l in range(len(lbounds)):
                    if lbounds[k] <= pos[0] <= ubounds[k] and lbounds[l] <= pos[1] <= ubounds[l]:
                        if (l + k) % 2 == 1 or (k == 2 and l == 2) or (k == 2 and l == 4):
                            continue
                        else:
                            sigma_a[i, j] = 10.
                            sigma_s[i, j] = 0.0

            if 3 <= pos[0] <= 4 and 3 <= pos[1] <= 4:
                source[i, j] = source_strength
    return sigma_a, sigma_s, source, dx, dy, L


def main():
    n_x = 20
    n_y = 20
    sigma_a, sigma_s, source, dx, dy, length = setup_lattice_test(n_x=n_x, n_y=n_y)
    # Angular discretization (Discrete Ordinates)
    q_order = 4  # quadrature order
    Q = CollisionOperatorHG3D(integrator_order=q_order, anisotropy_param=0.0)  # EXCHANGE BY nueral Network!
    omega = Q.get_quad_pts()
    w_q = Q.get_quad_weights()

    n_q = Q.n_q
    # Initialize angular flux and boundary conditions
    f_init = np.zeros((n_q[0], n_q[1], n_x, n_y))  # tensorized in (my,phi,x,y)
    # f_init[:, :, 5:10, 5:10] = 1.0
    # Define the time step and number of time steps
    dt = 0.4 * dx * dy / (dx + dy)  # obeys cfl
    n_t = 200

    print('Start simulation')
    f_tN = run_simulation(f_init, n_x, n_y, n_q, n_t, dt, dx, dy, sigma_s, sigma_a, source, omega, Q)
    print('Finished simulation at t_n = ' + str(dt * n_t) + ' sec.')

    f_res_tn = np.einsum('ij,ijkl', w_q, f_tN) / (4 * np.pi)
    f_res_t0 = np.einsum('ij,ijkl', w_q, f_init) / (4 * np.pi)
    # Plot the angular flux for one angle (e.g., the first angle)
    plt.imshow(f_res_tn.T, cmap='inferno', interpolation='nearest')
    plt.colorbar()
    plt.savefig('lattice_2d_tn.png')
    plt.savefig('lattice_2d_tn.pdf')
    plt.show()

    plt.clf()
    plt.imshow(f_res_t0.T, cmap='inferno', interpolation='nearest')
    plt.colorbar()
    plt.savefig('lattice_2d_t0.png')
    plt.savefig('lattice_2d_t0.pdf')
    return 0


def get_upwind_flux(i, j, omega_q, f_q, dx, dy):
    flux = 0.0
    flux += -dy * omega_q[0] * f_q[i, j] if -1 * omega_q[0] >= 0 else -dy * omega_q[0] * f_q[i - 1, j]  # left nbr
    flux += dy * omega_q[0] * f_q[i, j] if omega_q[0] >= 0 else dy * omega_q[0] * f_q[i + 1, j]  # right nbr
    flux += -dx * omega_q[1] * f_q[i, j] if -1 * omega_q[1] >= 0 else -dx * omega_q[1] * f_q[i, j - 1]  # down nbr
    flux += dx * omega_q[1] * f_q[i, j] if omega_q[1] >= 0 else dx * omega_q[1] * f_q[i, j + 1]  # up nbr

    return flux


def run_simulation(f_init, n_x, n_y, n_q, n_t, dt, dx, dy, sigma_s, sigma_a, source, omega, Q: CollisionOperatorHG3D):
    f_kin = np.copy(f_init)
    fluxes = np.zeros(shape=(n_q[0], n_q[1], n_x, n_y))
    area = (dy * dx) / (dx + dy)
    # Perform time-stepping
    for k in tqdm(range(n_t)):
        for i in range(1, n_x - 1):
            for j in range(1, n_y - 1):
                for q1 in range(n_q[0]):
                    for q2 in range(n_q[1]):
                        # fluxes in 2D
                        fluxes[q1, q2, i, j] = get_upwind_flux(i, j, omega[:, q1, q2], f_kin[q1, q2, :, :], dx, dy)

        for i in range(1, n_x - 1):
            for j in range(1, n_y - 1):
                collision = Q.evaluate_Q(f_kin[:, :, i, j])  # dependent on omega(my phi)=[v_x,v_y]

                for q1 in range(n_q[0]):
                    for q2 in range(n_q[1]):
                        # Update angular flux using finite difference method
                        f_kin[q1, q2, i, j] = (f_kin[q1, q2, i, j]
                                               - dt / area * fluxes[q1, q2, i, j]
                                               + dt * sigma_s[i, j] * collision[q1, q2]
                                               + dt * source[i, j]
                                               - dt * sigma_a[i, j] * f_kin[q1, q2, i, j])
    return f_kin


if __name__ == '__main__':
    # test_1d()
    main()
