from src.entropy_utils import EntropyTools

import matplotlib.pyplot as plt


def main():
    quad_order = 100
    moment_degree = 2
    regularization = 0.0
    et = EntropyTools(quad_order=quad_order, polynomial_degree=moment_degree, spatial_dimension=1, gamma=regularization)

    # Create Entropy Tools with given quadrature order
    n_samples = 10
    condition_treshold = 4  # higher means more anisotropic moments
    f, _, _, _ = et.rejection_sampling(n=n_samples, sigma=condition_treshold, max_alpha=1)
    v_q = et.quad_pts

    plt.figure()
    for i in range(n_samples):
        plt.plot(v_q, f[:, i])
    plt.show()

    return 0


if __name__ == '__main__':
    main()
