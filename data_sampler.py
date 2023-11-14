from src.entropy_utils import EntropyTools

import matplotlib.pyplot as plt
import csv


def main():
    quad_order = 20
    moment_degree = 3  # higher moment degree means multimodal densities etc ==> 2 is close to maxwellian
    regularization = 0.0
    et = EntropyTools(quad_order=quad_order, polynomial_degree=moment_degree, spatial_dimension=1, gamma=regularization)

    # Create Entropy Tools with given quadrature order
    n_samples = 10
    condition_treshold = 10  # higher means more anisotropic densities
    max_alpha = 1.0  # higher value means more anisotropic densities
    f, _, _, _ = et.rejection_sampling(n=n_samples, sigma=condition_treshold, max_alpha=max_alpha)
    v_q = et.quad_pts

    plt.figure()
    for i in range(n_samples):
        plt.plot(v_q, f[:, i])
    plt.show()

    with open('densities.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(f.T)

    return 0


if __name__ == '__main__':
    main()
