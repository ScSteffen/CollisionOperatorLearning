import numpy as np
from src.collision_operator_1D import CollisionOperator1D
from src.collision_operator_HG_1D import CollisionOperatorHG1D

import matplotlib.pyplot as plt


def main():
    integration_order = 14
    g = 0.5  # needs to be between -1 and 1, g=0 is isotropic
    Q = CollisionOperatorHG1D(integrator_order=integration_order, anisotropy_param=g)
    # Q = CollisionOperator1D(integrator_order=integration_order)

    sensor_pts = Q.get_quad_pts()

    f_vq = np.sin(np.pi * sensor_pts) + 1  # get an example function evaluated at sensor pts

    f_out = Q.evaluate_Q(f_vq)
    plt.plot(sensor_pts, f_vq, '-k')
    plt.plot(sensor_pts, f_out, '^-r')
    plt.legend(["f", "Q(f)"])
    plt.show()

    return 0


if __name__ == '__main__':
    main()
