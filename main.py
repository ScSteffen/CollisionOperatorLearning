import numpy as np
from src.collision_operator_1D import CollisionOperator1D

import matplotlib.pyplot as plt

def main():
    integration_order = 14
    Q = CollisionOperator1D(integration_order)

    sensor_pts = Q.get_quad_pts()

    f_vq  = np.sin(np.pi*sensor_pts) # get an example function evaluated at sensor pts

    f_out = Q.evaluate_Q(f_vq)
    plt.plot(sensor_pts, f_vq, '-k')
    plt.plot(sensor_pts, f_out, '^-r')
    plt.legend(["f", "Q(f)"])
    plt.show()

    return 0

if __name__ == '__main__':
    main()
