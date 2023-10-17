import numpy as np
from src.collision_operator_1D import CollisionOperator1D

import matplotlib.pyplot as plt


def main():
    integration_order = 10
    Q = CollisionOperator1D(integration_order)

    sensor_pts = Q.get_quad_pts()

    f_vq = np.sin(np.pi * sensor_pts) + 1  # get an example function evaluated at sensor pts

    f_out = Q.evaluate_Q(f_vq, g=0.9)
    f_out2 = Q.evaluate_Q(f_vq, g=0.5)
    f_out3 = Q.evaluate_Q(f_vq, g=0.2)
    f_out4 = Q.evaluate_Q(f_vq, g=-0.2)
    f_out5 = Q.evaluate_Q(f_vq, g=-0.5)
    f_out6 = Q.evaluate_Q(f_vq, g=-0.9)

    gs = np.linspace(-1, 1, 10)

    fig, ax = plt.subplots()
    ax.plot(sensor_pts, f_vq, '-k', label="f")
    for i in range(10):
        f_out = Q.evaluate_Q(f_vq, g=gs[i])
        ax.plot(sensor_pts, f_out, label="Q(f)," + str(gs[i]))
    ax.legend()
    fig.show()
    plt.savefig("collision.png")
    # plt.show()

    return 0


if __name__ == '__main__':
    main()
