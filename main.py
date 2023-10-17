import numpy as np
import scipy
import matplotlib.pyplot as plt

from src.collision_operator_1D import CollisionOperator1D
from src.collision_operator_HG_3D import CollisionOperatorHG3D
from src.collision_operator_3D import CollisionOperator3D
from src.collision_operator_HG_1D import CollisionOperatorHG1D


def test_1d():
    integration_order = 10
    Q = CollisionOperatorHG1D(integration_order, anisotropy_param=0.0)

    sensor_pts = Q.get_quad_pts()

    f_vq = np.sin(np.pi * sensor_pts) + 1  # get an example function evaluated at sensor pts
    gs = np.linspace(-1, 1, 10)

    fig, ax = plt.subplots()
    ax.plot(sensor_pts, f_vq, '-k', label="f")
    for i in range(10):
        Q.set_anisotropy(anisotropy_param=gs[i])
        f_out = Q.evaluate_Q(f_vq)
        ax.plot(sensor_pts, f_out, label="Q(f)," + str(gs[i]))
    ax.legend()
    fig.show()
    plt.savefig("collision.png")
    return 0


def test_3d():
    integration_order = 20
    n_tests = 10
    Q = CollisionOperatorHG3D(integration_order, anisotropy_param=0.0)

    sensor_pts = Q.get_quad_pts()  # quad points are in tensor format!
    theta_q, phi_q = Q.get_quad_pts_polar()  # polar coordinates - also in tensor format

    # get an example function evaluated at sensor pts
    f_vq = np.exp(
        1 + np.sin(theta_q) * np.sin(phi_q) + np.sin(theta_q) * np.cos(
            phi_q) + np.cos(theta_q))  # np.cos(theta_q)  # * np.ones(Q.n_q)
    # plot function as a test
    plot_on_sphere(np.abs(f_vq) * sensor_pts, f_vq, name='pre collision')
    gs = np.linspace(-1, 1, n_tests)  # anisotropy params

    for i in range(n_tests):
        Q.set_anisotropy(anisotropy_param=gs[i])
        f_out = Q.evaluate_Q(f_vq)
        plot_on_sphere(sensor_pts * np.abs(f_vq), f_out, name="collision_3d_g" + str(gs[i]))

    return 0


def plot_on_sphere(xzy, function_values, name="sphere"):
    # Create a figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Normalize function values to colormap range (0-1)

    # Set colorbar for the function values
    cmap = plt.cm.ScalarMappable(cmap=plt.get_cmap('seismic'))
    # cmap.set_clim(-0.5, 0.5)
    ax.plot_surface(xzy[0], xzy[1], xzy[2],
                    facecolors=cmap.to_rgba(function_values),
                    rstride=1, cstride=1)

    # cbar.set_label('Function Values')

    # Set axis labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(name)

    # Display the plot
    # plt.show()
    plt.savefig(name + '.png')
    return 0


if __name__ == '__main__':
    # test_1d()
    test_3d()
