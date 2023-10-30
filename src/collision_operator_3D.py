import numpy as np
from numpy.polynomial.legendre import leggauss


class CollisionOperator3D():

    def __init__(self, integrator_order: int):
        """

        :param n_q: order of the integration rule for the Gauss Legendre Quadrature
        """
        [quad_pts, quad_weights, theta, phi] = self.q_gauss_legendre_3d(integrator_order)
        self.theta = theta
        self.phi = phi
        self.w_q = quad_weights
        self.v_q = quad_pts
        self.integrator_order = integrator_order
        self.n_q = quad_weights.shape

    def get_quad_pts(self) -> np.array:
        return self.v_q

    def get_quad_pts_polar(self) -> np.array:
        return self.theta, self.phi

    def get_quad_weights(self) -> np.array:
        return self.w_q

    def q_gauss_legendre_3d(self, order):

        """
           order: order of quadrature, uses all quadpts... inefficient
           returns: [pts, weights] : quadrature points and weights, dim(pts) = nq x 2
        """

        """Quadrature points for GaussLegendre quadrature. Read from file."""
        mu, leggaussweights = leggauss(order)
        theta = np.arccos(mu)
        phi = [np.pi * (k + 1 / 2) / order for k in range(2 * order)]
        # convert sample points to meshgrid to plot function
        theta, phi = np.meshgrid(theta, phi)
        # Calculate the Cartesian coordinates of each point in the mesh.
        xyz = np.array([np.sin(theta) * np.sin(phi),
                        np.sin(theta) * np.cos(phi),
                        np.cos(theta)])

        w = np.ones(phi.shape) * leggaussweights * np.pi / order

        return xyz, w, theta, phi

    def integrate(self, integrand, weights) -> np.array:
        """
        params: weights = quadweights vector (at quadpoints) (dim = nq)
                integrand = integrand vector, evaluated at quadpts (dim = vectorlen x nq)
        returns: integral <integrand>
        """
        return np.dot(integrand, weights)

    def evaluate_Q(self, f_in: np.array) -> np.array:
        """
        (only for an isotropic collision kernel right now)

        :param f_in: function evaluated at quadrature points (use get_quad_pts to get them)
        :return: Q(f) using the specified kollision kernel
        """
        f_out = np.zeros(self.n_q)
        for q_1_theta in range(self.n_q[0]):
            for q_1_phi in range(self.n_q[1]):
                for q_2_theta in range(self.n_q[0]):
                    for q_2_phi in range(self.n_q[1]):
                        f_out[q_1_theta, q_1_phi] += self.w_q[q_2_theta, q_2_phi] * (
                                f_in[q_2_theta, q_2_phi] - f_in[q_1_theta, q_1_phi]) * 1 / (4 * np.pi)
        return f_out
