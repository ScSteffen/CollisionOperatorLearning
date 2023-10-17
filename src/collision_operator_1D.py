import numpy as np
from numpy.polynomial.legendre import leggauss


class CollisionOperator1D():

    def __init__(self, integrator_order: int):
        """

        :param n_q: order of the integration rule for the Gauss Legendre Quadrature
        """
        [quad_pts, quad_weights] = self.q_gauss_legendre_1d(integrator_order)

        self.w_q = quad_weights
        self.v_q = quad_pts
        self.integrator_order = integrator_order
        self.n_q = quad_weights.size

    def get_quad_pts(self) -> np.array:
        return self.v_q

    def get_quad_weights(self) -> np.array:
        return self.w_q

    def q_gauss_legendre_1d(self, order: int) -> tuple:
        """
        order: order of quadrature
        returns: [quad_pts, quad_weights] : quadrature points and weights (use to get sensor points)
        """
        return leggauss(order)

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
        for q_1 in range(self.n_q):
            for q_2 in range(self.n_q):
                f_out[q_1] += self.w_q[q_2] * (f_in[q_2] - f_in[q_1]) * 1 / (4 * np.pi)

        return f_out
