import numpy as np
from numpy.polynomial.legendre import leggauss


class CollisionOperator1D():

    def __init__(self, integrator_order: int):
        """

        :param n_q: order of the integration rule for the Gauss Legendre Quadrature
        """
        [quad_pts, quad_weights] = self.qGaussLegendre1D(integrator_order)

        self.w_q = quad_weights
        self.v_q = quad_pts
        self.integrator_order = integrator_order
        self.n_q = quad_weights.size

    def get_quad_pts(self) -> np.array:
        return self.v_q

    def get_quad_weights(self) -> np.array:
        return self.w_q

    def qGaussLegendre1D(self, order: int) -> tuple:
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

    def evaluate_Q(self, f_in: np.array, g: float) -> np.array:
        """
        (only for an isotropic collision kernel right now)

        :param f_in: function evaluated at quadrature points (use get_quad_pts to get them)
        :return: Q(f) using the specified kollision kernel
        """
        kernel = self.calc_h_g_scattering_kernel(self.v_q, self.v_q, g=g)
        #        f_out = kernel.dot(f_in * self.w_q) - kernel.dot(self.w_q) * f_in

        t2 = 0
        for i in range(len(f_in)):
            t2 += self.henyey_greenstein_kernel(v=0.3, v_prime=self.v_q[i], g=g) * self.w_q[i]

        t = kernel.dot(self.w_q)
        f_out = np.zeros(len(f_in))
        for i in range(len(f_in)):
            for j in range(len(f_in)):
                f_out[i] += kernel[i, j] * f_in[j] * self.w_q[j]
            f_out[i] -= np.dot()

        return f_out

    @staticmethod
    def calc_h_g_scattering_kernel(v: np.ndarray, v_prime: np.ndarray, g: float) -> np.ndarray:
        """
        :param v: velocity variable
        :param v_prime: integration velocity variable
        :param g: velocity variable
        :return: scattering kerenel for each
        """
        return (1 - g ** 2) / (4 * np.pi * np.sqrt(1 - 2 * g * np.outer(v, v_prime) + g ** 2) ** 3)

    # def henyey_greenstein_kernel(self, mu):
    @staticmethod
    def henyey_greenstein_kernel(v: float, v_prime: float, g: float) -> np.ndarray:
        """
        :param v: velocity variable
        :param v_prime: integration velocity variable
        :param g: velocity variable
        :return: scattering kerenel for each
        """
        return (1 - g ** 2) / (np.sqrt(2) * np.sqrt(1 - 2 * g * v * v_prime + g ** 2))
