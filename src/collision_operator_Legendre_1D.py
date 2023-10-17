import numpy as np

from .collision_operator_1D import CollisionOperator1D


class CollisionOperatorLegendre1D(CollisionOperator1D):
    legendre_order: int  # number of used Legendre functions

    def __init__(self, integrator_order: int, , legendre_order: int):
        super(CollisionOperatorLegendre1D, self).__init__(integrator_order=integrator_order)

        self.legendre_order = legendre_order
        #self.kernel = self.legendre_kernel_1d_slab(g, cos_theta_v, cos_theta_v_prime)

    def legendre_kernel_1d_slab(self, cos_theta_v, cos_theta_v_prime):
        """
        Creates a Henyey-Greenstein collision kernel in 1D slab geometry.
        """
        #TODO

        # Discretize the angle range from 0 to pi (180 degrees)
        # transform velocity to angle
        angle = np.arccos(self.v_q)

        # Calculate the Legendre polynomials up to order L for all angles
        legendre_values = np.zeros((L + 1, n_theta))
        for l in range(L + 1):
            legendre_values[l] = legendre(l)(np.cos(angles))

        # Compute the Legendre scattering kernel for each angle
        kernel = np.zeros(n_theta)
        for l, c_l in enumerate(coefficients):
            kernel += c_l * legendre_values[l]

        return angles, kernel

        return kernel

    def evaluate_Q(self, f_in: np.array) -> np.array:
        """
        (only for an isotropic collision kernel right now)

        :param f_in: function evaluated at quadrature points (use get_quad_pts to get them)
        :return: Q(f) using the specified kollision kernel
        """

        f_out = np.zeros(self.n_q)
        t = 0
        for q_1 in range(self.n_q):
            f_out[q_1] = 0.0
            for q_2 in range(self.n_q):
                f_out[q_1] += self.w_q[q_2] * self.kernel[q_1, q_2]  # * (f_in[q_2] - f_in[q_1])

            t += f_out[q_1] * self.w_q[q_1]
        return f_out
