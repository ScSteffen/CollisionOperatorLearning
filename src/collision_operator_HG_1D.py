import numpy as np

from .collision_operator_1D import CollisionOperator1D


class CollisionOperatorHG1D(CollisionOperator1D):
    anisotropy_param: float  # Anisotropy parameter (between -1 and 1).

    def __init__(self, integrator_order: int, anisotropy_param: float):
        super(CollisionOperatorHG1D, self).__init__(integrator_order=integrator_order)

        self.anisotropy_param = anisotropy_param
        # self.kernel = self.henyey_greenstein_kernel_1d_slab(self.anisotropy_param)

        # t = sum(self.kernel)
        # print("here")

    def set_anisotropy(self, anisotropy_param):
        if -1 > anisotropy_param or 1 < anisotropy_param:
            print("Error, HG kernel not defined for g outside [-1.1]")
            exit(1)
        self.anisotropy_param = anisotropy_param
        return 0

    @staticmethod
    def henyey_greenstein_kernel_1d_slab(g, cos_theta_v, cos_theta_v_prime):
        """
        Computes the anisotropic 1D slab geometry collision kernel.

        Parameters:
            g (float): Anisotropy parameter (between -1 and 1).
            theta_v (float): Pre-collision angle in radians.
            theta_v_prime (float): Post-collision angle in radians.

        Returns:
            kernel (float): Collision kernel value.
        """
        if not (-1 <= g <= 1):
            raise ValueError("The anisotropy parameter 'g' must be in the range [-1, 1].")

        # Compute the Henyey-Greenstein scattering kernel for the given angles
        kernel = (1 - g ** 2) / (1 + g ** 2 - 2 * g * cos_theta_v * cos_theta_v_prime) ** (3 / 2)

        if kernel < 0:
            print("error")
        return kernel

    def evaluate_Q(self, f_in: np.array) -> np.array:
        """
        (only for an isotropic collision kernel right now)

        :param f_in: function evaluated at quadrature points (use get_quad_pts to get them)
        :return: Q(f) using the specified kollision kernel
        """

        f_out = np.zeros(self.n_q)
        for q_1 in range(self.n_q):
            for q_2 in range(self.n_q):
                f_out[q_1] += (self.w_q[q_2] * (f_in[q_2] - f_in[q_1])
                               * self.henyey_greenstein_kernel_1d_slab(self.anisotropy_param, self.v_q[q_1],
                                                                       self.v_q[q_2]))

        return f_out
