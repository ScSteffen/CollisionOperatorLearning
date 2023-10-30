import numpy as np

from .collision_operator_3D import CollisionOperator3D


class CollisionOperatorHG3D(CollisionOperator3D):
    anisotropy_param: float  # Anisotropy parameter (between -1 and 1).

    def __init__(self, integrator_order: int, anisotropy_param: float):
        super(CollisionOperatorHG3D, self).__init__(integrator_order=integrator_order)

        self.anisotropy_param = anisotropy_param
        # self.kernel = self.henyey_greenstein_kernel_1d_slab(self.anisotropy_param)

        # t = sum(self.kernel)

    def set_anisotropy(self, anisotropy_param):
        if -1 > anisotropy_param or 1 < anisotropy_param:
            print("Error, HG kernel not defined for g outside [-1.1]")
            exit(1)
        self.anisotropy_param = anisotropy_param
        return 0

    @staticmethod
    def henyey_greenstein_kernel_3d(g, cos_theta):
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
        kernel = (1 - g ** 2) / (1 + g ** 2 - 2 * g * cos_theta) ** (3 / 2) * 1 / (4 * np.pi)

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
        for q_1_theta in range(self.n_q[0]):
            for q_1_phi in range(self.n_q[1]):
                for q_2_theta in range(self.n_q[0]):
                    for q_2_phi in range(self.n_q[1]):
                        omega = np.inner(self.v_q[:, q_1_theta, q_1_phi], self.v_q[:, q_2_theta, q_2_phi])
                        f_out[q_1_theta, q_1_phi] += self.w_q[q_2_theta, q_2_phi] * (
                                f_in[q_2_theta, q_2_phi] - f_in[q_1_theta, q_1_phi]) * self.henyey_greenstein_kernel_3d(
                            self.anisotropy_param, omega)

        return f_out
