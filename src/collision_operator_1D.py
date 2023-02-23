import numpy as np
from numpy.polynomial.legendre import leggauss

class CollisionOperator1D():

    def __init__(self, integrator_order:int):
        """

        :param n_q: order of the integration rule for the Gauss Legendre Quadrature
        """
        [quad_pts, quad_weights] = self.qGaussLegendre1D(integrator_order)

        self.w_q = quad_weights
        self.v_q = quad_pts
        self.integrator_order = integrator_order
        self.n_q = quad_weights.size

        self.kernel = 0.5*np.ones(self.n_q ) # placeholder for other kernels: Isotropic kernel

    def get_quad_pts(self)-> np.array:
        return self.v_q

    def get_quad_weights(self)-> np.array:
        return self.w_q

    def qGaussLegendre1D(self,order: int)-> tuple:
        """
        order: order of quadrature
        returns: [quad_pts, quad_weights] : quadrature points and weights (use to get sensor points)
        """
        return leggauss(order)

    def integrate(self,integrand, weights)-> np.array:
        """
        params: weights = quadweights vector (at quadpoints) (dim = nq)
                integrand = integrand vector, evaluated at quadpts (dim = vectorlen x nq)
        returns: integral <integrand>
        """
        return np.dot(integrand, weights)

    def evaluate_Q(self,f_in:np.array)-> np.array:
        """
        (only for an isotropic collision kernel right now)

        :param f_in: function evaluated at quadrature points (use get_quad_pts to get them)
        :return: Q(f) using the specified kollision kernel
        """

        f_out = np.dot(self.kernel*f_in,self.w_q) - f_in


        return f_out