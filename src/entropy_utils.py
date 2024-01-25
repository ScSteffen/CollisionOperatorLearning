"""
Script with functions for quadratures, moment basis for 1D-3D spatial dimensions
Author:  Steffen Schotthöfer
Date: 16.03.21
"""

import numpy as np
import scipy
import scipy.optimize as opt
from numpy.polynomial.legendre import leggauss


class EntropyTools:
    """
    Same functions implemented in the sobolev Network.
    Also uses Tensorflow
    """
    spatial_dimension: int
    poly_degree: int
    nq: int
    input_dim: int

    opti_u: np.ndarray
    moment_basis_np: np.ndarray

    # @brief: Regularization Parameter for regularized entropy. =0 means non regularized
    regularization_gamma: np.ndarray
    regularization_gamma_np: float

    def __init__(self, quad_order=10, polynomial_degree=1, spatial_dimension=1, gamma=0, basis="spherical_harmonics"):
        """
        Class to compute the 1D entropy closure up to degree N
        input: N  = degree of polynomial basis
        """

        # Create quadrature and momentBasis. Currently only for 1D problems
        self.poly_degree = polynomial_degree
        self.spatial_dimension = spatial_dimension
        if spatial_dimension == 1 and basis == "monomial":
            self.nq = quad_order
            [quad_pts, quad_weights] = q_gauss_legendre1_d(quad_order)  # order = nq
            m_basis = compute_monomial_basis1_d(quad_pts, self.poly_degree)  # dims = (N x nq)
        if spatial_dimension == 2 and basis == "monomial":
            [quad_pts, quad_weights, _, _] = q_gauss_legendre2_d(quad_order)  # dims = nq
            self.nq = quad_weights.size  # is not 10 * polyDegree
            m_basis = compute_monomial_basis2_d(quad_pts, self.poly_degree)  # dims = (N x nq)
        elif spatial_dimension == 3 and basis == "spherical_harmonics":
            [quad_pts, quad_weights, mu, phi] = q_gauss_legendre3_d(quad_order)  # dims = nq
            self.nq = quad_weights.size  # is not 20 * polyDegree
            m_basis = compute_spherical_harmonics(mu, phi, self.poly_degree)
        elif spatial_dimension == 2 and basis == "spherical_harmonics":
            [quad_pts, quad_weights, mu, phi] = q_gauss_legendre2_d(quad_order)  # dims = nq #
            self.nq = quad_weights.size  # is not 20 * polyDegree
            m_basis = compute_spherical_harmonics_2D(mu, phi, self.poly_degree)
        elif spatial_dimension == 1 and basis == "spherical_harmonics":
            [quad_pts, quad_weights] = q_gauss_legendre1_d(quad_order)  # dims = nq #
            self.nq = quad_weights.size  # is not 20 * polyDegree
            m_basis = compute_spherical_harmonics_1D(quad_pts, self.poly_degree)
        else:
            print("spatial dimension not yet supported for sobolev wrapper")
            exit()
        self.quad_pts = quad_pts
        self.quad_weights_np = quad_weights

        self.input_dim = m_basis.shape[0]
        self.moment_basis_np = m_basis

        self.regularization_gamma_np = gamma

    def reconstruct_alpha(self, alpha: np.ndarray) -> np.ndarray:
        f_quad = np.exp(np.einsum("i,iq->q", alpha, self.moment_basis_np[1:, :]))
        alpha_0 = - (np.log(self.moment_basis_np[0, 0]) + np.log(np.einsum('q,q->', f_quad, self.quad_weights_np)
                                                                 )) / self.moment_basis_np[0, 0]
        return np.append(alpha_0, alpha)

    def reconstruct_u(self, alpha: np.ndarray) -> np.ndarray:

        f_quad = np.exp(np.einsum("i,iq->q", alpha, self.moment_basis_np))
        recons_u = np.einsum('q,q,iq->i', f_quad, self.quad_weights_np, self.moment_basis_np)  # f*w*m
        t3 = self.regularization_gamma_np * alpha
        t3[0] = 0.0
        return recons_u + t3

    def reconstruct_f(self, alpha: np.ndarray):
        return np.exp(np.einsum("i,iq->q", alpha, self.moment_basis_np))

    def compute_u(self, f: np.ndarray) -> np.ndarray:
        return np.einsum('q,q,iq->i', f, self.quad_weights_np, self.moment_basis_np)  # <fm>

    def compute_h_dual(self, u: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        f_quad = np.exp(np.einsum("i,iq->q", alpha, self.moment_basis_np))
        t1 = np.einsum('q,q->', f_quad, self.quad_weights_np)  # <f>
        t2 = np.einsum("i,i->", alpha, u)
        t3 = self.regularization_gamma_np / 2.0 * np.linalg.norm(alpha[1:]) ** 2
        return -1 * (t1 - t2 + t3)

    def compute_h_primal(self, f: np.ndarray) -> float:
        """
        brief: computes the entropy functional h on u and alpha

        returns h = <f*ln(f)-f>
        """
        # Currently only for maxwell Boltzmann entropy
        eta_f = f * np.log(f) - f
        return np.einsum("q,q->", eta_f, self.quad_weights_np)

    def integrate_f(self, f: np.ndarray):

        return np.einsum("q,q->", f, self.quad_weights_np)

    def minimize_entropy(self, u: np.ndarray, alpha_start: np.ndarray) -> np.ndarray:
        """
        brief: computes the minimal entropy at u
        input: u = dims (n,1)
           start =  start_valu of alpha
        """
        self.opti_u = np.copy(u)
        opt_result = opt.minimize(fun=self.opti_entropy, x0=alpha_start, jac=self.opti_entropy_prime,
                                  hess=self.opti_entropy_prime2, tol=1e-6)

        if not opt_result.success:
            print("Moment")
            print(u)
            print("Optimization unsuccessfull!")

        return opt_result.x

    def opti_entropy(self, alpha: np.ndarray) -> np.ndarray:

        f_quad = np.exp(np.einsum("i,iq->q", alpha, self.moment_basis_np))
        t1 = np.einsum('q,q->', f_quad, self.quad_weights_np)  # f*w
        t2 = np.einsum("i,i->", alpha, self.opti_u)
        t3 = self.regularization_gamma_np / 2.0 * np.linalg.norm(alpha[1:]) ** 2
        return t1 - t2 + t3

    def opti_entropy_prime(self, alpha: np.ndarray) -> np.ndarray:

        f_quad = np.exp(np.einsum("i,iq->q", alpha, self.moment_basis_np))
        t1 = np.einsum('q,q,iq->i', f_quad, self.quad_weights_np, self.moment_basis_np)  # f*w*m
        t3 = self.regularization_gamma_np * alpha
        t3[0] = 0.0
        return t1 - self.opti_u + t3

    def opti_entropy_prime2(self, alpha: np.ndarray) -> np.ndarray:
        """
         brief: returns the 2nd derivative negative entropy functional with fixed u
         nS = batchSize
         N = basisSize
         nq = number of quadPts

         input: alpha, dims = (1 x N)
                u, dims = (1 x N)
         used members: m    , dims = (N x nq)
                     w    , dims = nq

         returns h =  <mxm*eta_*(alpha*m)>
        """
        # Currently only for maxwell Boltzmann entropy
        f_quad = np.exp(np.einsum("i,iq->q", alpha, self.moment_basis_np))

        hess = np.einsum("q,q,iq,jq-> ij", f_quad, self.quad_weights_np, self.moment_basis_np, self.moment_basis_np)

        t3 = self.regularization_gamma_np * np.identity(self.input_dim)
        t3[0, 0] = 0
        return -hess + t3

    def rejection_sampling(self, n: int, sigma: float, max_alpha: float = 10):
        """
        :param n: number of samples
        :param sigma: condition number threshold (determines anistotropy)
        :return: f= kinetic densities, u=oments, alpha=multipliers, h=entropy values
        """
        alpha = np.zeros(shape=(self.input_dim, n))
        h = np.zeros(shape=(n,))
        u = np.zeros(shape=(self.input_dim, n))
        f = np.zeros(shape=(self.nq, n))
        max_alpha = 2
        for i in range(n):
            condition_ok = False
            while not condition_ok:
                beta = np.random.normal(loc=0.0, scale=max_alpha / 3., size=self.input_dim - 1)

                alpha_full = self.reconstruct_alpha(beta)
                H = self.opti_entropy_prime2(alpha=alpha_full)
                c = np.linalg.cond(H, 'fro')
                # print(c)
                if c < sigma:
                    condition_ok = True
                    alpha[:, i] = alpha_full
                    #print('Density sampled: ' + str(i) + '/' + str(n))
            u[:, i] = self.reconstruct_u(alpha=alpha[:, i])
            h[i] = self.compute_h_dual(u=u[:, i], alpha=alpha[:, i])
            f[:, i] = self.reconstruct_f(alpha=alpha[:, i])  # need to reshape into tensor format
        return f, h, u, alpha


# Spherical Harmonics Basis


def q_gauss_legendre1_d(order: int):
    """
    order: order of quadrature
    returns: [mu, weights] : quadrature points and weights
    """
    return leggauss(order)


def q_gauss_legendre2_d(Qorder):
    """
       order: order of quadrature, uses all quadpts... inefficient
       returns: [pts, weights] : quadrature points and weights, dim(pts) = nq x 2
    """

    def computequadpoints(order):
        """Quadrature points for GaussLegendre quadrature. Read from file."""
        """
        mu in  [-1,0]
        phi in [0,2*pi]
        """
        mu, _ = leggauss(order)
        phi = [np.pi * (k + 1 / 2) / order for k in range(2 * order)]
        xy = np.zeros((order * order, 2))
        count = 0
        mu_arr = np.zeros((order * order,))
        phi_arr = np.zeros((order * order,))
        for i in range(int(order / 2.0)):
            for j in range(2 * order):
                mu_arr[count] = mu[i]
                phi_arr[count] = phi[j]
                mui = mu[i]
                phij = phi[j]
                xy[count, 0] = np.sqrt(1 - mui ** 2) * np.cos(phij)
                xy[count, 1] = np.sqrt(1 - mui ** 2) * np.sin(phij)
                # xyz[count, 2] = mui
                count += 1

        return xy, mu_arr, phi_arr

    def computequadweights(order):
        """Quadrature weights for GaussLegendre quadrature. Read from file."""
        _, leggaussweights = leggauss(order)
        w = np.zeros(order * order)
        count = 0
        for i in range(int(order / 2.0)):
            for j in range(2 * order):
                w[count] = 2.0 * np.pi / order * leggaussweights[i]
                count += 1
        return w

    pts, mu, phi = computequadpoints(Qorder)
    weights = computequadweights(Qorder)

    return [pts, weights, mu, phi]


def q_gauss_legendre3_d(Qorder):
    """
       order: order of quadrature, uses all quadpts... inefficient
       returns: [pts, weights] : quadrature points and weights, dim(pts) = nq x 2
    """

    def computequadpoints(order):
        """Quadrature points for GaussLegendre quadrature. Read from file."""
        mu, _ = leggauss(order)
        phi = [np.pi * (k + 1 / 2) / order for k in range(2 * order)]
        xyz = np.zeros((2 * order * order, 3))
        count = 0
        mu_arr = np.zeros((2 * order * order,))
        phi_arr = np.zeros((2 * order * order,))

        for i in range(int(order)):
            for j in range(2 * order):
                mu_arr[count] = mu[i]
                phi_arr[count] = phi[j]

                xyz[count, 0] = np.sqrt(1 - mu[i] ** 2) * np.cos(phi[j])
                xyz[count, 1] = np.sqrt(1 - mu[i] ** 2) * np.sin(phi[j])
                xyz[count, 2] = mu[i]
                count += 1

        return xyz, mu_arr, phi_arr

    def computequadweights(order):
        """Quadrature weights for GaussLegendre quadrature. Read from file."""
        _, leggaussweights = leggauss(order)
        w = np.zeros(2 * order * order)
        count = 0
        for i in range(int(order)):
            for j in range(2 * order):
                w[count] = np.pi / order * leggaussweights[i]
                count += 1
        return w

    pts, mu, phi = computequadpoints(Qorder)
    weights = computequadweights(Qorder)

    return [pts, weights, mu, phi]


# Basis Computation
def compute_monomial_basis1_d(quadPts, polyDegree):
    """
    params: quadPts = quadrature points to evaluate
            polyDegree = maximum degree of the basis
    return: monomial basis evaluated at quadrature points
    """
    basisLen = get_basis_size(polyDegree, 1)
    nq = quadPts.shape[0]
    monomialBasis = np.zeros((basisLen, nq))

    for idx_quad in range(0, nq):
        for idx_degree in range(0, polyDegree + 1):
            monomialBasis[idx_degree, idx_quad] = np.power(quadPts[idx_quad], idx_degree)
    return monomialBasis


def compute_monomial_basis2_d(quadPts, polyDegree):
    """
    brief: Same basis function ordering as in KiT-RT code
    params: quadPts = quadrature points to evaluate
            polyDegree = maximum degree of the basis
    return: monomial basis evaluated at quadrature points
    """
    basisLen = get_basis_size(polyDegree, 2)
    nq = quadPts.shape[0]
    monomialBasis = np.zeros((basisLen, nq))

    for idx_quad in range(0, nq):
        # Hardcoded for degree 1
        # monomialBasis[0, idx_quad] = 1.0
        # monomialBasis[1, idx_quad] = quadPts[idx_quad, 0]
        # monomialBasis[2, idx_quad] = quadPts[idx_quad, 1]

        omega_x = quadPts[idx_quad, 0]
        omega_y = quadPts[idx_quad, 1]

        idx_vector = 0
        for idx_degree in range(0, polyDegree + 1):
            for a in range(0, idx_degree + 1):
                b = idx_degree - a
                monomialBasis[idx_vector, idx_quad] = np.power(omega_x, a) * np.power(omega_y, b)
                idx_vector += 1

    return monomialBasis


def get_basis_size(polyDegree, spatialDim):
    """
    params: polyDegree = maximum Degree of the basis
            spatialDIm = spatial dimension of the basis
    returns: basis size
    """

    basisLen = 0

    for idx_degree in range(0, polyDegree + 1):
        basisLen += int(
            get_curr_degree_size(idx_degree, spatialDim))

    return basisLen


def get_curr_degree_size(currDegree, spatialDim):
    """
    Computes the number of polynomials of the current spatial dimension
    """
    return np.math.factorial(currDegree + spatialDim - 1) / (
            np.math.factorial(currDegree) * np.math.factorial(spatialDim - 1))


# --- spherical harmonics
def compute_spherical_harmonics_1D(mu: np.ndarray, degree: int) -> np.ndarray:
    # Tested against KiT-RT for degree 0-4 at 6th June 2023
    # assemble spherical harmonics
    n_system = degree + 1
    sh_basis_scipy = np.zeros((n_system, len(mu)))

    for i in range(len(mu)):
        for l in range(0, degree + 1):
            Y = scipy.special.sph_harm(0, l, 0, np.arccos(mu[i]), out=None)
            Y = Y.real

            sh_basis_scipy[l, i] = Y

        # sh_basis_scipy[0, i] = np.sqrt(1 / (2 * np.pi))

        # test against python implementation
    return sh_basis_scipy


def compute_spherical_harmonics_2D(mu: np.ndarray, phi: np.ndarray, degree: int) -> np.ndarray:
    # Tested against KiT-RT for degree 0-4 at 6th June 2023
    # assemble spherical harmonics
    input_dim_dict_2D: dict = {1: 3, 2: 6, 3: 10, 4: 15, 5: 21}

    n_system = input_dim_dict_2D[degree]
    sh_basis_scipy = np.zeros((n_system, len(mu)))

    for i in range(len(mu)):
        count = 0
        for l in range(0, degree + 1):
            for k in range(-l, l + 1):

                if (k + l) % 2 == 0:
                    if k < 0:
                        Y = scipy.special.sph_harm(np.abs(k), l, phi[i], np.arccos(mu[i]), out=None)
                        Y = np.sqrt(2) * (-1) ** (k + l) * Y.imag
                    if k > 0:
                        Y = scipy.special.sph_harm(k, l, phi[i], np.arccos(mu[i]), out=None)
                        Y = np.sqrt(2) * (-1) ** (k + l) * Y.real
                    if k == 0:
                        Y = scipy.special.sph_harm(k, l, phi[i], np.arccos(mu[i]), out=None)
                        Y = Y.real

                    sh_basis_scipy[count, i] = Y
                    count += 1

        # sh_basis_scipy[0, i] = np.sqrt(1 / (2 * np.pi))

        # test against python implementation
    return sh_basis_scipy


def compute_spherical_harmonics(mu: np.ndarray, phi: np.ndarray, degree: int) -> np.ndarray:
    # assemble spherical harmonics
    n_system = 2 * degree + degree ** 2 + 1
    sh_basis = np.zeros((n_system, len(mu)))
    idx_sys = 0
    for l in range(degree + 1):
        for k in range(-l, l + 1):
            idx_quad = 0
            for mui, phij in zip(mu, phi):
                Yvals = scipy.special.sph_harm(abs(k), l, phij, np.arccos(mui))
                if k < 0:
                    Yvals = np.sqrt(2) * Yvals.imag  # * (-1) ** (k + 1)
                elif k > 0:
                    Yvals = np.sqrt(2) * Yvals.real  # * (-1) ** (k + 1)
                elif k == 0:
                    Yvals = Yvals.real
                sh_basis[idx_sys, idx_quad] = Yvals
                idx_quad += 1
            idx_sys += 1

    return sh_basis
