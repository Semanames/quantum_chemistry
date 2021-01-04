import numpy as np

from SCF_method.logger import SCF_logger


class KineticEnergy:

    dx, dy, dz = 1e-5, 1e-5, 1e-5

    def __init__(self, basis, integrator):

        self.basis = basis
        self.integrator = integrator
        SCF_logger.info("Calculating T - kinetic energy matrix")
        self.matrix = self._calculate_self()

    @classmethod
    def laplacian(cls, f):
        def partial_dif(r):
            dx_vec = np.array([1, 0, 0]) * cls.dx
            dy_vec = np.array([0, 1, 0]) * cls.dy
            dz_vec = np.array([0, 0, 1]) * cls.dz

            laplace = (f(r + dx_vec) - 2 * f(r) + f(r - dx_vec)) / cls.dx ** 2 + \
                      (f(r + dy_vec) - 2 * f(r) + f(r - dy_vec)) / cls.dy ** 2 + \
                      (f(r + dz_vec) - 2 * f(r) + f(r - dz_vec)) / cls.dz ** 2
            return laplace

        return partial_dif

    def _integrand(self, base_i, base_j):
        laplace_base_j = self.laplacian(base_j)

        def kinetic_term(r):
            return -0.5 * base_i(r) * laplace_base_j(r)

        return kinetic_term

    def _calculate_self(self):
        basis_length = len(self.basis)
        T = np.zeros([basis_length, basis_length])
        for i, base_i in enumerate(self.basis):
            for j in range(i, basis_length):
                t_ij = self.integrator.integrate(self._integrand(base_i, self.basis[j]))
                if i == j:
                    T[i, j] = t_ij
                else:
                    T[i, j] = t_ij
                    T[j, i] = t_ij
        return T
