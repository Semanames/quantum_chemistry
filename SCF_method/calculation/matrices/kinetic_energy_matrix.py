from typing import Callable

import numpy as np

from SCF_method.calculation.basis.basis_functions import RootBasis
from SCF_method.calculation.integration.integrators import BaseIntegrator
from SCF_method.logger import SCF_logger


class KineticEnergy:
    """
    This class represents kinetic energy matrix term in the SCF calculation
    For given set of basis function it will calculate the corresponding matrix
    To calculate this term we need to define numerical value of laplacian (kinetic energy operator)
    """

    dx, dy, dz = 1e-5, 1e-5, 1e-5
    # these values are used for numeral calculation of derivative of function
    # probably should be moved to some config file in future

    def __init__(self,
                 basis: RootBasis,
                 integrator: BaseIntegrator):
        """
        :param basis: RootBasis (parent class), object representing basis set used for calculation
        :param integrator: BaseIntegrator (parent class), integrator object with predefined values of for integration
        """

        self.basis = basis
        self.integrator = integrator
        SCF_logger.info("Calculating T - kinetic energy matrix")
        self.matrix = self._calculate_self()

    @classmethod
    def laplacian(cls, f: Callable) -> Callable:
        """
        Numerical calculation of laplacian of function
        :param f: function of r, where r.shape = (N,3)
        :return: function ( laplace of the input function)
        """
        def partial_dif(r: np.ndarray):
            dx_vec = np.array([1, 0, 0]) * cls.dx
            dy_vec = np.array([0, 1, 0]) * cls.dy
            dz_vec = np.array([0, 0, 1]) * cls.dz

            laplace = (((f(r + dx_vec) - 2 * f(r) + f(r - dx_vec)) / cls.dx ** 2) +
                       ((f(r + dy_vec) - 2 * f(r) + f(r - dy_vec)) / cls.dy ** 2) +
                       ((f(r + dz_vec) - 2 * f(r) + f(r - dz_vec)) / cls.dz ** 2))
            return laplace

        return partial_dif

    def _integrand(self, base_i: Callable, base_j: Callable) -> Callable:
        """
        Returns function term for integration: base_i*laplace(base_j)
        :param base_i: function from the self.basis
        :param base_j: function from the self.basis
        :return: function to integrate
        """
        laplace_base_j = self.laplacian(base_j)

        def kinetic_term(r):
            return -0.5 * base_i(r) * laplace_base_j(r)
            # TODO refactor the terms with np.conj in case when basis is not real

        return kinetic_term

    def _calculate_self(self) -> np.ndarray:
        """
        Calculation of kinetic energy matrix itself
        :return: ndarray where array.shape = (len(basis),len(basis))
        """
        basis_length = len(self.basis)
        T = np.zeros([basis_length, basis_length])
        for i, base_i in enumerate(self.basis):
            for j in range(i, basis_length):
                t_ij = self.integrator.integrate(self._integrand(base_i, self.basis[j]))
                if i == j:
                    T[i, j] = t_ij
                else:
                    # TODO refactor the terms with np.conj in case when basis is not real, matrix have to be Hermitian
                    T[i, j] = t_ij
                    T[j, i] = t_ij
        return T
