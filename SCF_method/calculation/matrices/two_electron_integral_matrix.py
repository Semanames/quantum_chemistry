from typing import Callable

import numpy as np

from SCF_method.calculation.basis.basis_functions import RootBasis
from SCF_method.calculation.integration.integrators import BaseIntegrator
from SCF_method.logger import SCF_logger


class TwoElectronIntegral:
    """
    This class represents two electron Coulombic interaction matrix term in the SCF calculation
    In this repo is usually called mnls
    For given set of basis function it will calculate the corresponding matrix
    Calculation of this matrix is computationally most expensive due to its large dimension
    """
    def __init__(self,
                 basis: RootBasis,
                 integrator: BaseIntegrator):
        """
        :param basis: RootBasis (parent class), object representing basis set used for calculation
        :param integrator: BaseIntegrator (parent class), integrator object with predefined values of for integration
        """
        self.basis = basis
        self.integrator = integrator
        SCF_logger.info("Calculating mnls - two electron integral matrix")
        self.matrix = self._calculate_self()

    def _integrand(self, base_i: Callable, base_j: Callable, base_k: Callable, base_l: Callable) -> Callable:
        """
        Returns function term for integration: base_i(r1) * base_j(r1) * V_electron * base_k(r2) * base_l(r2)
        :param base_i: function from the self.basis
        :param base_j: function from the self.basis
        :param base_k: function from the self.basis
        :param base_l: function from the self.basis
        :return: function to integrate
        """
        V_electron = self.electron_coulomb_potential

        def electron_potential(r: np.ndarray):
            """
            :param r: ndarray, r.shape(N,6) because it covers coordinates for both electrons in calculation
            :return: function
            """
            return base_i(r[:, :3]) * base_j(r[:, :3]) * V_electron(r) * base_k(r[:, 3:]) * base_l(r[:, 3:])

        return electron_potential

    @staticmethod
    def electron_coulomb_potential(r: np.ndarray):
        return 1 / np.sqrt(np.sum((r[:, :3] - r[:, 3:]) ** 2, axis=1))

    def _calculate_self(self) ->np.ndarray:
        """
        Calculation of two electron interaction matrix itself
        :return: ndarray where array.shape = (len(basis), len(basis), len(basis), len(basis))
        """
        basis_length = len(self.basis)
        mnls = np.zeros([basis_length, basis_length, basis_length, basis_length])
        for i, base_i in enumerate(self.basis):
            for j, base_j in enumerate(self.basis):
                for k, base_k in enumerate(self.basis):
                    for l, base_l in enumerate(self.basis):
                        if not mnls[i, j, k, l]:
                            mnls[i, j, k, l] = self.integrator.integrate(self._integrand(base_i,
                                                                                         base_j,
                                                                                         base_k,
                                                                                         base_l))
                            mnls[j, i, k, l] = mnls[i, j, k, l]
                            mnls[i, j, l, k] = mnls[i, j, k, l]
                            mnls[j, i, l, k] = mnls[i, j, k, l]
                            mnls[k, l, i, j] = mnls[i, j, k, l]
                            mnls[l, k, i, j] = mnls[i, j, k, l]
                            mnls[k, l, j, i] = mnls[i, j, k, l]
                            mnls[l, k, j, i] = mnls[i, j, k, l]
        return mnls
