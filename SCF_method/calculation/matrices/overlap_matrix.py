from typing import Callable

import numpy as np

from SCF_method.calculation.basis.basis_functions import RootBasis
from SCF_method.calculation.integration.integrators import BaseIntegrator
from SCF_method.logger import SCF_logger


class Overlap:
    """
    This class represents nuclear orbital overlap matrix S term in the SCF calculation
    For given set of basis function it will calculate the corresponding matrix
    CALCULATION OF THIS MATRIX MUTATES THE BASIS OBJECT BY NORMALIZATION OF ITS ELEMENTS
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
        SCF_logger.info("Calculating S - orbital overlap matrix")
        self.matrix = self._calculate_self()

    @staticmethod
    def _integrand(base_i: Callable, base_j: Callable) -> Callable:
        """
        Returns function term for integration: base_i * base_j
        :param base_i: function from the self.basis
        :param base_j: function from the self.basis
        :return: function to integrate
        """
        def overlap_term(r: np.ndarray):
            return base_i(r)*base_j(r)
            # TODO refactor the terms with np.conj in case when basis is not real
        return overlap_term

    def _calculate_self(self) -> np.ndarray:
        """
        Calculation of orbital overlap matrix itself
        :return: ndarray where array.shape = (len(basis),len(basis))
        """
        basis_length = len(self.basis)
        S = np.zeros([basis_length, basis_length])
        for i, base_i in enumerate(self.basis):
            for j in range(i, basis_length):
                s_ij = self.integrator.integrate(self._integrand(base_i, self.basis[j]))
                if i == j:
                    S[i, j] = s_ij
                else:
                    S[i, j] = s_ij
                    S[j, i] = s_ij
        norm_coeffs = np.sqrt(np.diag(S))
        SCF_logger.info(f"Basis renormalization with coeffs: {norm_coeffs}")
        self.basis.renormalize(1./norm_coeffs)
        S = S / np.outer(norm_coeffs, norm_coeffs)
        return S
