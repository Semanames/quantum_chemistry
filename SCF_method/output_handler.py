from typing import Callable

import numpy as np

from SCF_method.calculation.basis.basis_functions import RootBasis
from SCF_method.calculation.calculation_iterator import SelfConsistentFieldCalculation


class OutputHandlerSCF:
    """
    Handler for SelfConsistentFieldCalculation data transformation to readable and interpretable format
    """
    def __init__(self,
                 SCF_obj: SelfConsistentFieldCalculation,
                 basis: RootBasis):
        """
        :param SCF_obj: SelfConsistentFieldCalculation, state of the SelfConsistentFieldCalculation object state
        :param basis: RootBasis (parent class), object representing basis set used for calculation
        """
        self.SCF_obj = SCF_obj
        self.basis = basis

    def electron_energies(self) -> np.ndarray:
        """
        :return: ndarray, one electron energies for each element of basis set
        """
        return self.SCF_obj.E

    def electron_density_matrix(self) -> np.ndarray:
        """
        :return: ndarray, Electron density matrix array.shape = (len(basis),len(basis))
        """
        return self.SCF_obj.P

    def fock_matrix(self) -> np.ndarray:
        """
        :return: ndarray, Fock matrix array.shape = (len(basis),len(basis))
        """
        return self.SCF_obj.F

    def coeff_matrix(self) -> np.ndarray:
        """
        :return: ndarray, Coefficient matrix array.shape = (len(basis),len(basis))
        """
        return self.SCF_obj.C

    def electron_density(self) -> Callable:
        """
        Calculates electron density as a function of coordinates
        :return: function
        """
        P = self.SCF_obj.P

        def rho(r):
            _rho = np.zeros(r.shape[0])
            for i, base_i in enumerate(self.basis):
                for j, base_j in enumerate(self.basis):
                    _rho += P[i, j]*base_i(r)*base_j(r)
            return _rho

        return rho
