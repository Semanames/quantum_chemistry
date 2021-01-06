from typing import Callable

import numpy as np

from SCF_method.calculation.basis.basis_functions import RootBasis
from SCF_method.calculation.integration.integrators import BaseIntegrator
from SCF_method.calculation.molecules.molecule import Molecule
from SCF_method.logger import SCF_logger


class NuclearAttraction:
    """
    This class represents nuclear potential energy matrix term in the SCF calculation
    For given set of basis function it will calculate the corresponding matrix
    To calculate this term we need to define Coulomb potential energy from given molecule definition
    """
    def __init__(self,
                 molecule: Molecule,
                 basis: RootBasis,
                 integrator: BaseIntegrator):
        """
        :param molecule: Molecule: object which defines the distribution of nuclei in space and their atomic numbers
        :param basis: RootBasis (parent class), object representing basis set used for calculation
        :param integrator: BaseIntegrator (parent class), integrator object with predefined values of for integration
        """
        self.molecule = molecule
        self.basis = basis
        self.integrator = integrator
        SCF_logger.info("Calculating V_nuc - nuclear attraction matrix")
        self.matrix = self._calculate_self()

    def _integrand(self, base_i: Callable, base_j: Callable) -> Callable:
        """
        Returns function term for integration: base_i * V_nuclear * base_j
        :param base_i: function from the self.basis
        :param base_j: function from the self.basis
        :return: function to integrate
        """
        V_nuclear = self.nuclear_coulomb_potential

        def nuclear_potential(r: np.ndarray):
            return base_i(r)*V_nuclear(r)*base_j(r)
            # TODO refactor the terms with np.conj in case when basis is not real

        return nuclear_potential

    def nuclear_coulomb_potential(self, r: np.ndarray):
        """
        Calculation of the nuclear Coulombic potential from given molecule:
        potential: SUMa (- Za / abs( r - Ra ))
        :param r:
        :return:
        """
        return np.sum([-self.molecule.atomic_numbers[i] /
                       np.sqrt(np.sum((r - self.molecule.nuclei_positions[i]) ** 2, axis=1))
                       for i in range(len(self.molecule.atomic_numbers))], axis=0)

    def _calculate_self(self) -> np.ndarray:
        """
        Calculation of nuclear Coulombic energy matrix itself
        :return: ndarray where array.shape = (len(basis),len(basis))
        """
        basis_length = len(self.basis)
        V_nuc = np.zeros([basis_length, basis_length])
        for i, base_i in enumerate(self.basis):
            for j in range(i, basis_length):
                v_ij = self.integrator.integrate(self._integrand(base_i, self.basis[j]))
                if i == j:
                    V_nuc[i, j] = v_ij
                else:
                    V_nuc[i, j] = v_ij
                    V_nuc[j, i] = v_ij
        return V_nuc
