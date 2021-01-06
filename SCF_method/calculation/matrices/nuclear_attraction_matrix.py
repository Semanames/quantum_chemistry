import numpy as np

from SCF_method.logger import SCF_logger


class NuclearAttraction:

    def __init__(self, molecule, basis, integrator):
        self.molecule = molecule
        self.basis = basis
        self.integrator = integrator
        SCF_logger.info("Calculating V_nuc - nuclear attraction matrix")
        self.matrix = self._calculate_self()

    def _integrand(self, base_i, base_j):
        V_nuclear = self.nuclear_coulomb_potential

        def nuclear_potential(r):
            return base_i(r)*V_nuclear(r)*base_j(r)

        return nuclear_potential

    def nuclear_coulomb_potential(self, r):
        return np.sum([-self.molecule.atomic_numbers[i] /
                       np.sqrt(np.sum((r - self.molecule.nuclei_positions[i]) ** 2, axis=1))
                       for i in range(len(self.molecule.atomic_numbers))], axis=0)

    def _calculate_self(self):
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
        #np.savetxt('calculation\matrices\precalculated\V', V_nuc)
        return V_nuc
