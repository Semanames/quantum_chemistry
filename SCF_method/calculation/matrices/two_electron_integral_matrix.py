import numpy as np

from SCF_method.logger import SCF_logger


class TwoElectronIntegral:

    def __init__(self, basis, integrator):
        self.basis = basis
        self.integrator = integrator
        SCF_logger.info("Calculating mnls - two electron integral matrix")
        self.matrix = self._calculate_self()

    def _integrand(self, base_i, base_j, base_k, base_l):
        V_electron = self.electron_coulomb_potential

        def electron_potential(r):
            return base_i(r[:, :3]) * base_j(r[:, :3]) * V_electron(r) * base_k(r[:, 3:]) * base_l(r[:, 3:])

        return electron_potential

    @staticmethod
    def electron_coulomb_potential(r):
        return 1 / np.sqrt(np.sum((r[:, :3] - r[:, 3:]) ** 2, axis=1))

    def _calculate_self(self):
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
        #np.save('calculation\matrices\precalculated\mnls', mnls)
        return mnls
