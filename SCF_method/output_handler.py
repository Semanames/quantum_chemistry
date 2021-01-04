import numpy as np

from SCF_method.calculation.calculation_iterator import SelfConsistentFieldCalculation


class OutputHandlerSCF:

    def __init__(self, SCF_obj: SelfConsistentFieldCalculation, basis):
        self.SCF_obj = SCF_obj
        self.basis = basis

    def electron_energies(self):
        return self.SCF_obj.E

    def electron_density_matrix(self):
        return self.SCF_obj.P

    def fock_matrix(self):
        return self.SCF_obj.F

    def coeff_matrix(self):
        return self.SCF_obj.C

    def electron_density(self):
        P = self.SCF_obj.P

        def rho(r):
            _rho = np.zeros(r.shape[0])
            for i, base_i in enumerate(self.basis):
                for j, base_j in enumerate(self.basis):
                    _rho += P[i, j]*base_i(r)*base_j(r)
            return _rho

        return rho
