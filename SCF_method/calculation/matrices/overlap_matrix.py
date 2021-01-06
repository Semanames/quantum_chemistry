import numpy as np

from SCF_method.logger import SCF_logger


class Overlap:

    def __init__(self, basis, integrator):

        self.basis = basis
        self.integrator = integrator
        SCF_logger.info("Calculating S - orbital overlap matrix")
        self.matrix = self._calculate_self()

    @staticmethod
    def _integrand(base_i, base_j):
        def overlap_term(r):
            return base_i(r)*base_j(r)
        return overlap_term

    def _calculate_self(self):
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
        #np.savetxt('calculation\matrices\precalculated\S', S)
        return S
