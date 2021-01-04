import numpy as np
from abc import ABC, abstractmethod


class RootBasis(ABC):
    def __init__(self, nuclei_positions, normalization_factors, *args, **kwargs):
        self.nuclei_positions = nuclei_positions
        self.normalization_factors = normalization_factors
        self.args = args
        self.kwargs = kwargs
        self.basis_set = self._create_basis_set(nuclei_position=nuclei_positions,
                                                normalization_factors=normalization_factors,
                                                *args, **kwargs)

    def __getitem__(self, item):
        return self.basis_set[item]

    def __len__(self):
        return len(self.basis_set)

    def __iter__(self):
        for base in self.basis_set:
            yield base

    @abstractmethod
    def _create_basis_set(self, *args, **kwargs):
        pass


class GaussianBasis(RootBasis):
    @staticmethod
    def gaussian_base_element(alpha: float,
                              r0: np.ndarray,
                              norm: float):
        def gauss(r):
            return norm*np.exp(-alpha*np.sum((r-r0)**2, axis=1))
        return gauss

    def _create_basis_set(self,  alphas, nuclei_position, normalization_factors):
        basis_set = []
        for i, R in enumerate(nuclei_position):
            for j, alpha in enumerate(alphas):
                basis_set.append(self.gaussian_base_element(alpha, R, normalization_factors[i][j]))

        return basis_set

    def renormalize(self, overlap_matrix_coeffs):
        self.normalization_factors = overlap_matrix_coeffs.reshape(self.normalization_factors.shape)
        self.basis_set = self._create_basis_set(normalization_factors=self.normalization_factors,
                                                nuclei_position=self.nuclei_positions,
                                                *self.args, **self.kwargs)


