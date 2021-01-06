from typing import List, Callable

import numpy as np
from abc import ABC, abstractmethod


class RootBasis(ABC):
    """
    Parent class of other possible types of basis
    Basis is also object with defined getitem, lenght and iter method
    for convenient handling
    """
    def __init__(self,
                 nuclei_positions: List,
                 normalization_factors: List,
                 *args, **kwargs):
        """

        :param nuclei_positions: list of lists (coordinates) for system of nuclei
        :param normalization_factors: list of lists (coefficients) for normalization of basis functions
        :param args: arbitrary arguments specific to certain basis
        :param kwargs: arbitrary keyword specific to certain basis
        """
        self.nuclei_positions = np.array(nuclei_positions)
        self.normalization_factors = np.array(normalization_factors)
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
        """
        This method should be implemented in all children basis classes
        Method will create list of basis functions, defining the set of basis functions
        used in next calculations
        :param args:
        :param kwargs:
        :return:
        """
        pass

    @abstractmethod
    def renormalize(self, overlap_matrix_coeffs: np.ndarray):
        """
        Normalization coefficients should be redefined in the way that the overlap matrix should
        have 1. on the diagonal terms. This method is intended for this purpose
        :param overlap_matrix_coeffs: ndarray of normalization coefficients
        :return: None
        """
        pass


class GaussianBasis(RootBasis):
    """
    Gaussian basis is quite common basis set for quantum chemistry calculations
    Definition: N*exp( alpha * (r - R)**2 )
    """
    @staticmethod
    def gaussian_base_element(alpha: float, r0: np.ndarray, norm: float) -> Callable:
        """
        Arguments defined in the definition of the basis
        This method is nested and should return function dependent only on spatial coordinates
        :param alpha: float coefficient from the definition
        :param r0: ndarray of coordinates for specific nucleus
        :param norm: float normalization coefficient
        :return: function of r, r.shape = (N,3), where N is arbitrary integer
        """
        def gauss(r):
            return norm*np.exp(-alpha*np.sum((r-r0)**2, axis=1))
        return gauss

    def _create_basis_set(self,  alphas: List, nuclei_position: np.ndarray, normalization_factors: List) -> List:
        basis_set = []
        for i, R in enumerate(nuclei_position):
            for j, alpha in enumerate(alphas):
                basis_set.append(self.gaussian_base_element(alpha, R, normalization_factors[i][j]))

        return basis_set

    def renormalize(self, overlap_matrix_coeffs: np.ndarray):
        """
        Normalization coefficients should be redefined in the way that the overlap matrix should
        have 1. on the diagonal terms. This method is intended for this purpose
        :param overlap_matrix_coeffs: ndarray of normalization coefficients
        :return: None
        """
        self.normalization_factors = overlap_matrix_coeffs.reshape(self.normalization_factors.shape)
        self.basis_set = self._create_basis_set(normalization_factors=self.normalization_factors,
                                                nuclei_position=self.nuclei_positions,
                                                *self.args, **self.kwargs)


