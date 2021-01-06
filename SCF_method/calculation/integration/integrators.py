from abc import ABC, abstractmethod
from typing import List, Callable

import numpy as np


class BaseIntegrator(ABC):
    """
    Parent class of other possible types of integrators
    Integrator has one necessary method "integrate"
    """

    @abstractmethod
    def integrate(self, func):
        """
        Method used for integration accordingly to input config definition
        :param func: function to integrate
        :return: integration result : float
        """
        pass


class MonteCarloIntegrator(BaseIntegrator):
    """
    Monte Carlo methods are often used in calculation of multi dimensional integral
    This class enables vectorized calculation of multidimensional integrals, although
    we need to defined finite ranges of the integration
    """

    def __init__(self,
                 n_samples: int,
                 boundaries: List,
                 dimensions: int):
        """

        :param n_samples: number of samples used for calculating average value
        :param boundaries: range of the integration in domain of multidimensional cube
        :param dimensions: dimension of the domain
        """
        self.n_samples = n_samples
        self.upper_bound = boundaries[1]
        self.lower_bound = boundaries[0]
        self.dimensions = dimensions
        self.samples = np.random.uniform(boundaries[1], boundaries[0], size=(n_samples, dimensions))

    def integrate(self, func: Callable) -> float:
        """
        Method is calculation np.mean value of the vectorized output for the input function
        multiplied by the domain
        :param func: vectorized funcion to integrate
        :return: float
        """
        domain = (self.upper_bound - self.lower_bound)**self.dimensions
        integration_output = np.mean(func(self.samples))*domain
        return integration_output
