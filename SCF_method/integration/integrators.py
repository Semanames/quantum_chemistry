from abc import ABC, abstractmethod

import numpy as np


class BaseIntegrator(ABC):

    @abstractmethod
    def integrate(self, func):
        pass


class MonteCarloIntegrator(BaseIntegrator):

    def __init__(self, n_samples, boundaries, dimensions):
        self.n_samples = n_samples
        self.upper_bound = boundaries[1]
        self.lower_bound = boundaries[0]
        self.dimensions = dimensions
        self.samples = np.random.uniform(boundaries[1], boundaries[0], size=(n_samples, dimensions))

    def integrate(self, func):
        domain = (self.upper_bound - self.lower_bound)**self.dimensions
        integration_output = np.mean(func(self.samples))*domain
        return integration_output
