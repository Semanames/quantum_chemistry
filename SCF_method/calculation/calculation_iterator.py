from typing import Tuple

import numpy as np

from SCF_method.calculation.convergence.convergence_config import ConvergenceConfig
from SCF_method.logger import SCF_logger


class SelfConsistentFieldCalculation:
    """
    Self consistent field iterator object is responsible for iterative calculation of SCF procedure
    In each step it is defined by the state of attributes (matrices)
    It is initialized with the matrices:
        Overlap matrix,
        Kinetic energy matrix
        Nuclear potential matrix
        Two electron interaction matrix
        Electron density matrix (guess)
    a part of initialization is number of electrons in calculated system and convergence config for
    convergence consideration
    This procedure is described in : Attila Szabo, Neil S. Ostlund; Modern Quantum Chemistry; page 146
    """

    def __init__(self,
                 N: int,
                 S: np.ndarray,
                 T: np.ndarray,
                 V_nuc: np.ndarray,
                 mnls: np.ndarray,
                 convergence_config: ConvergenceConfig,
                 P=None):
        """
        Initialization of Iterator object corresponds with 12. step procedure defined in :
        Attila Szabo, Neil S. Ostlund; Modern Quantum Chemistry; page 146
        :param N: int, number of electrons of a system
        :param S: ndarray, Overlap matrix
        :param T: ndarray, Kinetic energy matrix
        :param V_nuc: ndarray, Nuclear potential matrix
        :param mnls: ndarray, Two electron interaction matrix
        :param convergence_config: ConvergenceConfig, convergence consideration
        :param P: ndarray, initial guess of electron density matrix
        """
        self.N = N
        self.S = S  # Step 2. Molecular integrals
        self.T = T  # Step 2. Molecular integrals
        self.V_nuc = V_nuc  # Step 2. Molecular integrals
        self.mnls = mnls  # Step 2. Molecular integrals
        self.convergence_config = convergence_config
        self.P = P if P else np.identity(T.shape[0])  # Step 4. Density matrix initial guess
        s, U = np.linalg.eig(S)  # Step 3. Diagonalization of overlap matrix
        self.X = U@np.diag(s**(-0.5))@U.T  # Step 3. Diagonalization of overlap matrix
        self.G = self.calculate_g_matrix()  # Step 5. Calculation of G matrix
        self.F = self.calculate_fock_matrix()  # Step 6. Calculation of Fock matrix
        C, E = self.calculate_c_matrix()  # Steps. 7., 8., 9.
        self.C = C
        self.E = E
        self.P_new = self.calculate_electron_density_matrix()  # Step 10. Formulate a new electron density matrix

    def __iter__(self):
        """
        Initialization of iteration process
        (Creation of an Iterator object)
        self.iteration = 1 corresponds to the fact that the first iteration was run at initialization
        :return:
        """
        self.iteration = 1
        return self

    def __next__(self):
        """
        Calling "next" mutates the object itself defining the new state of iteration
        :return:
        """
        self.P = self.P_new
        self.G = self.calculate_g_matrix()
        self.F = self.calculate_fock_matrix()
        self.C, self.E = self.calculate_c_matrix()
        self.P_new = self.calculate_electron_density_matrix()
        self.iteration += 1

    def calculate_g_matrix(self) -> np.ndarray:
        G = np.zeros_like(self.T)
        for i in range(G.shape[0]):
            for j in range(G.shape[1]):
                G += self.P[i, j] * (self.mnls[:, :, j, i] - 0.5 * self.mnls[:, i, j, :])
        return G

    def calculate_fock_matrix(self) -> np.ndarray:
        F = self.T + self.V_nuc + self.G
        return F

    def calculate_c_matrix(self) -> Tuple:
        F_prime = self.X.T @ self.F @ self.X  # Step 7. Calculation of transformed Fock Matrix
        E, C_prime = np.linalg.eig(F_prime)  # Step 8. Diagonalization of transformed Fock Matrix
        C = self.X @ C_prime  # Step 9. Calculation of coefficient matrix
        return C, E

    def calculate_electron_density_matrix(self) -> np.ndarray:
        P = np.zeros_like(self.T)
        for a in range(self.N // 2):
            for mu in range(self.G.shape[0]):
                for nu in range(self.G.shape[1]):
                    P[mu, nu] += 2 * self.C[mu, a] * self.C[nu, a]
        return P

    def convergence_criterion(self) -> bool:
        """
        Convergence consideration according to the literature:
        Attila Szabo, Neil S. Ostlund; Modern Quantum Chemistry; page 149
        :return: Bool: logic which stops the iteration
        """
        epsilon = np.sqrt(np.sum((self.P - self.P_new) ** 2) / self.P_new.shape[0] ** 2)
        SCF_logger.info(f"Convergence factor is {epsilon}")
        if epsilon <= self.convergence_config.delta:
            return False
        if self.convergence_config.averaging:
            self.P_new = (self.P_new + self.P)/2
        if self.iteration == self.convergence_config.max_iteration:
            SCF_logger.info(f"SCF procedure reached {self.iteration} iterations: Iteration stopped")
            return False
        else:
            return True
