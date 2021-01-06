import numpy as np

from SCF_method.logger import SCF_logger


class SelfConsistentFieldCalculation:

    def __init__(self, N, S, T, V_nuc, mnls, covergence_config, P=None):
        self.N = N
        self.S = S
        self.T = T
        self.V_nuc = V_nuc
        self.mnls = mnls
        self.convergence_config = covergence_config
        self.P = P if P else np.identity(T.shape[0])
        s, U = np.linalg.eig(S)
        self.X = U@np.diag(s**(-0.5))@U.T
        self.G = self.calculate_g_matrix()
        self.F = self.calculate_fock_matrix()
        C, E = self.calculate_c_matrix()
        self.C = C
        self.E = E
        self.P_new = self.calculate_electron_density_matrix()

    def __iter__(self):
        self.iteration = 1
        return self

    def __next__(self):
        self.P = self.P_new
        self.G = self.calculate_g_matrix()
        self.F = self.calculate_fock_matrix()
        self.C, self.E = self.calculate_c_matrix()
        self.P_new = self.calculate_electron_density_matrix()
        self.iteration += 1

    def calculate_g_matrix(self):
        G = np.zeros_like(self.T)
        for i in range(G.shape[0]):
            for j in range(G.shape[1]):
                G += self.P[i, j] * (self.mnls[:, :, j, i] - 0.5 * self.mnls[:, i, j, :])
        return G

    def calculate_fock_matrix(self):
        F = self.T + self.V_nuc + self.G
        return F

    def calculate_c_matrix(self):
        F_prime = self.X.T @ self.F @ self.X
        E, C_prime = np.linalg.eig(F_prime)
        C = self.X @ C_prime
        return C, E

    def calculate_electron_density_matrix(self):
        P = np.zeros_like(self.T)
        for a in range(self.N // 2):
            for mu in range(self.G.shape[0]):
                for nu in range(self.G.shape[1]):
                    P[mu, nu] += 2 * self.C[mu, a] * self.C[nu, a]
        return P

    def convergence_criterion(self):
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
