class ConvergenceConfig:
    """
    Convergence config object is used in the SCF iteration process
    to cope with the iteration itself
    """

    def __init__(self,
                 max_iteration: int = 5000,
                 averaging: bool = False,
                 delta: float = 1e-6):
        """
        :param max_iteration: int, maximum number of iterations where the iteration should stop
                              (divergent or oscillating cases)

        :param averaging: bool, averaging process to speed up the convergence
        :param delta: float coefficient to consider whether the procedure has converged
        """
        self.max_iteration = max_iteration
        self.averaging = averaging
        self.delta = delta
