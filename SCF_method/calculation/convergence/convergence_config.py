class ConvergenceConfig:

    def __init__(self,
                 max_iteration: int = 5000,
                 averaging: bool = False,
                 delta: float = 1e-6):
        self.max_iteration = max_iteration
        self.averaging = averaging
        self.delta = delta
