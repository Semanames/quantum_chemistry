from SCF_method.calculation.integration.integrators import MonteCarloIntegrator
"""
This module is used for mapping of integrator classes 
used for further calculation defined in input json file
There might be more types in the future
"""
INTEGRATOR_TYPE_MAPPING = {
    "MC": MonteCarloIntegrator
}