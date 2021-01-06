import json
from typing import Dict

from SCF_method.calculation.basis.basis_mapping import BASIS_TYPE_MAPPING
from SCF_method.calculation.calculation_iterator import SelfConsistentFieldCalculation
from SCF_method.calculation.convergence.convergence_config import ConvergenceConfig
from SCF_method.calculation.integration.integrator_mapping import INTEGRATOR_TYPE_MAPPING
from SCF_method.calculation.molecules.molecule import Molecule
from SCF_method.calculation.procedure import SelfConsistentFieldProcedure
from SCF_method.logger import SCF_logger


class ExecutorSCF:
    """
    Executor is responsible for input data extraction to a custom objects
    Also enables execution of SCF calculation
    """

    def __init__(self, input_dict: Dict):
        """
        :param input_dict: Dict: dictionary of transformed data from input json
        """
        SCF_logger.info("Preparing input data")
        SCF_logger.info("Initializing molecule")
        self.molecule = Molecule(**input_dict['molecule_definition'])
        SCF_logger.info("Initializing basis set")
        self.basis = BASIS_TYPE_MAPPING[input_dict['basis']['type']](**input_dict['basis']['params'])
        SCF_logger.info("Initializing integrators")
        self.integrator_3D = INTEGRATOR_TYPE_MAPPING[input_dict['integration_config']['type']](
            n_samples=input_dict['integration_config']['n_samples'],
            boundaries=input_dict['integration_config']['boundaries'],
            dimensions=3
        )
        self.integrator_6D = INTEGRATOR_TYPE_MAPPING[input_dict['integration_config']['type']](
            n_samples=input_dict['integration_config']['n_samples'],
            boundaries=input_dict['integration_config']['boundaries'],
            dimensions=6
        )
        if "convergence_config" in input_dict.keys():
            self.convergence_config = ConvergenceConfig(**input_dict["convergence_config"])
        else:
            self.convergence_config = ConvergenceConfig()

    def run_calculation(self) -> SelfConsistentFieldCalculation:

        SCF_procedure = SelfConsistentFieldProcedure(input_basis=self.basis,
                                                     input_molecule=self.molecule,
                                                     integrator_3D=self.integrator_3D,
                                                     integrator_6D=self.integrator_6D,
                                                     convergence_config=self.convergence_config)

        SCF_obj = SCF_procedure.calculate()
        SCF_logger.info("SCF procedure succesfull")
        return SCF_obj
