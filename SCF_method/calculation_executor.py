import json

from SCF_method.basis.basis_mapping import BASIS_TYPE_MAPPING
from SCF_method.calculation.procedure import SelfConsistentFieldProcedure
from SCF_method.integration.integrator_mapping import INTEGRATOR_TYPE_MAPPING
from SCF_method.logger import SCF_logger
from SCF_method.molecules.molecule import Molecule


class ExecutorSCF:

    def __init__(self, input_file):
        SCF_logger.info("Preparing input data")
        input_dict = json.load(input_file)
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

    def run_calculation(self):

        SCF_procedure = SelfConsistentFieldProcedure(input_basis=self.basis,
                                                     input_molecule=self.molecule,
                                                     integrator_3D=self.integrator_3D,
                                                     integrator_6D=self.integrator_6D)

        SCF_obj = SCF_procedure.calculate()
        SCF_logger.info("SCF procedure succesfull")
        return SCF_obj
