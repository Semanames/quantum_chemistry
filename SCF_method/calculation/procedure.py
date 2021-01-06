from SCF_method.calculation.basis.basis_functions import RootBasis
from SCF_method.calculation.calculation_iterator import SelfConsistentFieldCalculation
from SCF_method.calculation.convergence.convergence_config import ConvergenceConfig
from SCF_method.calculation.integration.integrators import BaseIntegrator
from SCF_method.calculation.matrices.kinetic_energy_matrix import KineticEnergy
from SCF_method.calculation.matrices.nuclear_attraction_matrix import NuclearAttraction
from SCF_method.calculation.matrices.overlap_matrix import Overlap
from SCF_method.calculation.matrices.two_electron_integral_matrix import TwoElectronIntegral
from SCF_method.calculation.molecules.molecule import Molecule

from SCF_method.logger import SCF_logger


class SelfConsistentFieldProcedure:
    """
    Object responsible for running SCF procedure
    """

    def __init__(self,
                 input_basis: RootBasis,
                 input_molecule: Molecule,
                 integrator_3D: BaseIntegrator,
                 integrator_6D :BaseIntegrator,
                 convergence_config: ConvergenceConfig):
        """
        :param basis: RootBasis (parent class), object representing basis set used for calculation
        :param molecule: Molecule: object which defines the distribution of nuclei in space and their atomic numbers
        :param integrator_3D: BaseIntegrator (parent class), integrator object with predefined values of for integration
        :param integrator_6D: BaseIntegrator (parent class), integrator object with predefined values of for integration
        :param convergence_config: ConvergenceConfig, convergence consideration
        """
        self.input_basis = input_basis
        self.input_molecule = input_molecule
        self.integrator_3D = integrator_3D
        self.integrator_6D = integrator_6D
        self.convergence_config = convergence_config

    def calculate(self) -> SelfConsistentFieldCalculation:
        """
        Calculation of molecular integrals and iterative matrix SCF procedure
        :return: SelfConsistentFieldCalculation, state of the converged SCF calculation
        """
        S = Overlap(self.input_basis, self.integrator_3D)
        T = KineticEnergy(self.input_basis, self.integrator_3D)
        V_nuc = NuclearAttraction(self.input_molecule, self.input_basis, self.integrator_3D)
        mnls = TwoElectronIntegral(self.input_basis, self.integrator_6D)
        SCF_calc = SelfConsistentFieldCalculation(
            N=self.input_molecule.number_of_electrons,
            S=S.matrix,
            T=T.matrix,
            V_nuc=V_nuc.matrix,
            mnls=mnls.matrix,
            convergence_config=self.convergence_config
        )
        SCF_iter = iter(SCF_calc)
        SCF_logger.info("Running iterative SCF procedure")
        while SCF_calc.convergence_criterion():
            next(SCF_iter)

        return SCF_calc
