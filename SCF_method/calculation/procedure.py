from SCF_method.calculation.calculation_iterator import SelfConsistentFieldCalculation

from SCF_method.logger import SCF_logger
from SCF_method.matrices.kinetic_energy_matrix import KineticEnergy
from SCF_method.matrices.nuclear_attraction_matrix import NuclearAttraction
from SCF_method.matrices.overlap_matrix import Overlap
from SCF_method.matrices.two_electron_integral_matrix import TwoElectronIntegral


class SelfConsistentFieldProcedure:

    def __init__(self,
                 input_basis,
                 input_molecule,
                 integrator_3D,
                 integrator_6D):

        self.input_basis = input_basis
        self.input_molecule = input_molecule
        self.integrator_3D = integrator_3D
        self.integrator_6D = integrator_6D

    def calculate(self):
        S = Overlap(self.input_basis, self.integrator_3D)
        T = KineticEnergy(self.input_basis, self.integrator_3D)
        V_nuc = NuclearAttraction(self.input_molecule,
                                  self.input_basis,
                                  self.integrator_3D)
        mnls = TwoElectronIntegral(self.input_basis, self.integrator_6D)
        SCF_calc = SelfConsistentFieldCalculation(N=self.input_molecule.number_of_electrons,
                                                  S=S.matrix,
                                                  T=T.matrix,
                                                  V_nuc=V_nuc.matrix,
                                                  mnls=mnls.matrix)
        SCF_iter = iter(SCF_calc)
        SCF_logger.info("Running iterative SCF procedure")
        while SCF_calc.convergence_criterion():
            next(SCF_iter)

        return SCF_calc
