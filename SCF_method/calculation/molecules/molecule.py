from typing import List
import numpy as np


class Molecule:
    """
    Molecule definition object responsible for holding data about the molecule of interest
    """
    def __init__(self,
                 nuclei_positions: List,
                 atomic_numbers: List,
                 number_of_electrons: int):
        """
        :param nuclei_positions: List of coordinates for molecular nuclei
        :param atomic_numbers: List of atomic numbers of molecular nuclei
        :param number_of_electrons: int number of electrons in calculation
        """
        self.nuclei_positions = np.array(nuclei_positions)
        self.atomic_numbers = np.array(atomic_numbers)
        self.number_of_electrons = number_of_electrons
