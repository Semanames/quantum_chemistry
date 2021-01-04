from typing import List
import numpy as np


class Molecule:

    def __init__(self,
                 nuclei_positions: List,
                 atomic_numbers: List,
                 number_of_electrons: int):
        self.nuclei_positions = np.array(nuclei_positions)
        self.atomic_numbers = np.array(atomic_numbers)
        self.number_of_electrons = number_of_electrons
