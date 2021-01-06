from SCF_method.calculation.basis.basis_functions import GaussianBasis
"""
This module is used for mapping of basis classes 
used for further calculation defined in input json file
There might be more types in the future
"""

BASIS_TYPE_MAPPING = {
    "gaussian": GaussianBasis
}
