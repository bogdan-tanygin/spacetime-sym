#
# Copyright (C) 2024 Dr. Bogdan Tanygin <info@deeptech.business>
# Copyright (C) 2015, 2021 Benjamin J. Morgan
#
# This file is part of spacetime-sym.
#

from .version import __version__

from .linear_algebra import is_permutation_matrix, is_square, is_diagonal, set_copy_assignment,\
                            is_scalar, is_scalar_extended
from .symmetry_operation import SymmetryOperation, SymmetryOperationO3, SymmetryOperationSO3
from .symmetry_operation import LimitingSymmetryOperationO3, LimitingSymmetryOperationSO3
from .symmetry_group import SymmetryGroup, LimitingSymmetryGroupScalar
from .physical_quantity import PhysicalQuantity