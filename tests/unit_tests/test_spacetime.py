#
# Copyright (C) 2024 Dr. Bogdan Tanygin <info@deeptech.business>
#
# This file is part of spacetime-sym.
#

import unittest
from unittest.mock import Mock
import warnings 

class TestSpacetimesymTopLevelClasses( unittest.TestCase ):

    def test_spacetimesym_imports_PhysicalQuantity( self ):
        from spacetime import PhysicalQuantity
    
    def test_spacetimesym_imports_SG( self ):
        from spacetime import SymmetryGroup
    
    def test_spacetimesym_imports_SymmetryOperation( self ):
        from spacetime import SymmetryOperation

    def test_spacetimesym_imports_SymmetryOperationO3( self ):
        from spacetime import SymmetryOperationO3

    def test_spacetimesym_imports_SymmetryOperationSO3( self ):
        from spacetime import SymmetryOperationSO3
    
    def test_spacetimesym_imports_LimitingSymmetryOperationO3( self ):
        from spacetime import LimitingSymmetryOperationO3
    
    def test_spacetimesym_imports_LimitingSymmetryOperationSO3( self ):
        from spacetime import LimitingSymmetryOperationSO3
    
    def test_spacetimesym_imports_LimitingSymmetryGroupScalar( self ):
        from spacetime import LimitingSymmetryGroupScalar

if __name__ == '__main__':
    unittest.main()
