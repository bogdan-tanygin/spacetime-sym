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
    
    def test_spacetimesym_imports_PhysicalQuantity( self ):
        from spacetime import SpaceTimeGroup

if __name__ == '__main__':
    unittest.main()
