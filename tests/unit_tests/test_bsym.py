import unittest
from unittest.mock import Mock
import warnings 

class TestBsymTopLevelClasses( unittest.TestCase ):

    def test_bsym_imports_SymmetryOperation( self ):
        from spacetime import SymmetryOperation

    def test_bsym_imports_SymmetryGroup( self ):
        from spacetime import SymmetryGroup

    def test_bsym_imports_SpaceGroup( self ):
        from spacetime import SpaceGroup

    def test_bsym_imports_PointGroup( self ):
        from spacetime import PointGroup

    def test_bsym_imports_Configuration( self ):
        from spacetime import Configuration

    def test_bsym_imports_ConfigurationSpace( self ):
        from spacetime import ConfigurationSpace

    def test_bsym_imports_CoordinateConfigSpace( self ):
        from spacetime import CoordinateConfigSpace

    def test_bsym_imports_ColourOperation( self ):
        from spacetime import ColourOperation

if __name__ == '__main__':
    unittest.main()
