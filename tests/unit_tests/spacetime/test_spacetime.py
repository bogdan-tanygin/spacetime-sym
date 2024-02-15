#
# Copyright (C) 2024 Dr. Bogdan Tanygin <info@tanygin-holding.com>
#
# This file is part of Spacetime-sym.
#
# Spacetime-sym is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Spacetime-sym is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
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
