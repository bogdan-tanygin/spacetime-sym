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
from unittest.mock import Mock, patch
from spacetime.physicalquantity import PhysicalQuantity
import numpy as np

class TestPhysicalQuantity( unittest.TestCase ):

    def setUp( self ):
        self.parities = {"C":0, "P":0, "T":0}
#TODO
    def test_reassign_cpt_parities( self ):
        pq = Mock( spec = PhysicalQuantity )
        pq.parities = self.parities
        cpt_test = [
            {"C":-1, "P":-1, "T":-1},
            {"C":-1, "P": 0, "T": 0},
            {"T": 1, "C": 1, "P": 0}
        ]
        keys = cpt_test[0].keys()
        for i in range(len(cpt_test)):
            pq.parities = cpt_test[i]
            for key in keys:
                self.assertEqual( pq.parities[key], cpt_test[i] )

if __name__ == '__main__':
    unittest.main()
