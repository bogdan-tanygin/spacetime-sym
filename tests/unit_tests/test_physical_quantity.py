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
from spacetime.physical_quantity import PhysicalQuantity
import numpy as np

class TestPhysicalQuantity( unittest.TestCase ):

    def setUp( self ):
        self.dich = {"C":1, "P":1, "T":-1}
        self.scalar = 42.42e+3
        self.vector = [-1/2, 0.4, 637e-8]
        self.label = 'velocity'

    def test_reassign_cpt_dis( self ):
        pq = PhysicalQuantity()
        pq.dich = self.dich
        cpt_test = [
            {"C":-1, "P":-1, "T":-1},
            {"C":-1, "P": 1, "T": -1},
            {"T": 1, "C": 1, "P": 1}
        ]
        keys = cpt_test[0].keys()
        for i in range(len(cpt_test)):
            pq.dich = cpt_test[i]
            for key in keys:
                self.assertEqual( pq.dich[key], cpt_test[i][key] )
    
    def test_label( self ):
        pq = PhysicalQuantity(label = self.label)
        self.assertEqual( pq.label, self.label )
        test_label = 'temperature'
        pq.label = test_label
        self.assertEqual( pq.label, test_label )
    
    def test_add_new_dis_symmetry( self ):
        pq = PhysicalQuantity()
        # default dis
        dich_test = self.dich
        pq.dich = dich_test
        # mass inversion dis
        pq.dich["M"] = -1
        # make sure that the CPT symmetry stay the same
        keys = dich_test.keys()
        for key in keys:
            self.assertEqual( pq.dich[key], dich_test[key] )
        # test mass inversion dis
        self.assertEqual( pq.dich["M"], -1 )
    
    def test_assigning_dich_with_wrong_value( self ):
        pq = PhysicalQuantity()
        # new incorrect dis value
        dich_test = {"X":-2}
        with self.assertRaises( ValueError ):
            # new incorrect dich setting
            pq.dich = dich_test

    def test_init_dich_with_wrong_value( self ):
        # incorrect dich value
        dich_test = {"X":-2}
        with self.assertRaises( ValueError ):
            # object init with incorrect dich
            pq = PhysicalQuantity(dich = dich_test)
        dich_test = {2:-2}
        with self.assertRaises( TypeError ):
            # object init with incorrect dich
            pq = PhysicalQuantity(dich = dich_test)

    def test_reassigning_dis_with_wrong_value( self ):
        pq = PhysicalQuantity()
        # default dis
        pq.dich = self.dich
        # new incorrect dis value
        dich_test = {"X":-2}
        with self.assertRaises( ValueError ):
            # extra incorrect dis setting
            pq.dich = dich_test

    def test_reassign_scalar_value( self ):
        pq = PhysicalQuantity()
        pq.value = self.scalar
        scalar_test = - np.sqrt(3.0)
        pq.value = scalar_test
        self.assertEqual( pq.value, scalar_test)

    def test_reassign_vector_value( self ):
        pq = PhysicalQuantity()
        pq.value = self.vector
        vector_test =[np.sqrt(3.0), np.exp(-2.3), 0]
        pq.value = vector_test
        for i in range(len(vector_test)):
            self.assertEqual( pq.value[i], vector_test[i])

    def test_init_scalar_value( self ):
        pq = PhysicalQuantity( value = self.scalar )
        scalar_test = self.scalar
        self.assertEqual( pq.value, scalar_test)
    
    def test_init_vector_value( self ):
        pq = PhysicalQuantity( value = self.vector )
        vector_test = self.vector
        for i in range(len(vector_test)):
            self.assertEqual( pq.value[i], vector_test[i])
    
if __name__ == '__main__':
    unittest.main()
