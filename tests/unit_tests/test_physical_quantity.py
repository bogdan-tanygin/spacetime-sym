#
# Copyright (C) 2024 Dr. Bogdan Tanygin <info@deeptech.business>
#
# This file is part of spacetime-sym.
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
    
    def test_init_scalar_value_as_tensor( self ):
        tensor = np.identity( 3 ) * self.scalar
        pq = PhysicalQuantity( value = tensor )
        scalar_test = self.scalar
        pq_test = PhysicalQuantity( value = scalar_test )
        scalar_test_2 = (-1) * self.scalar
        pq_test_2 = PhysicalQuantity( value = scalar_test_2 )
        self.assertEqual( ( pq == pq_test ), True )
        self.assertEqual( ( pq_test == pq ), True )
        self.assertEqual( ( pq_test != pq ), False )
        self.assertEqual( ( pq == pq_test_2 ), False )
        self.assertEqual( ( pq != pq_test_2 ), True )

if __name__ == '__main__':
    unittest.main()
