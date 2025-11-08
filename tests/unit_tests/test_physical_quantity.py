#
# Copyright (C) 2024-2025 Dr. Bogdan Tanygin <info@deeptech.business>
#
# This file is part of spacetime-sym.
#

import unittest
from spacetime.physical_quantity import PhysicalQuantity
import numpy as np

class TestPhysicalQuantity( unittest.TestCase ):

    def setUp( self ):
        self.dich = {"C":1, "P":1, "T":-1}
        self.scalar = 42.42e+3 * np.sqrt( 7 )
        self.vector = np.array( [ -1/2, 0.4, self.scalar ] )
        self.label = 'velocity'
        #TODO correct atol globally - make it stricter
        self.atol = 1e-6
        self.rtol = 1e-6
        # random tensor
        self.tensor = np.array( [ [ -1, np.sqrt( np.pi ), 1.3454675432e6 ],
                                    [ 743.3566, 0, -1.456 ],
                                    [ 1.4567, -877865, np.exp( -4 ) ] ] )

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
    
    def test_add_new_dich_symmetry( self ):
        pq = PhysicalQuantity()
        # default dich
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
        self.assertEqual( pq.value == scalar_test , True )

    def test_reassign_vector_value( self ):
        pq = PhysicalQuantity()
        pq.value = self.vector
        vector_test = np.array( [np.sqrt(3.0), np.exp(-2.3), 0] )
        pq.value = vector_test
        np.testing.assert_allclose( pq.value, vector_test, rtol = self.rtol )
        self.assertEqual( np.array_equiv( pq.value, vector_test ), True )

    def test_init_scalar_value( self ):
        pq = PhysicalQuantity( value = self.scalar )
        scalar_test = self.scalar
        self.assertEqual( pq.value, scalar_test)
        self.assertEqual( pq.value == scalar_test, True)
    
    def test_init_vector_value( self ):
        pq = PhysicalQuantity( value = self.vector )
        vector_test = self.vector
        np.testing.assert_allclose( pq.value, vector_test, rtol = self.rtol )
        self.assertEqual( np.array_equiv( pq.value, vector_test ), True )
    
    def test_init_bidirector_value( self ):
        pq_1 = PhysicalQuantity( value = self.vector, bidirector = True )
        vector_test = - np.array( self.vector )
        pq_2 = PhysicalQuantity( value = vector_test, bidirector = True )
        self.assertEqual( ( pq_1 == pq_2 ), True )
        with self.assertRaises( ValueError ):
            pq_1.bidirector = 2
        with self.assertRaises( ValueError ):
            pq_3 = PhysicalQuantity( value = self.vector, bidirector = 'a' )
    
    def test__eq__( self ):
        pq_1 = PhysicalQuantity( value = self.scalar )
        pq_2 = PhysicalQuantity( value = self.scalar )
        self.assertEqual( pq_1 == pq_2, True )
        self.assertEqual( pq_2 == pq_1, True )
        self.assertEqual( pq_1 == pq_1, True )
        pq_1 = PhysicalQuantity( value = self.vector )
        pq_2 = PhysicalQuantity( value = self.vector )
        self.assertEqual( pq_1 == pq_2, True )
        self.assertEqual( pq_2 == pq_1, True )
        self.assertEqual( pq_1 == pq_1, True )
        pq_1 = PhysicalQuantity( value = self.tensor )
        pq_2 = PhysicalQuantity( value = self.tensor )
        self.assertEqual( pq_1 == pq_2, True )
        self.assertEqual( pq_2 != pq_1, False )
        self.assertEqual( pq_1 == pq_1, True )

    def test__eq__bidirectors( self ):
        pq_1 = PhysicalQuantity( value = self.scalar, bidirector = True )
        pq_2 = PhysicalQuantity( value = self.scalar, bidirector = True )
        self.assertEqual( pq_1 == pq_2, True )
        self.assertEqual( pq_2 == pq_1, True )
        self.assertEqual( pq_1 == pq_1, True )
        pq_1 = PhysicalQuantity( value = self.vector, bidirector = True )
        pq_2 = PhysicalQuantity( value = self.vector, bidirector = True )
        self.assertEqual( pq_1 == pq_2, True )
        self.assertEqual( pq_2 == pq_1, True )
        self.assertEqual( pq_1 == pq_1, True )
        pq_1 = PhysicalQuantity( value = self.tensor, bidirector = True )
        pq_2 = PhysicalQuantity( value = self.tensor, bidirector = True )
        self.assertEqual( pq_1 == pq_2, True )
        self.assertEqual( pq_2 == pq_1, True )
        self.assertEqual( pq_1 == pq_1, True )
        pq_1 = PhysicalQuantity( value = - self.scalar, bidirector = True )
        pq_2 = PhysicalQuantity( value = self.scalar, bidirector = True )
        self.assertEqual( pq_1 == pq_2, True )
        self.assertEqual( pq_2 != pq_1, False )
        self.assertEqual( pq_1 == pq_1, True )
        pq_1 = PhysicalQuantity( value = self.vector, bidirector = True )
        pq_2 = PhysicalQuantity( value = - self.vector, bidirector = True )
        self.assertEqual( pq_1 == pq_2, True )
        self.assertEqual( pq_2 == pq_1, True )
        self.assertEqual( pq_1 == pq_1, True )
        pq_1 = PhysicalQuantity( value = self.tensor, bidirector = True )
        pq_2 = PhysicalQuantity( value = - self.tensor, bidirector = True )
        self.assertEqual( pq_1 == pq_2, True )
        self.assertEqual( pq_2 == pq_1, True )
        self.assertEqual( pq_1 == pq_1, True )
        pq_1 = PhysicalQuantity( value = - 2 * self.scalar, bidirector = True )
        pq_2 = PhysicalQuantity( value = self.scalar, bidirector = True )
        self.assertEqual( pq_1 == pq_2, False )
        self.assertEqual( pq_2 == pq_1, False )
        self.assertEqual( pq_1 == pq_1, True )
        pq_1 = PhysicalQuantity( value = self.vector, bidirector = True )
        pq_2 = PhysicalQuantity( value = - 2 * self.vector, bidirector = True )
        self.assertEqual( pq_1 == pq_2, False )
        self.assertEqual( pq_2 == pq_1, False )
        self.assertEqual( pq_1 == pq_1, True )
        pq_1 = PhysicalQuantity( value = 2 * self.tensor, bidirector = True )
        pq_2 = PhysicalQuantity( value = - self.tensor, bidirector = True )
        self.assertEqual( pq_1 == pq_2, False )
        self.assertEqual( pq_2 != pq_1, True )
        self.assertEqual( pq_1 == pq_1, True )
        # incompatible types tests
        pq_1 = PhysicalQuantity( value = - 2 * self.scalar, bidirector = True )
        pq_2 = PhysicalQuantity( value = - 2 * self.vector, bidirector = True )
        with self.assertRaises( TypeError ):
            res = ( pq_1 == pq_2 )
        with self.assertRaises( TypeError ):
            res = ( pq_1 != pq_2 )
        pq_1 = PhysicalQuantity( value = - 2 * self.scalar, bidirector = True )
        pq_2 = PhysicalQuantity( value = - 2 * self.scalar, bidirector = False )
        with self.assertRaises( TypeError ):
            res = ( pq_1 == pq_2 )
        pq_1 = PhysicalQuantity( value = - 2 * self.tensor, bidirector = True )
        pq_2 = PhysicalQuantity( value = - 2 * self.vector, bidirector = True )
        with self.assertRaises( TypeError ):
            res = ( pq_1 == pq_2 )

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
