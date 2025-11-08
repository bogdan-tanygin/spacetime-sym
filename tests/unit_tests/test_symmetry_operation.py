#
# Copyright (C) 2024 Dr. Bogdan Tanygin <info@deeptech.business>
# Copyright (C) 2015, 2021 Benjamin J. Morgan
#
# This file is part of spacetime-sym.
#

import unittest
import numpy as np
from math import pi, sqrt
from numpy.linalg import det
from copy import deepcopy
from scipy.spatial.transform import Rotation
from spacetime import SymmetryOperation, SymmetryOperationO3, LimitingSymmetryOperationO3
from spacetime import SymmetryOperationSO3, LimitingSymmetryOperationSO3, PhysicalQuantity
from unittest.mock import patch
import io
from spacetime.linear_algebra import is_square, is_permutation_matrix

class SymmetryOperationTestCase( unittest.TestCase ):
    """Tests for symmetry operation functions"""

    def test_symmetry_operation_is_initialised_from_an_array( self ):
        array = np.array( [ [ 1, 0 ], [ 0, 1 ] ] )
        so = SymmetryOperation( array )
        np.testing.assert_array_equal( so.matrix, np.array( array ) )

    def test_symmetry_operation_is_initialised_from_a_list( self ):
        this_list = [ [ 1, 0 ], [ 0, 1 ] ]
        so = SymmetryOperation( this_list )
        np.testing.assert_array_equal( so.matrix, np.array( this_list ) )

    def test_symmetry_operation_raises_typeerror_for_invalid_type( self ):
        objects = [ 'foo', 1, None ]
        for o in objects:
            with self.assertRaises( TypeError ):
                SymmetryOperation( o )

    def test_symmetry_operation_raises_valueerror_for_nonsquare_matrix( self ):
        array = np.array( [ [ 1, 0, 0 ], [ 0, 0, 1 ] ] )
        with self.assertRaises( ValueError ):
            SymmetryOperation( array )

    def test_symmetry_operation_is_initialised_with_label( self ):
        matrix = np.array( [ [ 1, 0 ], [ 0, 1 ] ] )
        label = 'E'
        so = SymmetryOperation( matrix, label=label ) 
        self.assertEqual( so.label, label )

    def test_mul( self ):
        matrix_a = np.array( [ [ 0, 1 ], [ 1, 0 ] ] )
        matrix_b = np.array( [ [ 1, 0 ], [ 0, 1 ] ] )
        so_a = SymmetryOperation( matrix_a )
        so_b = SymmetryOperation( matrix_b )
        np.testing.assert_array_equal( ( so_a * so_b ).matrix , np.array( [ [ 0, 1 ], [ 1, 0 ] ] ) )

    def test_invert( self ):
        matrix_a = np.array( [ [ 0, 1, 0 ], [ 0, 0, 1 ], [ 1, 0, 0 ] ] )
        matrix_b = np.array( [ [ 0, 0, 1 ], [ 1, 0, 0 ], [ 0, 1, 0 ] ] )
        so = SymmetryOperation( matrix_a )
        np.testing.assert_array_equal( so.invert().matrix, matrix_b )

    def test_invert_sets_label( self ):
        matrix_a = np.array( [ [ 0, 1, 0 ], [ 0, 0, 1 ], [ 1, 0, 0 ] ] )
        so = SymmetryOperation( matrix_a ).invert( label='A' )
        self.assertEqual( so.label, 'A' )

    def test_symmetry_operation_is_initialised_with_label( self ):
        matrix = np.array( [ [ 1, 0 ], [ 0, 1 ] ] )
        label = 'E'
        so = SymmetryOperation( matrix, label=label )
        self.assertEqual( so.label, label )

    def test_similarity_transform( self ):
        matrix_a = np.array( [ [ 0, 1, 0 ], [ 0, 0, 1 ], [ 1, 0, 0 ] ] )
        matrix_b = np.array( [ [ 1, 0, 0 ], [ 0, 0, 1 ], [ 0, 1, 0 ] ] )
        matrix_c = np.linalg.inv( matrix_a )
        so_a = SymmetryOperation( matrix_a )
        so_b = SymmetryOperation( matrix_b )
        np.testing.assert_array_equal( so_a.similarity_transform( so_b ).matrix, matrix_c )

    def test_similarity_transform_with_label( self ):
        matrix_a = np.array( [ [ 0, 1, 0 ], [ 0, 0, 1 ], [ 1, 0, 0 ] ] )
        matrix_b = np.array( [ [ 1, 0, 0 ], [ 0, 0, 1 ], [ 0, 1, 0 ] ] )
        matrix_c = np.linalg.inv( matrix_a )
        so_a = SymmetryOperation( matrix_a )
        so_b = SymmetryOperation( matrix_b )
        label = 'foo'
        np.testing.assert_array_equal( so_a.similarity_transform( so_b, label=label ).label, label )

    def test_character( self ):
        matrix = np.array( [ [ 1, 0 ], [ 0, 1 ] ] )
        so = SymmetryOperation( matrix )
        self.assertEqual( so.character(), 2 )

    def test_se_label( self ):
        matrix = np.array( [ [ 1, 0 ], [ 0, 1 ] ] )
        so = SymmetryOperation( matrix )
        so.set_label( 'new_label' )
        self.assertEqual( so.label, 'new_label' )

    def test_repr( self ):
        matrix = np.array( [ [ 1, 0 ], [ 0, 1 ] ] )
        so = SymmetryOperation( matrix, label='L' )
        this_repr = so.__repr__()
        self.assertNotEqual( this_repr.find( 'L' ), 0 )
        self.assertNotEqual( this_repr.find( "[[1, 0],\n[0, 1]]" ), 0 )

class SymmetryOperationModuleFunctionsTestCase( unittest.TestCase ):

    def test_is_square_returns_true_if_matrix_is_square( self ):
        matrix = np.array( [ [ 1, 0 ], [ 0, 1 ] ] )
        self.assertEqual( is_square( matrix ), True )

    def test_is_square_returns_false_if_matrix_is_not_square( self ):
        matrix = np.array( [ [ 1, 0, 1 ], [ 0, 1, 1 ] ] )
        self.assertEqual( is_square( matrix ), False )

    def test_is_permutation_matrix_returns_true_if_true( self ):
        matrix = np.array( [ [ 0, 1 ], [ 1, 0 ] ] )
        self.assertEqual( is_permutation_matrix( matrix ), True ) 

    def test_is_permutation_matrix_returns_false_if_false( self ):
        matrix = np.array( [ [ 1, 1 ], [ 0, 0 ] ] )
        self.assertEqual( is_permutation_matrix( matrix ), False ) 

class SymmetryOperationO3TestCase( unittest.TestCase ):
    """Tests for O(3) symmetry operation functions"""
    def setUp(self):
        # absolute tolerance
        self.atol = 1e-6
        # relative tolerance
        self.rtol = 1e-6
        # decimal power tolerance 10**(-ptol)
        self.ptol = 6
        # random scalar
        self.scalar_0 = - 2.57297 * sqrt(5.0)
        # rotational angle to compare to
        self.angle_0 = 2 * pi / sqrt(26.643)
        # rotational vector
        self.rot_vec_0 = np.array( [0.4578, - 1.639, - 4.25] )
        # rotational matrix (as a list object) to compare to
        self.list_0 = [[ 1, 0,                      0                    ],
                       [ 0, np.cos(self.angle_0), - np.sin(self.angle_0) ],
                       [ 0, np.sin(self.angle_0),   np.cos(self.angle_0) ]]
        # same as an NumPy array
        self.array_0 = np.array( self.list_0 )
        # as advanced SciPy Rotation object
        self.rotation_0 = Rotation.from_matrix( self.list_0 )
        # rotational vector along direction [111], mag = 2pi/3 rad
        self.rot_vec_111_2pi_3 = (2 * pi / 3) * np.array( [1, 1, 1] ) / sqrt(3)
        # reflection (mirror) plane (100)
        self.m001 = np.array( [ [  1, 0, 0 ],
                                 [ 0, 1, 0 ],
                                 [ 0, 0, -1 ] ] )
        # Unitary determinant, but non-rotational matrix
        self.matrix_unit_det = [[ 0.5, 0, 0 ],
                                [ 0,   2, 0 ],
                                [ 0,   0, 1 ]]

    def test_symmetry_operation_is_initialised_from_an_array( self ):
        so = SymmetryOperationO3( self.array_0 )
        np.testing.assert_allclose( so.matrix, self.array_0 )
    
    def test_symmetry_operation_is_initialised_from_a_list( self ):
        so = SymmetryOperationO3( self.list_0 )
        np.testing.assert_allclose( so.matrix, np.array( self.list_0 ) )
    
    def test_symmetry_operation_is_initialised_from_a_rotation( self ):
        so = SymmetryOperationO3( self.rotation_0 )
        np.testing.assert_allclose( so.matrix, self.rotation_0.as_matrix() )

    def test_symmetry_operation_raises_typeerror_for_invalid_type( self ):
        objects = [ 'foo', 1, None ]
        for o in objects:
            with self.assertRaises( TypeError ):
                SymmetryOperationO3( o )

    def test_symmetry_operation_raises_valueerror_for_nonsquare_matrix( self ):
        array = np.array( [ [ 1, 0, 0 ], [ 0, 0, 1 ] ] )
        with self.assertRaises( ValueError ):
            SymmetryOperationO3( array )

    def test_symmetry_operation_is_initialised_with_label( self ):
        matrix = np.array( [ [ 1, 0 ], [ 0, 1 ] ] )
        label = 'E'
        so = SymmetryOperationO3( matrix, label=label ) 
        self.assertEqual( so.label, label )

    def test_mul( self ):
        rot_0 = Rotation.from_rotvec(self.rot_vec_0)
        rot_2 = Rotation.from_rotvec(2 * self.rot_vec_0)
        so_0 = SymmetryOperationO3( rot_0 )
        so_2 = SymmetryOperationO3( rot_2 )
        np.testing.assert_allclose( ( so_0 * so_0 ).matrix , so_2.matrix )

    def test_invertO3( self ):
        matrix_a = self.array_0
        so = SymmetryOperationO3( matrix_a )
        so_inv = so.invert()
        np.testing.assert_allclose( (so * so_inv).matrix, np.identity( 3 ),
                                     atol = self.atol)

    def test_invert_sets_label( self ):
        matrix_a = np.array( [ [ 0, 1, 0 ], [ 0, 0, 1 ], [ 1, 0, 0 ] ] )
        so = SymmetryOperationO3( matrix_a ).invert( label='A' )
        self.assertEqual( so.label, 'A' )

    def test_symmetry_operation_is_initialised_with_label( self ):
        matrix = np.array( self.list_0 )
        label = 'E'
        so = SymmetryOperationO3( matrix, label=label )
        self.assertEqual( so.label, label )

    def test_similarity_transform( self ):
        matrix_a = np.array( [ [ 0, 1, 0 ], [ 0, 0, 1 ], [ 1, 0, 0 ] ] )
        matrix_b = np.array( [ [ 1, 0, 0 ], [ 0, 0, 1 ], [ 0, 1, 0 ] ] )
        matrix_c = np.linalg.inv( matrix_a )
        so_a = SymmetryOperationO3( matrix_a )
        so_b = SymmetryOperationO3( matrix_b )
        np.testing.assert_array_equal( so_a.similarity_transform( so_b ).matrix, matrix_c )

    def test_se_label( self ):
        matrix = np.array( self.list_0 )
        so = SymmetryOperationO3( matrix )
        so.set_label( 'new_label' )
        self.assertEqual( so.label, 'new_label' )

    def test_repr( self ):
        matrix = np.array( self.list_0 )
        so = SymmetryOperationO3( matrix, label='L' )
        this_repr = so.__repr__()
        self.assertNotEqual( this_repr.find( 'L' ), 0 )
        self.assertNotEqual( this_repr.find( "[[1, 0],\n[0, 1]]" ), 0 )
    
    def test_ensure_unit_det_properO3( self ):
        #first, let's check the regular proper rotation
        so_1 = SymmetryOperationO3( self.rotation_0 )
        det_1 = det( so_1.matrix )
        np.testing.assert_approx_equal( det_1, 1, self.ptol )
        #let's modify the transformation matrix by a random coefficient
        with self.assertRaises( ValueError ):
            so_2 = SymmetryOperationO3 ( self.scalar_0 * self.array_0 )
        #now, let's test the case of unitary determinant, but non-rotational matrix
        with self.assertRaises( ValueError ):
            so_4 = SymmetryOperationO3 ( self.matrix_unit_det )

    def test_ensure_unit_det_improper( self ):
        #first, let's check the regular improper rotation
        so_1 = SymmetryOperationO3( - 1 * self.array_0 )
        det_1 = det( so_1.matrix )
        np.testing.assert_approx_equal(det_1, - 1, self.ptol)
        #now, let's modify the transformation matrix by a random coefficient
        with self.assertRaises( ValueError ):
            so_2 = SymmetryOperationO3 ( self.scalar_0 * self.array_0 )
        so_3_1 = SymmetryOperationO3( self.m001 )
        so_3_2 = SymmetryOperationO3( self.rotation_0 )
        so_3 = so_3_1 * so_3_2 
        det_3 = det( so_3.matrix )
        np.testing.assert_approx_equal(det_3, - 1, self.ptol)
    
    def test_improper_flags( self ):
        #first, let's set the regular proper rotation
        so_1 = SymmetryOperationO3( self.rotation_0 )
        #then, let's set the regular improper rotation
        so_2 = SymmetryOperationO3( - 1 * self.array_0 )
        self.assertEqual( so_1.improper, False )
        self.assertEqual( so_2.improper, True )
        self.assertEqual( (so_1 * so_2).improper, True )
        self.assertEqual( (so_2 * so_2).improper, False )
        # now, let's check the dichromatic reversal flag 'P'
        self.assertEqual( len( so_1.dich_operations ), 0 )
        self.assertEqual( so_2.dich_operations, { 'P' } )
        self.assertEqual( (so_1 * so_2).dich_operations, { 'P' } )
        self.assertEqual( len( ( so_2 * so_2 ).dich_operations ), 0 )
        so_3 = SymmetryOperationO3( - 1 * self.array_0, dich_operations = 'C' )
        self.assertEqual( so_3.dich_operations, { 'P', 'C' } )
        self.assertEqual( ( so_3 * so_1 ).dich_operations, { 'C', 'P' } )
        so_3.dich_operations = { 'T', 'P' }
        self.assertEqual( so_3.dich_operations, { 'P', 'T' } )
        # one cannot assign the parity reversal to the proper rotation transformation
        with self.assertRaises( ValueError ):
            so_1.dich_operations = { 'P' }
        with self.assertRaises( ValueError ):
            so_1.dich_operations = { 'P', 'T' }
        # and vice versa
        with self.assertRaises( ValueError ):
            so_3.matrix = self.array_0

    def test_mul_physical_scalar( self ):
        dich_test = {"C":1, "P":1, "T":1}
        scalar_test = 273.15
        pq = PhysicalQuantity( value = scalar_test, label = 'temperature',
                               dich = dich_test )
        #first, let's set the regular proper rotation
        so_1 = SymmetryOperationO3( matrix = self.rotation_0 )
        # let's add time-reversal operation here
        # it must have no effect on this scalar
        so_1.dich_operations = {"T"}
        so_1_copy = deepcopy( so_1 )
        with self.assertRaises( ValueError ):
            so_1_copy.dich_operations = "P"
        #then, let's set the regular improper rotation
        so_inv = SymmetryOperationO3( matrix = ( -1 ) * np.identity( 3 ) )
        so_2 = SymmetryOperationO3( matrix = self.rotation_0 ) * so_inv
        # let's add charge-reversal operation here
        # it must have no effect on this scalar
        so_2.add_dich_operations("C")
        #test cases
        pq_updated = so_1 * pq
        self.assertEqual( pq_updated.value, scalar_test )
        self.assertEqual( pq_updated.dich, dich_test )
        self.assertEqual( pq_updated.label, pq.label )
        pq_updated = so_2 * pq
        self.assertEqual( pq_updated.value, scalar_test )
        self.assertEqual( pq_updated.dich, dich_test )
        self.assertEqual( pq_updated.label, pq.label )
        pq_updated = (so_2 * so_1 ) * pq
        self.assertEqual( pq_updated.value, scalar_test )
        self.assertEqual( pq_updated.dich, dich_test )
        self.assertEqual( pq_updated.label, pq.label )

    def test_mul_physical_pseudoscalar( self ):
        dich_test = {"C":1, "P":-1, "T":1}
        pseudoscalar_test = sqrt( 42 )
        pseudoscalar_test_tensor = sqrt( 42 ) * np.identity( 3 )
        pq = PhysicalQuantity( value = pseudoscalar_test, label = 'the pseudoscalar',
                               dich = dich_test )
        pq_tensor = PhysicalQuantity( label = 'the pseudoscalar',
                               dich = dich_test )
        pq_tensor.value = pseudoscalar_test_tensor
        #first, let's set the regular proper rotation
        so_1 = SymmetryOperationO3( matrix = self.rotation_0 )
        # let's add time-reversal operation here
        # it must have no effect on this pseudoscalar
        so_1.dich_operations = {"T"}
        #then, let's set the regular improper rotation
        so_inv = SymmetryOperationO3( matrix = ( -1 ) * np.identity( 3 ) )
        so_2 = SymmetryOperationO3( matrix = self.rotation_0 ) * so_inv
        # let's add charge-reversal operation here
        # it must have no effect on this pseudoscalar
        so_2.add_dich_operations("C")
        #test cases
        pq_updated = PhysicalQuantity()
        pq_updated = so_1 * pq
        self.assertEqual( pq_updated.value, pseudoscalar_test )
        self.assertEqual( pq_updated.dich, dich_test )
        self.assertEqual( pq_updated.label, pq.label )
        pq_updated = so_2 * pq
        self.assertEqual( pq_updated.value, - pseudoscalar_test )
        self.assertEqual( pq_updated.dich, dich_test )
        self.assertEqual( pq_updated.label, pq.label )
        pq_updated = (so_2 * so_1 ) * pq
        self.assertEqual( pq_updated.value, - pseudoscalar_test )
        self.assertEqual( pq_updated.dich, dich_test )
        self.assertEqual( pq_updated.label, pq.label )
        pq_updated = so_2 * (so_1 * pq)
        self.assertEqual( pq_updated.value, - pseudoscalar_test )
        self.assertEqual( pq_updated.dich, dich_test )
        self.assertEqual( pq_updated.label, pq.label )
        pq_updated = so_2 * so_1 * pq
        self.assertEqual( pq_updated.value, - pseudoscalar_test )
        self.assertEqual( pq_updated.dich, dich_test )
        self.assertEqual( pq_updated.label, pq.label )
        pq_updated = so_1 * so_2 * pq
        self.assertEqual( pq_updated.value, - pseudoscalar_test )
        self.assertEqual( pq_updated.dich, dich_test )
        self.assertEqual( pq_updated.label, pq.label )

        pq_updated = PhysicalQuantity()
        pq_updated = so_1 * pq_tensor
        self.assertEqual( pq_updated == pq, True )
        self.assertEqual( pq == pq_updated, True )
        self.assertEqual( pq != pq_updated, False )

    def test_mul_electric_field( self ):
        # electrical field, a polar time-even vector.
        # It has the odd dichromatic symmetry against charge inversion.
        dich_test = {"C":-1, "P":1, "T":1}
        vec_test_0 = np.array( [75**(1/3.0), 0, 0] )
        pq = PhysicalQuantity( value = vec_test_0, label = 'E',
                               dich = dich_test )
        #first, let's set the regular proper rotation
        # 120-degrees rotation around direction [111]
        rot_1 = Rotation.from_rotvec( self.rot_vec_111_2pi_3, degrees = False )
        so_1 = SymmetryOperationO3( matrix = rot_1 )
        # expected vector after this rotation:
        vec_test_updated = np.array( [0, 75**(1/3.0), 0] )
        # let's add time-reversal operation here
        # it must have no effect on this polar vector
        so_1.dich_operations = {"T"}

        #now, let's apply spatial inversion along with charge reversal operation.
        so_inv = SymmetryOperationO3( matrix = ( -1 ) * np.identity( 3 ) )
        so_2 = SymmetryOperationO3() * so_inv
        # let's add charge-reversal operation here
        # it must reverse electric field
        so_2.add_dich_operations("C")
        
        #now, let's do just charge inversion
        so_3 = SymmetryOperationO3()
        # let's add charge-reversal operation here
        # it must reverse electric field
        so_3.add_dich_operations("C")

        #now, let's do just spatial inversion
        so_inv = SymmetryOperationO3( matrix = ( -1 ) * np.identity( 3 ) )
        so_4 = SymmetryOperationO3() * so_inv
        # let's an additional reversal operation here
        # it must have no effect on this time-even polar vector
        so_4.dich_operations = {"T"}

        #test cases
        pq_updated = so_1 * pq
        np.testing.assert_allclose( pq_updated.value, vec_test_updated,
                                     atol = self.atol)
        np.testing.assert_array_equal( pq_updated.dich, dich_test )
        np.testing.assert_array_equal( pq_updated.label, 'E' )

        pq_updated = so_2 * pq
        np.testing.assert_allclose( pq_updated.value, vec_test_0,
                                     rtol = self.rtol)
        np.testing.assert_array_equal( pq_updated.dich, dich_test )
        np.testing.assert_array_equal( pq_updated.label, pq.label )

        pq_updated = so_3 * pq
        np.testing.assert_allclose( pq_updated.value, - vec_test_0,
                                     rtol = self.rtol)
        np.testing.assert_array_equal( pq_updated.dich, dich_test )
        np.testing.assert_array_equal( pq_updated.label, 'E' )

        pq_updated = so_4 * pq
        np.testing.assert_allclose( pq_updated.value, - vec_test_0,
                                     rtol = self.rtol)
        np.testing.assert_array_equal( pq_updated.dich, dich_test )
        np.testing.assert_array_equal( pq_updated.label, pq.label )

    def test_mul_magnetic_field( self ):
        # magnetic field, an axial time-odd vector.
        # It has the odd dichromatic symmetry against charge inversion.
        dich_test = {"C":-1, "P":-1, "T":-1}
        vec_test_0 = np.array( [75**(1/3.0), 0, 0] )
        pq = PhysicalQuantity( value = vec_test_0, label = 'B',
                               dich = dich_test )
        # first, let's set the regular proper rotation
        # 120-degrees rotation around direction [111]
        rot_1 = Rotation.from_rotvec( self.rot_vec_111_2pi_3, degrees = False )
        so_1 = SymmetryOperationO3( matrix = rot_1 )
        # let's add time-reversal operation here
        # it must invert this time-odd vector
        so_1.dich_operations = {"T"}
        # expected vector after this rotation:
        vec_test_updated_1 = (-1) * np.array( [0, 75**(1/3.0), 0] )

        # now, let's apply spatial inversion along with charge reversal operation.
        # inversion should have no effect on an axial vector
        so_inv = SymmetryOperationO3( matrix = ( -1 ) * np.identity( 3 ) )
        so_2 = SymmetryOperationO3() * so_inv
        # let's add charge-reversal operation here
        # it must reverse magnetic field
        so_2.add_dich_operations("C")
        vec_test_updated_2 = - vec_test_0
        
        #now, let's do just charge inversion
        so_3 = SymmetryOperationO3()
        # let's add charge-reversal operation here
        # it must reverse magnetic field
        so_3.add_dich_operations("C")
        vec_test_updated_3 = - vec_test_0

        #now, let's do just spatial inversion
        so_inv = SymmetryOperationO3( matrix = ( -1 ) * np.identity( 3 ) )
        so_4 = SymmetryOperationO3() * so_inv
        # let's add an additional reversal operation here
        # it must have reverse this axial time-odd vector
        so_4.dich_operations = {"T"}
        vec_test_updated_4 = - vec_test_0

        #test cases
        pq_updated = so_1 * pq
        np.testing.assert_allclose( pq_updated.value, vec_test_updated_1,
                                     atol = self.atol)
        np.testing.assert_array_equal( pq_updated.dich, dich_test )
        np.testing.assert_array_equal( pq_updated.label, "B" )

        pq_updated = so_2 * pq
        np.testing.assert_allclose( pq_updated.value, vec_test_updated_2,
                                     rtol = self.rtol)
        np.testing.assert_array_equal( pq_updated.dich, dich_test )
        np.testing.assert_array_equal( pq_updated.label, pq.label )

        pq_updated = so_3 * pq
        np.testing.assert_allclose( pq_updated.value, vec_test_updated_3,
                                     rtol = self.rtol)
        np.testing.assert_array_equal( pq_updated.dich, dich_test )
        np.testing.assert_array_equal( pq_updated.label, "B" )

        pq_updated = so_4 * pq
        np.testing.assert_allclose( pq_updated.value, vec_test_updated_4,
                                     rtol = self.rtol)
        np.testing.assert_array_equal( pq_updated.dich, dich_test )
        np.testing.assert_array_equal( pq_updated.label, pq.label )

class SymmetryOperationSO3TestCase( unittest.TestCase ):
    """Tests for SO(3) symmetry operation functions"""
    def setUp(self):
        # absolute tolerance
        self.atol = 1e-15
        # relative tolerance
        self.rtol = 1e-6
        # decimal power tolerance 10**(-ptol)
        self.ptol = 6
        # random scalar
        self.scalar_0 = - 1.57297 * sqrt(6.0)
        # rotational angle to compare to
        self.angle_0 = 2 * pi / sqrt(11.643)
        # rotational vector
        self.rot_vec_0 = np.array( [0.4578, - 1.639, - 4.25] )
        # rotational matrix (as a list object) to compare to
        self.list_0 = [[ 1, 0,                      0                    ],
                       [ 0, np.cos(self.angle_0), - np.sin(self.angle_0) ],
                       [ 0, np.sin(self.angle_0),   np.cos(self.angle_0) ]]
        # same as an NumPy array
        self.array_0  =   np.array( self.list_0 )
        # as advanced SciPy Rotation object
        self.rotation_0 = Rotation.from_matrix(self.list_0)
        # rotational vector along direction [111], mag = 2pi/3 rad
        self.rot_vec_111_2pi_3 = (2 * pi / 3) * np.array( [1, 1, 1] ) / sqrt(3)
        # reflection (mirror) plane (100)
        self.m001 = np.array( [ [  1, 0, 0 ],
                                 [ 0, 1, 0 ],
                                 [ 0, 0, -1 ] ] )
        # Unitary determinant, but non-rotational matrix
        self.matrix_unit_det = [[ 0.5, 0, 0 ],
                                [ 0,   2, 0 ],
                                [ 0,   0, 1 ]]

    def test_ensure_SO3_is_proper_rot( self ):
        with self.assertRaises( ValueError ):
            #the regular proper rotation class with wrong matrix
            so_1 = SymmetryOperationSO3( matrix = - 1 * self.array_0 )

    def test_ensure_SO3_is_proper_rot_dich( self ):
        with self.assertRaises( ValueError ):
            #the regular proper rotation class with wrong dichromatic reversal
            so_1 = SymmetryOperationSO3( matrix = self.array_0, dich_operations = { 'P' } )

    def test_symmetry_operation_is_initialised_from_a_list( self ):
        so = SymmetryOperationSO3( self.list_0 )
        np.testing.assert_allclose( so.matrix, np.array( self.list_0 ) )
    
    def test_symmetry_operation_is_initialised_from_a_rotation( self ):
        so = SymmetryOperationSO3( self.rotation_0 )
        np.testing.assert_allclose( so.matrix, self.rotation_0.as_matrix() )
    
    def test_ensure_unit_det_properSO3( self ):
        #first, let's check the regular proper rotation
        so_1 = SymmetryOperationSO3( self.rotation_0 )
        det_1 = det( so_1.matrix )
        np.testing.assert_approx_equal( det_1, 1, self.ptol )
        #now, let's modify the transformation matrix by a random coefficient
        with self.assertRaises( ValueError ):
            so_2 = SymmetryOperationSO3( self.scalar_0 * self.array_0 )
        #let's test the improper case
        with self.assertRaises( ValueError ):
            so_3 = SymmetryOperationSO3( self.m001 )
        #let's test unit determinant, but non-rotational case
        with self.assertRaises( ValueError ):
            so_3 = SymmetryOperationSO3( self.matrix_unit_det )
    
    def test_invert( self ):
        matrix_a = self.array_0
        so = SymmetryOperationSO3( matrix_a )
        so_inv = so.invert()
        np.testing.assert_allclose( (so * so_inv).matrix, np.identity( 3 ),
                                     atol = self.atol)
    
    def test_mul( self ):
        rot_0 = Rotation.from_rotvec(self.rot_vec_0)
        rot_2 = Rotation.from_rotvec(2 * self.rot_vec_0)
        so_0 = SymmetryOperationSO3( rot_0 )
        so_2 = SymmetryOperationSO3( rot_2 )
        np.testing.assert_allclose( ( so_0 * so_0 ).matrix , so_2.matrix )
    
    def test_returnO3_with_improper_flag( self ):
        #first, let's set the regular proper rotation
        so_1 = SymmetryOperationSO3( self.rotation_0 )
        #then, let's set the regular improper rotation
        so_2 = SymmetryOperationO3( - 1 * self.array_0 )
        self.assertEqual( (so_1 * so_2).improper, True )
    
    def test_mul_physical_scalar( self ):
        dich_test = {"C":1, "P":1, "T":1}
        scalar_test = 273.15
        pq = PhysicalQuantity( value = scalar_test, label = 'temperature',
                               dich = dich_test )
        #first, let's set the regular proper rotation
        so_1 = SymmetryOperationSO3( matrix = self.rotation_0 )
        # let's add time-reversal operation here
        # it must have no effect on this scalar
        so_1.dich_operations = {"T"}
        #then, let's set the regular proper rotation
        so_2 = SymmetryOperationSO3( matrix = self.rotation_0 )
        # let's add charge-reversal operation here
        # it must have no effect on this scalar
        so_2.add_dich_operations("C")
        #test cases
        pq_updated = so_1 * pq
        self.assertEqual( pq_updated.value, scalar_test )
        self.assertEqual( pq_updated.dich, dich_test )
        self.assertEqual( pq_updated.label, pq.label )
        pq_updated = so_2 * pq
        self.assertEqual( pq_updated.value, scalar_test )
        self.assertEqual( pq_updated.dich, dich_test )
        self.assertEqual( pq_updated.label, pq.label )
        pq_updated = (so_2 * so_1 ) * pq
        self.assertEqual( pq_updated.value, scalar_test )
        self.assertEqual( pq_updated.dich, dich_test )
        self.assertEqual( pq_updated.label, pq.label )

    def test_mul_physical_pseudoscalar( self ):
        dich_test = {"C":-1, "P":-1, "T":1}
        pseudoscalar_test = sqrt( 42 )
        pq = PhysicalQuantity( value = pseudoscalar_test, label = 'the pseudoscalar',
                               dich = dich_test )
        #first, let's set the regular proper rotation
        so_1 = SymmetryOperationSO3( matrix = self.rotation_0 )
        # let's add time-reversal operation here
        # it must have no effect on this pseudoscalar
        so_1.dich_operations = {"T"}
        #then, let's set the regular proper rotation
        so_2 = SymmetryOperationSO3( matrix = self.rotation_0 )
        # let's add charge-reversal operation here
        # it must invert this charge-odd pseudoscalar
        so_2.dich_operations = {"C"}
        #test cases
        pq_updated = PhysicalQuantity()
        pq_updated = so_1 * pq
        self.assertEqual( pq_updated.value, pseudoscalar_test )
        self.assertEqual( pq_updated.dich, dich_test )
        self.assertEqual( pq_updated.label, pq.label )
        pq_updated = so_1 * pq
        pt = pseudoscalar_test
        pseudoscalar_test_as_tensor = np.array([[pt, 0, 0],
                                                [0, pt, 0],
                                                [0, 0, pt]
                                                ])
        pq_as_tensor = deepcopy( pq )
        pq_as_tensor.value = pseudoscalar_test_as_tensor
        self.assertEqual( pq_updated == pq_as_tensor, True )
        self.assertEqual( pq_updated != pq_as_tensor, False )
        self.assertEqual( pq_as_tensor == pq_updated, True )
        self.assertEqual( pq_as_tensor != pq_updated, False )
        self.assertEqual( pq_updated.dich, dich_test )
        self.assertEqual( pq_updated.label, pq.label )
        pq_updated = so_2 * pq
        self.assertEqual( pq_updated.value, - pseudoscalar_test )
        self.assertEqual( pq_updated.dich, dich_test )
        self.assertEqual( pq_updated.label, pq.label )
        pq_updated = (so_2 * so_1 ) * pq
        self.assertEqual( pq_updated.value, - pseudoscalar_test )
        self.assertEqual( pq_updated.dich, dich_test )
        self.assertEqual( pq_updated.label, pq.label )
        pq_updated = so_2 * (so_1 * pq)
        self.assertEqual( pq_updated.value, - pseudoscalar_test )
        self.assertEqual( pq_updated.dich, dich_test )
        self.assertEqual( pq_updated.label, pq.label )
        pq_updated = so_2 * so_1 * pq
        self.assertEqual( pq_updated.value, - pseudoscalar_test )
        self.assertEqual( pq_updated.dich, dich_test )
        self.assertEqual( pq_updated.label, pq.label )
        pq_updated = so_1 * so_2 * pq
        self.assertEqual( pq_updated.value, - pseudoscalar_test )
        self.assertEqual( pq_updated.dich, dich_test )
        self.assertEqual( pq_updated.label, pq.label )

    def test_mul_electric_field( self ):
        # electrical field, a polar time-even vector.
        # It has the odd dichromatic symmetry against charge inversion.
        dich_test = {"C":-1, "P":1, "T":1}
        vec_test_0 = np.array( [75**(1/3.0), 0, 0] )
        pq = PhysicalQuantity( value = vec_test_0, label = 'E',
                               dich = dich_test )
        #first, let's set the regular proper rotation
        # 120-degrees rotation around direction [111]
        rot_1 = Rotation.from_rotvec( self.rot_vec_111_2pi_3, degrees = False )
        so_1 = SymmetryOperationSO3( matrix = rot_1 )
        # expected vector after this rotation:
        vec_test_updated = np.array( [0, 75**(1/3.0), 0] )
        # let's add time-reversal operation here
        # it must have no effect on this polar vector
        so_1.dich_operations = {"T"}

        so_2 = SymmetryOperationSO3()
        # let's add charge-reversal operation here
        # it must reverse electric field
        so_2.add_dich_operations("C")

        so_3 = SymmetryOperationSO3()
        # let's an additional reversal operation here
        # it must have no effect on this time-even polar vector
        so_3.dich_operations = {"T"}

        #test cases
        pq_updated = so_1 * pq
        np.testing.assert_allclose( pq_updated.value, vec_test_updated,
                                     atol = self.atol)
        np.testing.assert_array_equal( pq_updated.dich, dich_test )
        np.testing.assert_array_equal( pq_updated.label, 'E' )

        pq_updated = so_2 * pq
        np.testing.assert_allclose( pq_updated.value, - vec_test_0,
                                     rtol = self.rtol)
        np.testing.assert_array_equal( pq_updated.dich, dich_test )
        np.testing.assert_array_equal( pq_updated.label, pq.label )

        pq_updated = so_3 * pq
        np.testing.assert_allclose( pq_updated.value, vec_test_0,
                                     rtol = self.rtol)
        np.testing.assert_array_equal( pq_updated.dich, dich_test )
        np.testing.assert_array_equal( pq_updated.label, pq.label )

    def test_mul_magnetic_field( self ):
        # magnetic field, an axial time-odd vector.
        # It has the odd dichromatic symmetry against charge inversion.
        dich_test = {"C":-1, "P":-1, "T":-1}
        vec_test_0 = np.array( [75**(1/3.0), 0, 0] )
        pq = PhysicalQuantity( value = vec_test_0, label = 'B',
                               dich = dich_test )
        # first, let's set the regular proper rotation
        # 120-degrees rotation around direction [111]
        rot_1 = Rotation.from_rotvec( self.rot_vec_111_2pi_3, degrees = False )
        so_1 = SymmetryOperationSO3( matrix = rot_1 )
        # let's add time-reversal operation here
        # it must invert this time-odd vector
        so_1.dich_operations = {"T"}
        # expected vector after this rotation:
        vec_test_updated_1 = (-1) * np.array( [0, 75**(1/3.0), 0] )

        so_2 = SymmetryOperationSO3()
        # let's add charge-reversal operation here
        # it must reverse magnetic field
        so_2.add_dich_operations("C")
        vec_test_updated_2 = - vec_test_0

        so_3 = SymmetryOperationSO3()
        # let's add an additional reversal operation here
        # it must have reverse this axial time-odd vector
        so_3.dich_operations = {"T"}
        vec_test_updated_3 = - vec_test_0

        #test cases
        pq_updated = so_1 * pq
        np.testing.assert_allclose( vec_test_updated_1, pq_updated.value,
                                     rtol = self.rtol, atol = self.atol )
        np.testing.assert_array_equal( pq_updated.dich, dich_test )
        np.testing.assert_array_equal( pq_updated.label, "B" )

        pq_updated = so_2 * pq
        np.testing.assert_allclose( pq_updated.value, vec_test_updated_2,
                                     rtol = self.rtol)
        np.testing.assert_array_equal( pq_updated.dich, dich_test )
        np.testing.assert_array_equal( pq_updated.label, pq.label )

        pq_updated = so_3 * pq
        np.testing.assert_allclose( pq_updated.value, vec_test_updated_3,
                                     rtol = self.rtol)
        np.testing.assert_array_equal( pq_updated.dich, dich_test )
        np.testing.assert_array_equal( pq_updated.label, pq.label )

class LimitingSymmetryOperationO3TestCase( unittest.TestCase ):
    """Tests for limiting O(3) symmetry operation functions"""
    def setUp(self):
        # absolute tolerance
        self.atol = 1e-6
        # relaive tolerance
        self.rtol = 1e-6
        # decimal power tolerance 10**(-ptol)
        self.ptol = 6
        # axis of an infinitesimal rotation operation
        self.array_0 = np.array( [0.4578, - 1.639, - 4.25] )
        self.list_0 = [0.4578, - 1.639, - 4.25]
        self.array_Z = np.array( [0, 0, 1] )

    def test_symmetry_operation_is_initialised_from_an_array( self ):
        so = LimitingSymmetryOperationO3( self.array_0 )
        np.testing.assert_allclose( so.axis, self.array_0 )
    
    def test_symmetry_operation_is_initialised_from_a_list( self ):
        so = LimitingSymmetryOperationO3( self.list_0 )
        np.testing.assert_allclose( so.axis, self.array_0 )

    def test_symmetry_operation_is_initialised_by_defaults( self ):
        so = LimitingSymmetryOperationO3()
        np.testing.assert_allclose( so.axis, self.array_Z )

    def test_symmetry_operation_raises_errors_for_invalid_type( self ):
        objects = [ 'foo', None, 1 ]
        for o in objects:
            with self.assertRaises( TypeError ):
                LimitingSymmetryOperationO3( o )
        with self.assertRaises( ValueError ):
                LimitingSymmetryOperationO3( [1, 2] )
    
    def test_limiting_sym_operation_matrix( self ):
        so = LimitingSymmetryOperationO3()
        with self.assertRaises( ValueError ):
            so.matrix
    
    def test_repr_sym_oper( self ):
        so = LimitingSymmetryOperationO3( self.list_0 )
        print_io = io.StringIO()
        print( so, file = print_io)
        printed_str = print_io.getvalue()
        print_io.close()
        expected_str = 'SymmetryOperation\nlabel(∞)' + \
                       '\nAxis: [ 0.4578 -1.639  -4.25  ]' + \
                       '\nDichromatic reversals: \n'
        self.assertEqual( printed_str, expected_str )

    def test_repr_sym_oper_w_dich( self ):
        so = LimitingSymmetryOperationO3( self.list_0 )
        so.dich_operations = {'P','C'}
        so.add_dich_operations('T')
        so.add_dich_operations('P')
        so.add_dich_operations({'P','C'})
        print_io = io.StringIO()
        print( so, file = print_io)
        printed_str = print_io.getvalue()
        print_io.close()
        expected_str = 'SymmetryOperation\nlabel(∞-\'*)' + \
                       '\nAxis: [ 0.4578 -1.639  -4.25  ]' + \
                       '\nDichromatic reversals: [\'C\', \'P\', \'T\']\n'
        self.assertEqual( printed_str, expected_str )

    def test_repr_sym_oper_w_dich_var2( self ):
        so = LimitingSymmetryOperationO3( self.list_0 )
        so.add_dich_operations('T')
        so.add_dich_operations('P')
        so.add_dich_operations({'P','C'})
        print_io = io.StringIO()
        print( so, file = print_io)
        printed_str = print_io.getvalue()
        print_io.close()
        expected_str = 'SymmetryOperation\nlabel(∞-\'*)' + \
                       '\nAxis: [ 0.4578 -1.639  -4.25  ]' + \
                       '\nDichromatic reversals: [\'C\', \'P\', \'T\']\n'
        self.assertEqual( printed_str, expected_str )

class LimitingSymmetryOperationSO3TestCase( unittest.TestCase ):
    """Tests for limiting SO(3) symmetry operation functions"""
    def setUp(self):
        # absolute tolerance
        self.atol = 1e-6
        # relative tolerance
        self.rtol = 1e-6
        # decimal power tolerance 10**(-ptol)
        self.ptol = 6
        # axis of an infinitesimal rotation operation
        self.array_0 = np.array( [0.4578, - 1.639, - 4.25] )
        self.list_0 = [0.4578, - 1.639, - 4.25]
        self.array_Z = np.array( [0, 0, 1] )

    def test_symmetry_operation_is_initialised_from_an_array( self ):
        so = LimitingSymmetryOperationSO3( self.array_0 )
        np.testing.assert_allclose( so.axis, self.array_0 )
    
    def test_symmetry_operation_is_initialised_from_a_list( self ):
        so = LimitingSymmetryOperationSO3( self.list_0 )
        np.testing.assert_allclose( so.axis, self.array_0 )

    def test_symmetry_operation_is_initialised_by_defaults( self ):
        so = LimitingSymmetryOperationSO3()
        np.testing.assert_allclose( so.axis, self.array_Z )

    def test_symmetry_operation_raises_errors_for_invalid_type( self ):
        objects = [ 'foo', None, 1 ]
        for o in objects:
            with self.assertRaises( TypeError ):
                LimitingSymmetryOperationSO3( o )
        with self.assertRaises( ValueError ):
                LimitingSymmetryOperationSO3( [1, 2] )
    
    def test_limiting_sym_operation_matrix( self ):
        so = LimitingSymmetryOperationSO3()
        with self.assertRaises( ValueError ):
            so.matrix
    
    def test_repr_sym_oper( self ):
        so = LimitingSymmetryOperationSO3( self.list_0 )
        print_io = io.StringIO()
        print( so, file = print_io)
        printed_str = print_io.getvalue()
        print_io.close()
        expected_str = 'SymmetryOperation\nlabel(∞)' + \
                       '\nAxis: [ 0.4578 -1.639  -4.25  ]' + \
                       '\nDichromatic reversals: \n'
        self.assertEqual( printed_str, expected_str )

    def test_repr_sym_oper_w_dich( self ):
        so = LimitingSymmetryOperationSO3( self.list_0 )
        so.add_dich_operations('T')
        so.add_dich_operations('C')
        so.add_dich_operations({'C'})
        print_io = io.StringIO()
        print( so, file = print_io)
        printed_str = print_io.getvalue()
        print_io.close()
        expected_str = 'SymmetryOperation\nlabel(∞\'*)' + \
                       '\nAxis: [ 0.4578 -1.639  -4.25  ]' + \
                       '\nDichromatic reversals: [\'C\', \'T\']\n'
        self.assertEqual( printed_str, expected_str )

    def test_repr_sym_oper_w_wrong_dich( self ):
        with self.assertRaises( ValueError ):
            so = LimitingSymmetryOperationSO3( self.list_0, dich_operations = {'C','P'} )
        so = LimitingSymmetryOperationSO3( self.list_0 )
        so.add_dich_operations('T')
        with self.assertRaises( ValueError ):
            so.add_dich_operations('P')

if __name__ == '__main__':
    unittest.main()
