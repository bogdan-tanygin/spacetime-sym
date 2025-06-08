#
# Copyright (C) 2024 Dr. Bogdan Tanygin <info@deeptech.business>
# Copyright (C) 2015, 2021 Benjamin J. Morgan
#
# This file is part of spacetime-sym.
#
from math import pi, sqrt
import unittest
from spacetime import SymmetryGroup, SymmetryOperation, SymmetryOperationO3, SymmetryOperationSO3
from spacetime import LimitingSymmetryGroupScalar
from spacetime import PhysicalQuantity
from unittest.mock import Mock, MagicMock, patch, call
import numpy as np
from scipy.spatial.transform import Rotation
from numpy.linalg import det, matrix_rank
from copy import deepcopy
import io

class SymmetryGroupTestCase( unittest.TestCase ):
    """Tests for SymmetryGroup class"""

    def setUp(self):
        # absolute tolerance
        self.atol = 1e-6
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
        # rotational vector along direction [111], mag = pi rad
        self.rot_vec_111_pi = pi * np.array( [1, 1, 1] ) / sqrt(3)
        # rotational vector along direction [111], mag = 2pi/3 rad
        self.rot_vec_111_2pi_3 = (2 * pi / 3) * np.array( [1, 1, 1] ) / sqrt(3)
        # rotational vector along direction [100], mag = 2pi/6 rad
        self.rot_vec_100_2pi_6 = (2 * pi / 6) * np.array( [1, 0, 0] )
        # rotational vector along direction [111], mag = 2pi/6 rad
        self.rot_vec_111_2pi_6 = (2 * pi / 6) * np.array( [1, 1, 1] ) / sqrt(3)
        # reflection (mirror) plane (100)
        self.m001 = np.array( [ [  1, 0, 0 ],
                                 [ 0, 1, 0 ],
                                 [ 0, 0, -1 ] ] )
        # random vector
        self.vector_0 = np.array( [ -1, np.sqrt( np.pi ), 1.3454675432e6 ] )
        # random tensor
        self.tensor_0 = np.array( [ [ -1, np.sqrt( np.pi ), 1.3454675432e6 ],
                                    [ 743.3566, 0, -1.456 ],
                                    [ 1.4567, -877865, np.exp( -4 ) ] ] )

    def _compare_lists_of_sym_opers( self, so_list, so_list_expected ):
        so_exp_matched = set()
        so_unexpected = set()
        so_txt = ''
        expected_so_txt = ''
        for so_exp in so_list_expected:
            expected_so_txt += '\n{}'.format( so_exp.matrix )
            if isinstance( so_exp, SymmetryOperationO3 ):
                expected_so_txt += '\n dichromatic reversals: {} \n\n'.format( so_exp.dich_operations )
        matched_indxs_of_expected_so = set()
        indx_exp_so = set( range( len( so_list_expected ) ) )
        for so in so_list:
            so_txt += '\n{}'.format( so.matrix )
            if isinstance( so, SymmetryOperationO3 ):
                so_txt += '\n dichromatic reversals: {} \n\n'.format( so.dich_operations )
            so_matched_flag = False
            for i_exp in range( len( so_list_expected ) ):
                so_exp = so_list_expected[i_exp]
                if np.allclose( so.matrix, so_exp.matrix, atol = self.atol):
                    # default
                    dich_match_flag = True
                    if isinstance( so_exp, SymmetryOperationO3 ) or isinstance( so, SymmetryOperationO3 ):
                        if ( not isinstance( so_exp, SymmetryOperationO3 ) ) or ( not isinstance( so, SymmetryOperationO3 ) ):
                            dich_match_flag = False
                        elif ( so.dich_operations != so_exp.dich_operations ) and ( len( so.dich_operations ) > 0
                                                                                   or len( so_exp.dich_operations ) >0 ):
                            dich_match_flag = False
                    if dich_match_flag:
                        so_exp_matched.add( so_exp )
                        so_matched_flag = True
                        matched_indxs_of_expected_so.add( i_exp )
                        break
            if not so_matched_flag:
                so_unexpected.add( so )
        missed_indxs_of_expected_so = indx_exp_so - matched_indxs_of_expected_so
        missed_exp_so_txt = ''
        for i_exp in missed_indxs_of_expected_so:
            missed_exp_so_txt += '\n{}'.format( so_list_expected[i_exp].matrix )
            if isinstance( so_list_expected[i_exp], SymmetryOperationO3 ):
                missed_exp_so_txt += '\n dichromatic reversals: {} \n\n'.format( so_list_expected[i_exp].dich_operations )
        unexpected_so_txt = ''
        for so_unexp in so_unexpected:
            unexpected_so_txt += '\n{}'.format( so_unexp.matrix )
            if isinstance( so_unexp, SymmetryOperationO3 ):
                unexpected_so_txt += '\n dichromatic reversals: {} \n\n'.format( so_unexp.dich_operations )
        res_message = ''
        if len( so_unexpected ) > 0 and len( so_exp_matched ) < len( so_list_expected ):
            res_message = 'The symmetry operations are not expected' \
                          + 'Some of the expected symmetry operations are missing' \
                           + '\nExpected: ' + expected_so_txt \
                           + '\nPresent: ' + so_txt \
                           + '\nMissing:' + missed_exp_so_txt \
                           + '\nUnexpected:' + unexpected_so_txt
        elif len( so_unexpected ) == 0 and len( so_exp_matched ) < len( so_list_expected ):
            res_message =  'Some of the expected symmetry operations are missing' \
                           + '\nExpected: ' + expected_so_txt \
                           + '\nPresent: ' + so_txt \
                           + '\nMissing:' + missed_exp_so_txt
        elif len( so_unexpected ) > 0 and len( so_exp_matched ) == len( so_list_expected ):
            res_message = 'The symmetry operations are not expected' \
                           + '\nExpected: ' + expected_so_txt \
                           + '\nPresent: ' + so_txt \
                           + '\nUnexpected:' + unexpected_so_txt
        if len( res_message ) > 0:
            self.assertEqual( True, False, res_message)

    def test_init_group_wo_operations( self ):
        sg = SymmetryGroup()
        self.assertEqual( sg.order(), 1 )
        # single 3-dim identity element by default
        self.assertEqual( det( sg.symmetry_operations[ 0 ].matrix ), 1 )
        self.assertEqual( matrix_rank( sg.symmetry_operations[ 0 ].matrix ), 3 )

    def test_symmetry_group_is_initialised_w_duplicated_opers( self ):
        m1 = np.identity( 2 )
        m2 = np.identity( 2 )
        s0, s1 = SymmetryOperation( matrix = m1, force_permutation = True ), \
                 SymmetryOperation( matrix = m2, force_permutation = True )
        sg = SymmetryGroup( symmetry_operations = [ s0, s1 ] )
        # after deduplication in the group
        self._compare_lists_of_sym_opers( sg.symmetry_operations, [s0] )

    def test_symmetry_group_is_initialised_w_duplicated_opers_dich( self ):
        m0 = np.identity( 3 )
        m1 = np.identity( 3 )
        m2 = np.identity( 3 )
        s1, s2, e = SymmetryOperationO3( matrix = m1, dich_operations = {'T','C'} ), \
                 SymmetryOperationO3( matrix = m2, dich_operations = {'C','T'} ), \
                 SymmetryOperationO3( matrix = m0 )
        sg = SymmetryGroup( symmetry_operations = [ s1, s2 ] )
        # after deduplication in the group
        self._compare_lists_of_sym_opers( sg.symmetry_operations, [s2, e] )
        self.assertEqual( len( s2.dich_operations ), 2 )
        self.assertEqual( len( e.dich_operations ), 0 )

        m0 = np.identity( 3 )
        m1 = np.identity( 3 )
        m2 = np.identity( 3 )
        s1, s2, e = SymmetryOperationO3( matrix = m1, dich_operations = {'T','C'} ), \
                 SymmetryOperationO3( matrix = m2, dich_operations = {'C','T'} ), \
                 SymmetryOperationSO3( matrix = m0 )
        sg = SymmetryGroup( symmetry_operations = [ s1, s2 ] )
        # after deduplication in the group
        self._compare_lists_of_sym_opers( sg.symmetry_operations, [s2, e] )
        self.assertEqual( len( s2.dich_operations ), 2 )
        self.assertEqual( len( e.dich_operations ), 0 )

        m0 = np.identity( 3 )
        m1 = - np.identity( 3 )
        m2 = - np.identity( 3 )
        s1, s2, e = SymmetryOperationO3( matrix = m1, dich_operations = {'P','C'} ), \
                 SymmetryOperationO3( matrix = m2, dich_operations = {'C','P'} ), \
                 SymmetryOperationSO3( matrix = m0 )
        sg = SymmetryGroup( symmetry_operations = [ s1, s2 ] )
        # after deduplication in the group
        self._compare_lists_of_sym_opers( sg.symmetry_operations, [s2, e] )
        self.assertEqual( len( s2.dich_operations ), 2 )
        self.assertEqual( len( e.dich_operations ), 0 )

        m0 = np.identity( 3 )
        rot_120 = Rotation.from_rotvec( self.rot_vec_111_2pi_3, degrees = False )
        rot_120m = Rotation.from_rotvec( - self.rot_vec_111_2pi_3, degrees = False )
        e = SymmetryOperationO3( matrix = m0, dich_operations = {} )
        s_120 = SymmetryOperationO3( matrix = rot_120)
        s_120_2 = SymmetryOperationO3( matrix = rot_120)
        s_120m = SymmetryOperationO3( matrix = rot_120m)
        sg = SymmetryGroup( symmetry_operations = [ s_120, s_120_2 ] )
        # after deduplication in the group
        self._compare_lists_of_sym_opers( sg.symmetry_operations, [s_120, e, s_120m] )
        self.assertEqual( len( e.dich_operations ), 0 )

    def test_symmetry_group_is_initialised_w_different_opers( self ):
        m1 = np.identity( 2 )
        m2 = - np.identity( 2 )
        s0, s1 = SymmetryOperation( matrix = m1, force_permutation = False ), \
                 SymmetryOperation( matrix = m2, force_permutation = False )
        sg = SymmetryGroup( symmetry_operations = [ s0, s1 ] )
        # after deduplication in the group
        self._compare_lists_of_sym_opers( sg.symmetry_operations, [s1, s0] ) 

    def test_symmetry_group_is_initialised_w_different_opers_dich( self ):
        m0 = np.identity( 3 )
        m_inv = - np.identity( 3 )
        rot_120 = Rotation.from_rotvec( self.rot_vec_111_2pi_3, degrees = False )
        rot_120m = Rotation.from_rotvec( - self.rot_vec_111_2pi_3, degrees = False )
        e = SymmetryOperationSO3( matrix = m0, dich_operations = {} )
        e_T = SymmetryOperationSO3( matrix = m0, dich_operations = {'T'} )
        inv = SymmetryOperationO3( matrix = m_inv, dich_operations = {} )
        inv_T = inv * e_T
        s_120 = SymmetryOperationO3( matrix = rot_120, dich_operations = {})
        s_120_inv = inv * s_120
        s_120_T = s_120 * e_T
        s_120_inv_T = inv * s_120_T
        s_120m = SymmetryOperationO3( matrix = rot_120m, dich_operations = {})
        s_120m_inv = inv * s_120m
        s_120m_T = s_120m * e_T
        s_120m_inv_T = inv * s_120m_T
        sg = SymmetryGroup( symmetry_operations = [ s_120, inv, e_T ] )
        self._compare_lists_of_sym_opers( sg.symmetry_operations, [e, e_T, inv, inv_T,
                                                                    s_120, s_120m,
                                                                    s_120_T, s_120m_T,
                                                                    s_120_inv, s_120m_inv,
                                                                    s_120_inv_T, s_120m_inv_T ] )
        self.assertEqual( sg.order(), 12 )
        sg = SymmetryGroup( symmetry_operations = [ e_T, s_120_inv ] )
        self._compare_lists_of_sym_opers( sg.symmetry_operations, [s_120m_inv, e, e_T, inv, inv_T,
                                                                    s_120, s_120m,
                                                                    s_120_T, s_120m_T,
                                                                    s_120_inv,
                                                                    s_120_inv_T, s_120m_inv_T ] )
        self.assertEqual( sg.order(), 12 )
        sg2 = SymmetryGroup( symmetry_operations = [ s_120_inv_T ] )
        self._compare_lists_of_sym_opers( sg2.symmetry_operations, [e, inv_T,
                                                                    s_120, s_120m,
                                                                    s_120_inv_T, s_120m_inv_T ] )
        self.assertEqual( sg2.order(), 6 )

    def test_symmetry_group_is_initialised_w_rotational_axis_6_dich( self ):
        #first, let's set the regular proper rotation
        # 60-degrees rotation around direction [111]
        m0 = np.identity( 3 )
        e = SymmetryOperationO3( matrix = m0 )
        e_T = SymmetryOperationSO3( matrix = m0, dich_operations = {'T'} )
        rot_60 = Rotation.from_rotvec( self.rot_vec_111_2pi_6, degrees = False )
        s_60 = SymmetryOperationO3( matrix = rot_60)
        s_60_T = s_60 * e_T
        sg = SymmetryGroup( symmetry_operations = [ s_60_T ] )
        # expected extra symmetry operations to be generated
        rot_minus60 = Rotation.from_rotvec( - self.rot_vec_111_2pi_6, degrees = False )
        s_minus_60 = SymmetryOperationO3( matrix = rot_minus60 )
        s_minus_60_T = e_T * s_minus_60
        rot_120 = Rotation.from_rotvec( self.rot_vec_111_2pi_3, degrees = False )
        s_120 = SymmetryOperationO3( matrix = rot_120)
        rot_minus120 = Rotation.from_rotvec( - self.rot_vec_111_2pi_3, degrees = False )
        s_minus_120 = SymmetryOperationO3( matrix = rot_minus120 )
        rot_180 = Rotation.from_rotvec( self.rot_vec_111_pi, degrees = False )
        s_180 = SymmetryOperationO3( matrix = rot_180)
        s_180_T = e_T * s_180
        expected_sym_opers = [s_60_T, s_minus_60_T, e,
                              s_120, s_minus_120,
                              s_180_T]
        self._compare_lists_of_sym_opers( sg.symmetry_operations, expected_sym_opers ) 
        self.assertEqual( sg.order(), 6 )

    def test_symmetry_group_is_initialised_w_rotational_axis_3( self ):
        #first, let's set the regular proper rotation
        # 120-degrees rotation around direction [111]
        m0 = np.identity( 3 )
        e = SymmetryOperation( matrix = m0 )
        rot_120 = Rotation.from_rotvec( self.rot_vec_111_2pi_3, degrees = False )
        s_120 = SymmetryOperation( matrix = rot_120)
        sg = SymmetryGroup( symmetry_operations = [ s_120 ] )
        # expected extra symmetry operations to be generated
        rot_minus120 = Rotation.from_rotvec( - self.rot_vec_111_2pi_3, degrees = False )
        s_minus_120 = SymmetryOperation( matrix = rot_minus120 )
        expected_sym_opers = [s_120, s_minus_120, e]
        self._compare_lists_of_sym_opers( sg.symmetry_operations, expected_sym_opers ) 
        self.assertEqual( sg.order(), 3 )

    def test_symmetry_group_is_initialized_w_errors(self):
        e = SymmetryOperation( matrix = np.identity( 3 ) )
        e_4d = SymmetryOperation( matrix = np.identity( 4 ) )
        # init w/ mismatched dimensions
        with self.assertRaises( ValueError ):
            sg = SymmetryGroup( symmetry_operations = [e, e_4d] )
        # init w/ a wrong types (non-symmetry operations)
        with self.assertRaises( TypeError ):
            sg = SymmetryGroup( symmetry_operations = [e, 6.3] )

    def test_add( self ):
        rot_120p = Rotation.from_rotvec(   self.rot_vec_111_2pi_3, degrees = False )
        rot_120m = Rotation.from_rotvec( - self.rot_vec_111_2pi_3, degrees = False )
        e = np.identity( 3 )
        inv = - np.identity( 3 )
        s_0 = SymmetryOperation( matrix = e)
        s_inv = SymmetryOperation( matrix = inv)
        s_rot_120p = SymmetryOperation( matrix = rot_120p)
        s_rot_120m = SymmetryOperation( matrix = rot_120m)
        s_rot_120p_inv = s_rot_120p * s_inv
        s_rot_120m_inv = s_inv * s_rot_120m
        sg = SymmetryGroup( symmetry_operations=[ s_rot_120p ] )
        sg.add_and_generate( s_inv )
        self._compare_lists_of_sym_opers( sg.symmetry_operations, [ s_0, s_inv, s_rot_120p, s_rot_120m,
                                                                    s_rot_120p_inv, s_rot_120m_inv ] )
        # adding a symmetry operation of the wrong dimensionality must raise an error
        s_0_4d = SymmetryOperation( matrix = np.identity( 4 ))
        with self.assertRaises( ValueError ):
            sg.add_and_generate( s_0_4d )

    def test_by_label( self ):
        m1 = np.identity( 2 )
        m2 = - np.identity( 2 )
        s0, s1 = SymmetryOperation( matrix = m1 ), \
                 SymmetryOperation( matrix = m2 )
        s0.label = 'A'
        s1.label = 'B'
        sg = SymmetryGroup( symmetry_operations=[ s0, s1 ] )
        self.assertEqual( sg.by_label( 'A' ), s0 )
        self.assertEqual( sg.by_label( 'B' ), s1 )
  
    def test_labels( self ):
        m1 = np.identity( 2 )
        m2 = - np.identity( 2 )
        s0, s1 = SymmetryOperation( matrix = m1 ), \
                 SymmetryOperation( matrix = m2 )
        s0.label = 'A'
        s1.label = 'B'
        sg = SymmetryGroup( symmetry_operations=[ s0, s1 ] )
        self.assertEqual( set( sg.labels ), { 'A', 'B' } )
    
    def test_repr_symmetry_group( self ):
        e = SymmetryOperationO3( )
        e_T = SymmetryOperationO3( dich_operations = 'T' )
        inv = SymmetryOperationO3( matrix = - np.identity(3) )
        sg = SymmetryGroup( symmetry_operations = [e_T, inv] )
        print_io = io.StringIO()
        print( sg, file = print_io)
        printed_str = print_io.getvalue()
        print_io.close()
        expected_str = '''\
SymmetryGroup
SymmetryOperation
label(---)
[[-1. -0. -0.]
 [-0. -1. -0.]
 [-0. -0. -1.]]
Dichromatic reversals: ['P']
SymmetryOperation
label(---)
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
Dichromatic reversals: ['T']
SymmetryOperation
label(E)
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
Dichromatic reversals: 
SymmetryOperation
label(---)
[[-1.  0.  0.]
 [ 0. -1.  0.]
 [ 0.  0. -1.]]
Dichromatic reversals: ['P', 'T']

'''.format(length='multi-line')
        self.assertEqual( printed_str, expected_str )
    
    def test_invariants_symmetry_group_axis_3( self ):
        #first, let's set the regular proper rotation
        # 120-degrees rotation around direction [111]
        rot_120 = Rotation.from_rotvec( self.rot_vec_111_2pi_3, degrees = False )
        s_120 = SymmetryOperationSO3( matrix = rot_120)
        sg = SymmetryGroup( symmetry_operations = [ s_120 ] )
        # now, let's test invariants
        pq_scalar = PhysicalQuantity( value = np.sqrt( 7 ) )
        self.assertTrue( sg.is_invariant( pq_scalar ) )
        pq_pseudoscalar = PhysicalQuantity( value = np.sqrt( 7 ), dich = { 'P':-1 } )
        self.assertTrue( sg.is_invariant( pq_pseudoscalar ) )
        pq_scalar = PhysicalQuantity( value = np.sqrt( 7 ) * np.identity( 3 ), dich = { 'P':1 } )
        self.assertTrue( sg.is_invariant( pq_scalar ) )
        pq_tensor = PhysicalQuantity( value = np.ones( ( 3, 3 ) ) )
        self.assertTrue( sg.is_invariant( pq_tensor ) )
        pq_tensor = PhysicalQuantity( value = self.tensor_0 )
        self.assertFalse( sg.is_invariant( pq_tensor ) )
        # time-odd polar vector
        pq_vector = PhysicalQuantity( value = [ 1, 1, 1 ], dich = { 'P':1, 'T':-1 })
        self.assertTrue( sg.is_invariant( pq_vector ) )
        # axial vector
        pq_axial_vector = PhysicalQuantity( value = [ 1, 1, 1 ], dich = { 'P':-1 } )
        self.assertTrue( sg.is_invariant( pq_axial_vector ) )
        pq_vector = PhysicalQuantity( value = [ 1, 1, -1 ] )
        self.assertFalse( sg.is_invariant( pq_vector ) )
        pq_vector = PhysicalQuantity( value = self.vector_0 )
        self.assertFalse( sg.is_invariant( pq_vector ) )
        # assuming tolerance atol = 1e-6
        pq_vector = PhysicalQuantity( value = [ 1 + 1e-10, 1, 1 ] )
        self.assertTrue( sg.is_invariant( pq_vector ) )
        # assuming tolerance atol = 1e-6
        pq_vector = PhysicalQuantity( value = [ 1 + 2e-5, 1, 1 ] )
        self.assertFalse( sg.is_invariant( pq_vector ) )

    def test_invariants_symmetry_group_axis_refl_6m_dich( self ):
        #first, let's set the regular proper rotation
        # 120-degrees rotation around direction [100]
        rot_120 = Rotation.from_rotvec( self.rot_vec_100_2pi_6, degrees = False )
        # rotation with time-reversal 6'
        s_120 = SymmetryOperationSO3( matrix = rot_120, dich_operations = { 'T' } )
        s_m001 = SymmetryOperationO3( matrix = self.m001)
        sg = SymmetryGroup( symmetry_operations = [ s_120, s_m001 ] )
        # now, let's test invariants
        # time-even scalar
        pq_scalar = PhysicalQuantity( value = np.sqrt( 7 ), dich = { 'P':1, 'T':1 } )
        self.assertTrue( sg.is_invariant( pq_scalar ) )
        # time-odd scalar
        pq_scalar = PhysicalQuantity( value = np.sqrt( 7 ), dich = { 'P':1, 'T':-1 } )
        self.assertFalse( sg.is_invariant( pq_scalar ) )
        # time-even pseudoscalar
        pq_pseudoscalar = PhysicalQuantity( value = np.sqrt( 7 ), dich = { 'P':-1, 'T':1 } )
        self.assertFalse( sg.is_invariant( pq_pseudoscalar ) )
        pq_scalar = PhysicalQuantity( value = np.sqrt( 7 ) * np.identity( 3 ), dich = { 'P':1, 'T':1 } )
        self.assertTrue( sg.is_invariant( pq_scalar ))
        pq_tensor = PhysicalQuantity( value = self.tensor_0 )
        self.assertFalse( sg.is_invariant( pq_tensor ) )
        # time-even polar vector
        pq_vector = PhysicalQuantity( value = [ 1, 0, 0 ] )
        self.assertTrue( sg.is_invariant( pq_vector ) )
        # time-odd polar vector
        pq_vector = PhysicalQuantity( value = [ 1, 0, 0 ], dich = { 'P':1, 'T':-1 } )
        self.assertFalse( sg.is_invariant( pq_vector ) )
        # time-even axial vector
        pq_axial_vector = PhysicalQuantity( value = [ 1, 0, 0 ], dich = { 'P':-1, 'T':1 } )
        self.assertFalse( sg.is_invariant( pq_axial_vector ) )
        pq_vector = PhysicalQuantity( value = [ 1, 1, 1 ] )
        self.assertFalse( sg.is_invariant( pq_vector ) )
        pq_vector = PhysicalQuantity( value = self.vector_0 )
        self.assertFalse( sg.is_invariant( pq_vector ) )
        # assuming tolerance atol = 1e-6
        pq_vector = PhysicalQuantity( value = [ 1 + 1e-10, 0, 0 ] )
        self.assertTrue( sg.is_invariant( pq_vector ) )
        # assuming tolerance atol = 1e-6
        pq_vector = PhysicalQuantity( value = [ 1, 2e-5, 0 ] )
        self.assertFalse( sg.is_invariant( pq_vector ) )

    def test_lim_symmetry_group_scalar(self):
        sg = LimitingSymmetryGroupScalar()
        # now, let's test invariants
        # time-even scalar
        pq_scalar = PhysicalQuantity( value = np.sqrt( 7 ), dich = { 'P':1, 'T':1 } )
        self.assertTrue( sg.is_invariant( pq_scalar ) )
        # time-odd scalar
        pq_scalar = PhysicalQuantity( value = np.sqrt( 7 ), dich = { 'P':1, 'T':-1 } )
        self.assertTrue( sg.is_invariant( pq_scalar ) )
        # time-even pseudoscalar
        pq_pseudoscalar = PhysicalQuantity( value = np.sqrt( 7 ), dich = { 'P':-1, 'T':1 } )
        self.assertTrue( sg.is_invariant( pq_pseudoscalar ) )
        pq_scalar = PhysicalQuantity( value = np.sqrt( 7 ) * np.identity( 3 ), dich = { 'P':1, 'T':1 } )
        self.assertTrue( sg.is_invariant( pq_scalar ))
        pq_tensor = PhysicalQuantity( value = self.tensor_0 )
        self.assertFalse( sg.is_invariant( pq_tensor ) )
        pq_vector = PhysicalQuantity( value = self.vector_0 )
        self.assertFalse( sg.is_invariant( pq_vector ) )

    def test_lim_symmetry_group_scalar_dich(self):
        e_T = SymmetryOperationSO3( matrix = np.identity( 3 ), dich_operations = {'T'} )
        inv = SymmetryOperationO3( matrix = - np.identity( 3 ), dich_operations = {} )
        sg = LimitingSymmetryGroupScalar( scalar_symmetry_operations = [ e_T, inv ] )
        # now, let's test invariants
        # time-even scalar
        pq_scalar = PhysicalQuantity( value = np.sqrt( 7 ), dich = { 'P':1, 'T':1 } )
        self.assertTrue( sg.is_invariant( pq_scalar ) )
        # time-odd scalar
        pq_scalar = PhysicalQuantity( value = np.sqrt( 7 ), dich = { 'P':1, 'T':-1 } )
        self.assertFalse( sg.is_invariant( pq_scalar ) )
        # time-odd pseudoscalar
        pq_pseudoscalar = PhysicalQuantity( value = np.sqrt( 7 ), dich = { 'P':-1, 'T':-1 } )
        self.assertFalse( sg.is_invariant( pq_pseudoscalar ) )
        # time-even pseudoscalar
        pq_pseudoscalar = PhysicalQuantity( value = np.sqrt( 7 ), dich = { 'P':-1, 'T':1 } )
        self.assertFalse( sg.is_invariant( pq_pseudoscalar ) )

        # space with PT-invariance
        inv_T = SymmetryOperationO3( matrix = - np.identity( 3 ), dich_operations = {'T'} )
        sg = LimitingSymmetryGroupScalar( scalar_symmetry_operations = [ inv_T ] )
         # time-odd pseudoscalar
        pq_pseudoscalar = PhysicalQuantity( value = np.sqrt( 7 ), dich = { 'P':-1, 'T':-1 } )
        self.assertTrue( sg.is_invariant( pq_pseudoscalar ) )
         # time-even pseudoscalar
        pq_pseudoscalar = PhysicalQuantity( value = np.sqrt( 7 ), dich = { 'P':-1, 'T':1 } )
        self.assertFalse( sg.is_invariant( pq_pseudoscalar ) )

    def test_repr_lim_scalar_symmetry_group( self ):
        inv_CT = SymmetryOperationO3( matrix = - np.identity( 3 ), dich_operations = { 'C', 'T' } )
        sg = LimitingSymmetryGroupScalar( scalar_symmetry_operations = [ inv_CT ] )
        print_io = io.StringIO()
        print( sg, file = print_io)
        printed_str = print_io.getvalue()
        print_io.close()
        #TODO to deduplicate
        expected_str = '''\
∞∞m*'
∞∞m*'
LimitingSymmetryGroupScalar
SymmetryOperation
label(E)
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
Dichromatic reversals: 
SymmetryOperation
label(---)
[[-1. -0. -0.]
 [-0. -1. -0.]
 [-0. -0. -1.]]
Dichromatic reversals: ['C', 'P', 'T']

'''.format(length='multi-line')
        self.assertEqual( printed_str, expected_str )

if __name__ == '__main__':
    unittest.main()
