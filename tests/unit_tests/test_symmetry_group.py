#
# Copyright (C) 2024 Dr. Bogdan Tanygin <info@deeptech.business>
# Copyright (C) 2015, 2021 Benjamin J. Morgan
#
# This file is part of spacetime-sym.
#

from math import pi, sqrt
import unittest
from spacetime import SymmetryGroup, SymmetryOperation, SymmetryOperationO3, SymmetryOperationSO3
from unittest.mock import Mock, MagicMock, patch, call
import numpy as np
from scipy.spatial.transform import Rotation
from copy import deepcopy

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
        # rotational vector along direction [111], mag = 2pi/3 rad
        self.rot_vec_111_2pi_3 = (2 * pi / 3) * np.array( [1, 1, 1] ) / sqrt(3)

    def _compare_lists_of_sym_opers( self, so_list, so_list_expected ):
        so_exp_matched = set()
        so_unexpected = set()
        so_txt = ''
        expected_so_txt = ''
        for so_exp in so_list_expected:
            expected_so_txt += '\n{}'.format( so_exp.matrix )
        matched_indxs_of_expected_so = set()
        indx_exp_so = set( range( len( so_list_expected ) ) )
        for so in so_list:
            so_txt += '\n{}'.format( so.matrix )
            so_matched_flag = False
            for i_exp in range( len( so_list_expected ) ):
                so_exp = so_list_expected[i_exp]
                if np.allclose( so.matrix, so_exp.matrix, atol = self.atol):
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
        unexpected_so_txt = ''
        for so_unexp in so_unexpected:
            unexpected_so_txt += '\n{}'.format( so_unexp.matrix )
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

    def test_symmetry_group_is_initialised_w_duplicated_opers( self ):
        m1 = np.identity( 2 )
        m2 = np.identity( 2 )
        s0, s1 = SymmetryOperation( matrix = m1, force_permutation = True ), \
                 SymmetryOperation( matrix = m2, force_permutation = True )
        sg = SymmetryGroup( symmetry_operations = [ s0, s1 ] )
        # after deduplication in the group
        self._compare_lists_of_sym_opers( sg.symmetry_operations, [s0] )

    def test_symmetry_group_is_initialised_w_different_opers( self ):
        m1 = np.identity( 2 )
        m2 = - np.identity( 2 )
        s0, s1 = SymmetryOperation( matrix = m1, force_permutation = False ), \
                 SymmetryOperation( matrix = m2, force_permutation = False )
        sg = SymmetryGroup( symmetry_operations = [ s0, s1 ] )
        # after deduplication in the group
        self._compare_lists_of_sym_opers( sg.symmetry_operations, [s1, s0] ) 

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

    #TODO same as above init -- with dich

    # Release Note: this functionality is deprecated in spacetime-sym
    #def test_read_from_file( self ):
    #    s0, s1 = Mock( spec=SymmetryOperation ), Mock( spec=SymmetryOperation )
    #    with patch( 'numpy.loadtxt' ) as mock_np_loadtxt:
    #        mock_np_loadtxt.return_value = np.array( [ [ 1, 2 ], [ 2, 1 ] ] )
    #        with patch( 'spacetime.symmetry_operation.SymmetryOperation.from_vector' ) as mock_from_vector:
    #            mock_from_vector.side_effect = [ s0, s1 ]
    #            sg = SymmetryGroup.read_from_file( 'mock_filename' )
    #            self.assertEqual( sg.symmetry_operations[0], s0 )
    #            self.assertEqual( sg.symmetry_operations[1], s1 )
    #            self.assertEqual( mock_from_vector.call_args_list[0], call( [ 1, 2 ] ) )
    #            self.assertEqual( mock_from_vector.call_args_list[1], call( [ 2, 1 ] ) )

    # Release Note: this functionality is deprecated in spacetime-sym
    #def test_read_from_file_with_labels( self ):
    #    s0, s1 = MagicMock( spec=SymmetryOperation ), MagicMock( spec=SymmetryOperation )
    #    with patch( 'numpy.genfromtxt' ) as mock_np_genfromtxt:
    #        mock_np_genfromtxt.return_value = np.array( [ [ 'E', '1', '2' ], [ 'C2', '2', '1' ] ] )
    #        with patch( 'spacetime.symmetry_operation.SymmetryOperation.from_vector' ) as mock_from_vector:
    #            mock_from_vector.side_effect = [ s0, s1 ]
    #            sg = SymmetryGroup.read_from_file_with_labels( 'mock_filename' )
    #           self.assertEqual( sg.symmetry_operations[0], s0 )
    #           self.assertEqual( sg.symmetry_operations[1], s1 )
    #           self.assertEqual( mock_from_vector.call_args_list[0], call( [ 1, 2 ] ) )
    #           self.assertEqual( mock_from_vector.call_args_list[1], call( [ 2, 1 ] ) )
    #           self.assertEqual( s0.set_label.call_args, call( 'E' ) )
    #           self.assertEqual( s1.set_label.call_args, call( 'C2' ) )

    # Release Note: this functionality is deprecated in spacetime-sym
    #def test_save_symmetry_operation_vectors_to( self ):
    #    s0 = SymmetryOperation.from_vector([ 1, 2 ])
    #    s1 = SymmetryOperation.from_vector([ 2, 1 ])
    #    sg = SymmetryGroup( symmetry_operations=[ s0, s1 ] )
    #    with patch( 'numpy.savetxt' ) as mock_savetxt:
    #        sg.save_symmetry_operation_vectors_to( 'filename' ) 
    #        self.assertEqual( mock_savetxt.call_args[0][0], 'filename' )
    #        np.testing.assert_array_equal( mock_savetxt.call_args[0][1], np.array( [ [ 2, 1 ], [ 1, 2 ] ] ) )

    #TODO (backlog) add & generate -- infinite group detection: docs. Test corresponding rotations w/ irrational angle

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
  
if __name__ == '__main__':
    unittest.main()
