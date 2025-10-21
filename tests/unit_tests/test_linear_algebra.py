#
# Copyright (C) 2024-2025 Dr. Bogdan Tanygin <info@deeptech.business>
# Copyright (C) 2015, 2021 Benjamin J. Morgan
#
# This file is part of spacetime-sym.
#

import unittest
from math import pi, sqrt
from scipy.spatial.transform import Rotation
from spacetime import linear_algebra as la
from spacetime.linear_algebra import is_square, is_diagonal, is_scalar, is_scalar_extended, \
                                     is_rotational_3D, is_rotational_proper_3D
import numpy as np

class GenericLinearAlgebraTestCase( unittest.TestCase ):
    """Tests for linear algebra routines"""

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
        self.list_0_proper = [[ 1, 0,                      0                    ],
                              [ 0, np.cos(self.angle_0), - np.sin(self.angle_0) ],
                              [ 0, np.sin(self.angle_0),   np.cos(self.angle_0) ]]
        # as advanced SciPy Rotation object
        self.rotation_0_proper = Rotation.from_matrix( self.list_0_proper )
        # improper rotational matrix (as a list object) to compare to
        self.list_0_improper = [[ - 1, 0,                      0                    ],
                                [ 0,   np.cos(self.angle_0), - np.sin(self.angle_0) ],
                                [ 0,   np.sin(self.angle_0),   np.cos(self.angle_0) ]]
        self.list_0_improper_2 = [[ np.cos(self.angle_0), 0, - np.sin(self.angle_0) ],
                                  [ 0                   , -1 , 0 ],
                                  [ np.sin(self.angle_0), 0. ,   np.cos(self.angle_0) ]]
        # rotational vector along direction [111], mag = 2pi/3 rad
        self.rot_vec_111_2pi_3 = (2 * pi / 3) * np.array( [1, 1, 1] ) / sqrt(3)
        self.rot_120_dir111 = Rotation.from_rotvec( self.rot_vec_111_2pi_3, degrees = False )
        # rotational vector along direction [100], mag = 2pi/6 rad
        self.rot_vec_100_2pi_6 = (2 * pi / 6) * np.array( [1, 0, 0] )
        self.rot_120_dir100 = Rotation.from_rotvec( self.rot_vec_100_2pi_6, degrees = False )
        # rotational vector along direction [111], mag = 2pi/6 rad
        self.rot_vec_111_2pi_6 = (2 * pi / 6) * np.array( [1, 1, 1] ) / sqrt(3)
        # reflection (mirror) plane (001)
        self.m001 = np.array( [ [  1, 0, 0 ],
                                 [ 0, 1, 0 ],
                                 [ 0, 0, -1 ] ] )
        # 4d matrix
        self.matrix4d = np.ones( (4,4) )
        # random tensor
        self.tensor_0_rnd = np.array( [ [ -1, np.sqrt( np.pi ), 1.3454675432e6 ],
                                    [ 743.3566, 0, -1.456 ],
                                    [ 1.4567, -877865, np.exp( -4 ) ] ] )

    def test_is_square_positive( self ):
        m = np.identity( 5 )
        self.assertEqual( is_square( m ), True )
        m = np.array( [ [ 3, 5, 2.7 ], 
                        [ -1, np.sqrt(3), 0 ],
                        [ 0, 0, 0 ]] )
        self.assertEqual( is_square( m ), True )
        m = np.zeros( (4, 4) )
        self.assertEqual( is_square( m ), True )
    
    def test_is_square_negative( self ):
        m = np.ones( (2, 3) )
        self.assertEqual( is_square( m ), False )
        m = np.array( [ [ 3, 2.7 ], 
                        [ -1, np.sqrt(3) ],
                        [ 0, 0 ]] )
        self.assertEqual( is_square( m ), False )

    def test_is_diagonal_positive( self ):
        m = np.zeros( (5, 5) )
        self.assertEqual( is_diagonal( m ), True )
        m = np.zeros( (5, 5) ) * np.sqrt( np.pi )
        self.assertEqual( is_diagonal( m ), True )
        m = np.identity( 5 )
        self.assertEqual( is_diagonal( m ), True )
        m = np.array( [ [ np.sqrt(3), 0, 0 ], 
                        [ 0, np.sqrt(3), 0 ],
                        [ 0, 0, np.sqrt(3) ]] )
        self.assertEqual( is_diagonal( m ), True )
        m = np.zeros( (4, 4) )
        self.assertEqual( is_diagonal( m ), True )

    def test_is_diagonal_negative( self ):
        m = np.ones( ( 5, 5 ) )
        self.assertEqual( is_diagonal( m ), False )
        m = np.array( [ [ np.sqrt(3), 0, 0 ], 
                        [ 0, np.sqrt(3), 0 ],
                        [ 1, 0, np.sqrt(3) ]] )
        self.assertEqual( is_diagonal( m ), False )
    
    def test_is_rotational_3D_positive( self ):
        self.assertEqual( is_rotational_3D( self.list_0_proper ), True )
        self.assertEqual( is_rotational_proper_3D( self.list_0_proper ), True )
        self.assertEqual( is_rotational_3D( self.list_0_improper_2 ), True )
        self.assertEqual( is_rotational_3D( self.rot_120_dir100.as_matrix() ), True )
        self.assertEqual( is_rotational_proper_3D( self.rot_120_dir100.as_matrix() ), True )
        self.assertEqual( is_rotational_3D( self.list_0_improper ), True )

    def test_is_rotational_3D_negative( self ):
        self.assertEqual( is_rotational_3D( self.matrix4d ), False )
        self.assertEqual( is_rotational_proper_3D( self.list_0_improper_2 ), False )
        self.assertEqual( is_rotational_3D( self.tensor_0_rnd ), False )

    def test_is_scalar_extended_positive( self ):
        m = np.zeros( ( 5, 5 ) ) * np.sqrt( np.pi )
        self.assertEqual( is_scalar_extended( m ), True )
        m = np.identity( 5 ) * np.sqrt( np.pi )
        self.assertEqual( is_scalar_extended( m ), True )
        m = np.identity( 1 ) * np.sqrt( np.pi )
        self.assertEqual( is_scalar_extended( m ), True )
        m = np.array( [ np.sqrt( np.pi ) ] )
        self.assertEqual( is_scalar_extended( m ), True )

    def test_is_scalar_extended_negative( self ):
        m = np.ones( ( 5, 5) ) * np.sqrt( np.pi )
        self.assertEqual( is_scalar_extended( m ), False )
        m = np.array( [ np.sqrt( np.pi ), 2 ])
        self.assertEqual( is_scalar_extended( m ), False )
        m = None
        self.assertEqual( is_scalar_extended( m ), False )
    
    def test_is_scalar_positive( self ):
        x = np.array( [ np.sqrt( np.pi ) ] )
        self.assertEqual( is_scalar( x ), True )
        x = np.sqrt( 7 )
        self.assertEqual( is_scalar( x ), True )
        x = 'X'
        self.assertEqual( is_scalar( x ), True )

    def test_is_scalar_negative( self ):
        x = np.array( [ np.sqrt( np.pi ), np.sqrt( np.pi ) ] )
        self.assertEqual( is_scalar( x ), False )
        x = ( 1, 2 )
        self.assertEqual( is_scalar( x ), False )
        x = None
        self.assertEqual( is_scalar( x ), False )

class PermutationsTestCase( unittest.TestCase ):
    """Tests for permutations functions"""

    def test_flatten_list( self ):
        l = [ [ 1, 2 ], [ 3, 4, 5 ] ]
        self.assertEqual( la.flatten_list( l ), [ 1, 2, 3, 4, 5 ] )

    def test_number_of_unique_permutations(self):
        a = [1,1,0,0]
        self.assertEqual( la.number_of_unique_permutations( a ), 6 )
        b = [1]*8 + [0]*8
        self.assertEqual( la.number_of_unique_permutations( b ), 12870 )
        c = [1,1,2,2,3,3]
        self.assertEqual( la.number_of_unique_permutations( c ), 90 )
  
    def test_unique_permuations( self ):
        all_permutations = [ [ 1, 1, 0, 0 ],
                             [ 1, 0, 1, 0 ],
                             [ 1, 0, 0, 1 ],
                             [ 0, 1, 1, 0 ],
                             [ 0, 1, 0, 1 ],
                             [ 0, 0, 1, 1 ] ]
        for p in all_permutations:
            unique_permutations = list( la.unique_permutations( p ) )
            self.assertEqual( len( all_permutations ), len( unique_permutations ) )
            # check that every list in all_permutations has been generated
            for p2 in all_permutations:
                self.assertEqual( p2 in unique_permutations, True )
 
if __name__ == '__main__':
    unittest.main()
