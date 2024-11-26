#
# Copyright (C) 2024 Dr. Bogdan Tanygin <info@deeptech.business>
# Copyright (C) 2015, 2021 Benjamin J. Morgan
#
# This file is part of spacetime-sym.
#

import unittest
from spacetime import linear_algebra as la
from spacetime.linear_algebra import is_square, is_diagonal, is_scalar, is_scalar_extended
import numpy as np

class GenericLinearAlgebraTestCase( unittest.TestCase ):
    """Tests for linear algebra routines"""

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
