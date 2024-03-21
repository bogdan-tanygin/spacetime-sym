import unittest
import numpy as np
from numpy.linalg import det
from scipy.spatial.transform import Rotation
from spacetime import SymmetryOperation, SymmetryOperationO3, Configuration
from spacetime import SymmetryOperationSO3
from unittest.mock import patch
import io
from spacetime.symmetry_operation import is_square, is_permutation_matrix

class SymmetryOperationTestCase( unittest.TestCase ):
    """Tests for symmetry operation functions"""

    def test_symmetry_operation_is_initialised_from_a_matrix( self ):
        matrix = np.matrix( [ [ 1, 0 ], [ 0, 1 ] ] )
        so = SymmetryOperation( matrix )
        np.testing.assert_array_equal( so.matrix, np.array(matrix) )

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

    def test_mul_with_configuration( self ):
        so = SymmetryOperation.from_vector( [ 2, 3, 1 ] )
        conf = Configuration( [ 1, 2, 3 ] )
        new_conf = so * conf
        self.assertEqual( type( new_conf ), Configuration )
        self.assertEqual( new_conf.matches( Configuration( [ 3, 1, 2 ] ) ), True )

    def test_mul_raises_TypeError_with_invalid_type( self ):
        so = SymmetryOperation.from_vector( [ 2, 3, 1 ] )
        with self.assertRaises( TypeError ):
            new_conf = so * 'foo'

    def test_invert( self ):
        matrix_a = np.array( [ [ 0, 1, 0 ], [ 0, 0, 1 ], [ 1, 0, 0 ] ] )
        matrix_b = np.array( [ [ 0, 0, 1 ], [ 1, 0, 0 ], [ 0, 1, 0 ] ] )
        so = SymmetryOperation( matrix_a )
        np.testing.assert_array_equal( so.invert().matrix, matrix_b )

    def test_invert_sets_label( self ):
        matrix_a = np.array( [ [ 0, 1, 0 ], [ 0, 0, 1 ], [ 1, 0, 0 ] ] )
        so = SymmetryOperation( matrix_a ).invert( label='A' )
        self.assertEqual( so.label, 'A' )
    
    def test_from_vector( self ):
        vector = [ 2, 3, 1 ]
        so = SymmetryOperation.from_vector( vector )
        np.testing.assert_array_equal( so.matrix, np.array( [ [ 0, 0, 1 ], [ 1, 0, 0 ], [ 0, 1, 0 ] ] ) )    

    def test_from_vector_with_label( self ):
        vector = [ 2, 3, 1 ]
        label = 'A'
        so = SymmetryOperation.from_vector( vector, label=label )
        np.testing.assert_array_equal( so.matrix, np.array( [ [ 0, 0, 1 ], [ 1, 0, 0 ], [ 0, 1, 0 ] ] ) )
        self.assertEqual( so.label, label )

    def test_symmetry_operation_is_initialised_with_label( self ):
        matrix = np.array( [ [ 1, 0 ], [ 0, 1 ] ] )
        label = 'E'
        so = SymmetryOperation( matrix, label=label )
        self.assertEqual( so.label, label )

    def test_from_vector_counting_from_zero( self ):
        vector = [ 1, 2, 0 ]
        so = SymmetryOperation.from_vector( vector, count_from_zero=True )
        np.testing.assert_array_equal( so.matrix, np.array( [ [ 0, 0, 1 ], [ 1, 0, 0 ], [ 0, 1, 0 ] ] ) )    

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

    def test_operate_on( self ):
        matrix = np.array( [ [ 0, 1, 0 ], [ 0, 0, 1 ], [ 1, 0, 0 ] ] )
        so = SymmetryOperation( matrix )
        configuration = Configuration( [ 1, 2, 3 ] )
        so.operate_on( configuration )
        np.testing.assert_array_equal( so.operate_on( configuration ).vector, np.array( [ 2, 3, 1 ] ) )  

    def test_operate_on_raises_TypeError_with_invalid_type( self ):
        matrix = np.array( [ [ 0, 1, 0 ], [ 0, 0, 1 ], [ 1, 0, 0 ] ] )
        so = SymmetryOperation( matrix )
        with self.assertRaises( TypeError ):
            so.operate_on( 'foo' )

    def test_character( self ):
        matrix = np.array( [ [ 1, 0 ], [ 0, 1 ] ] )
        so = SymmetryOperation( matrix )
        self.assertEqual( so.character(), 2 )

    def test_as_vector( self ):
        matrix = np.array( [ [ 0, 0, 1 ], [ 1, 0, 0 ], [ 0, 1, 0 ] ] )
        so = SymmetryOperation( matrix )
        self.assertEqual( so.as_vector(), [ 2, 3, 1 ] )
  
    def test_as_vector_counting_from_zero( self ):
        matrix = np.array( [ [ 1, 0 ], [ 0, 1 ] ] )
        so = SymmetryOperation( matrix )
        self.assertEqual( so.as_vector( count_from_zero=True ), [ 0, 1 ] )

    def test_se_label( self ):
        matrix = np.array( [ [ 1, 0 ], [ 0, 1 ] ] )
        so = SymmetryOperation( matrix )
        so.set_label( 'new_label' )
        self.assertEqual( so.label, 'new_label' )

    def test_pprint( self ):
        matrix = np.array( [ [ 1, 0 ], [ 0, 1 ] ] )
        so = SymmetryOperation( matrix )
        with patch( 'sys.stdout', new=io.StringIO() ) as mock_stdout:
            so.pprint()
            self.assertEqual( mock_stdout.getvalue(), '--- : 1 2\n' ) 

    def test_pprint_with_label( self ):
        matrix = np.array( [ [ 1, 0 ], [ 0, 1 ] ] )
        so = SymmetryOperation( matrix, label='L' )
        with patch( 'sys.stdout', new=io.StringIO() ) as mock_stdout:
            so.pprint()
            self.assertEqual( mock_stdout.getvalue(), 'L : 1 2\n' ) 

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
    """Tests for O(3) and SO(3) symmetry operation functions"""
    def setUp(self):
        # absolute tolerance
        self.atol = 1e-6
        # decimal power tolerance 10**(-ptol)
        self.ptol = 6
        # random scalar
        self.scalar_0 = - 2.57297 * np.sqrt(5.0)
        # rotational angle to compare to
        self.angle_0 = 2 * np.pi / np.sqrt(26.643)
        # rotational matrix (as a list object) to compare to
        self.list_0 = [[ 1, 0,                      0                    ],
                       [ 0, np.cos(self.angle_0), - np.sin(self.angle_0) ],
                       [ 0, np.sin(self.angle_0),   np.cos(self.angle_0) ]]
        self.matrix_0 = np.matrix( self.list_0 )
        # same as an NumPy array
        self.array_0  =   np.array( self.list_0 )
        # as advanced SciPy Rotation object
        self.rotation_0 = Rotation.from_matrix(self.list_0)

    def test_symmetry_operation_is_initialised_from_a_matrix( self ):
        so = SymmetryOperationO3( self.matrix_0 )
        np.testing.assert_allclose( so.matrix, self.matrix_0)

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
        rot_vec_0 = np.array( [0.4578, - 1.639, - 4.25] )
        rot_0 = Rotation.from_rotvec(rot_vec_0)
        rot_2 = Rotation.from_rotvec(2 * rot_vec_0)
        so_0 = SymmetryOperationO3( rot_0 )
        so_2 = SymmetryOperationO3( rot_2 )
        np.testing.assert_allclose( ( so_0 * so_0 ).matrix , so_2.matrix )

    def test_mul_raises_TypeError_with_invalid_type( self ):
        so = SymmetryOperationO3.from_vector( [ 2, 3, 1 ] )
        with self.assertRaises( TypeError ):
            new_so = so * 'foo'

    def test_invertO3( self ):
        matrix_a = self.array_0
        so = SymmetryOperationO3( matrix_a )
        so_inv = so.invertO3()
        np.testing.assert_allclose( (so * so_inv).matrix, np.identity( 3 ),
                                     atol = self.atol)

    def test_invert_sets_label( self ):
        matrix_a = np.array( [ [ 0, 1, 0 ], [ 0, 0, 1 ], [ 1, 0, 0 ] ] )
        so = SymmetryOperationO3( matrix_a ).invert( label='A' )
        self.assertEqual( so.label, 'A' )

    def test_symmetry_operation_is_initialised_with_label( self ):
        matrix = np.array( [ [ 1, 0 ], [ 0, 1 ] ] )
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

    def test_similarity_transform_with_label( self ):
        matrix_a = np.array( [ [ 0, 1, 0 ], [ 0, 0, 1 ], [ 1, 0, 0 ] ] )
        matrix_b = np.array( [ [ 1, 0, 0 ], [ 0, 0, 1 ], [ 0, 1, 0 ] ] )
        matrix_c = np.linalg.inv( matrix_a )
        so_a = SymmetryOperationO3( matrix_a )
        so_b = SymmetryOperationO3( matrix_b )
        label = 'foo'
        np.testing.assert_array_equal( so_a.similarity_transform( so_b, label=label ).label, label )

    def test_character( self ):
        matrix = np.array( [ [ 1, 0 ], [ 0, 1 ] ] )
        so = SymmetryOperationO3( matrix )
        self.assertEqual( so.character(), 2 )

    def test_as_vector( self ):
        matrix = np.array( [ [ 0, 0, 1 ], [ 1, 0, 0 ], [ 0, 1, 0 ] ] )
        so = SymmetryOperationO3( matrix )
        self.assertEqual( so.as_vector(), [ 2, 3, 1 ] )
  
    def test_as_vector_counting_from_zero( self ):
        matrix = np.array( [ [ 1, 0 ], [ 0, 1 ] ] )
        so = SymmetryOperationO3( matrix )
        self.assertEqual( so.as_vector( count_from_zero=True ), [ 0, 1 ] )

    def test_se_label( self ):
        matrix = np.array( [ [ 1, 0 ], [ 0, 1 ] ] )
        so = SymmetryOperationO3( matrix )
        so.set_label( 'new_label' )
        self.assertEqual( so.label, 'new_label' )

    def test_pprint( self ):
        matrix = np.array( [ [ 1, 0 ], [ 0, 1 ] ] )
        so = SymmetryOperationO3( matrix )
        with patch( 'sys.stdout', new=io.StringIO() ) as mock_stdout:
            so.pprint()
            self.assertEqual( mock_stdout.getvalue(), '--- : 1 2\n' ) 

    def test_pprint_with_label( self ):
        matrix = np.array( [ [ 1, 0 ], [ 0, 1 ] ] )
        so = SymmetryOperationO3( matrix, label='L' )
        with patch( 'sys.stdout', new=io.StringIO() ) as mock_stdout:
            so.pprint()
            self.assertEqual( mock_stdout.getvalue(), 'L : 1 2\n' ) 

    def test_repr( self ):
        matrix = np.array( [ [ 1, 0 ], [ 0, 1 ] ] )
        so = SymmetryOperationO3( matrix, label='L' )
        this_repr = so.__repr__()
        self.assertNotEqual( this_repr.find( 'L' ), 0 )
        self.assertNotEqual( this_repr.find( "[[1, 0],\n[0, 1]]" ), 0 )
    
    def test_ensure_unit_det_properO3( self ):
        #first, let's check the regular proper rotation
        so_1 = SymmetryOperationO3( self.rotation_0 )
        det_1 = det( so_1.matrix )
        np.testing.assert_approx_equal( det_1, 1, self.ptol )
        #now, let's modify the transformation matrix by a random coefficient
        with self.assertRaises( ValueError ):
            so_2 = SymmetryOperationO3 ( self.scalar_0 * self.array_0 )
    
    def test_ensure_unit_det_properSO3( self ):
        #first, let's check the regular proper rotation
        so_1 = SymmetryOperationSO3( self.rotation_0 )
        det_1 = det( so_1.matrix )
        np.testing.assert_approx_equal( det_1, 1, self.ptol )
        #now, let's modify the transformation matrix by a random coefficient
        with self.assertRaises( ValueError ):
            so_2 = SymmetryOperationO3 ( self.scalar_0 * self.array_0 )
    
    def test_ensure_unit_det_improper( self ):
        #first, let's check the regular improper rotation
        so_1 = SymmetryOperationO3( - 1 * self.array_0 )
        det_1 = det( so_1.matrix )
        np.testing.assert_approx_equal(det_1, - 1, self.ptol)
        #now, let's modify the transformation matrix by a random coefficient
        with self.assertRaises( ValueError ):
            so_2 = SymmetryOperationO3 ( self.scalar_0 * self.array_0 )
    
    def test_improper_flag( self ):
        #first, let's set the regular proper rotation
        so_1 = SymmetryOperationO3( self.rotation_0 )
        #then, let's set the regular improper rotation
        so_2 = SymmetryOperationO3( - 1 * self.array_0 )
        self.assertEqual(so_1.improper, False)
        self.assertEqual(so_2.improper, True)
        so_1.improper = True
        self.assertEqual(so_1.improper, True)
        with self.assertRaises( TypeError ):
            so_1.improper = 0
    
    def test_ensure_SO3_is_proper_rot( self ):
        with self.assertRaises( ValueError ):
            #the regular proper rotation class with wrong matrix
            so_1 = SymmetryOperationSO3( - 1 * self.array_0 )
        
if __name__ == '__main__':
    unittest.main()
