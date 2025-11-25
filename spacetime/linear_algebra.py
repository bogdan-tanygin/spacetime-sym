#
# Copyright (C) 2024-2025 Dr. Bogdan Tanygin <info@deeptech.business>
# Copyright (C) 2015, 2021 Benjamin J. Morgan
#
# This file is part of spacetime-sym.
#

import numpy as np
from numpy.linalg import det, norm
from scipy.linalg import issymmetric
from scipy.spatial.transform import Rotation
from collections import Counter
from math import factorial, pi, sqrt, asin, isclose
from functools import reduce
from operator import mul
from copy import deepcopy

def is_square( m ):
    """
    Test whether a numpy matrix is square.

    Args:
        m (numpy.matrix|numpy.ndarray|list): The matrix.

    Returns:
        (bool): True | False.

    Raise:
        TypeError: non-matrix type
    """
    # ensure the numpy type
    m = np.array( m )
    shape0 = m.shape
    if len( shape0 ) != 2:
        return False
    return all ( len ( rw ) == len ( m ) for rw in m )

def is_permutation_matrix( m ):
    """
    Test whether a numpy array is a `permutation matrix`_.

    .. _permutation_matrix: https://en.wikipedia.org/wiki/Permutation_matrix
    
    Args:
        m (mp.matrix): The matrix.

    Returns:
        (bool): True | False.
    """
    # ensure the numpy type
    m = np.array( m )
    return (m.ndim == 2 and m.shape[0] == m.shape[1] and
            (m.sum(axis=0) == 1).all() and 
            (m.sum(axis=1) == 1).all() and
            ((m == 1) | (m == 0)).all())

def is_diagonal( m ):
    """
    Test whether a numpy matrix is diagonal.

    Args:
        m (numpy.matrix|numpy.ndarray|list): The matrix.

    Returns:
        (bool): True | False.
    """
    # ensure the numpy type
    m = np.array( m )
    if not is_square( m ):
        return False
    # indices of nonzero values
    nonzero_inds = np.nonzero( m )
    if len( nonzero_inds[0] ) > 0:
        for k in range( len( nonzero_inds[0] ) ):
            # check whether indices of non-diagonal elements exist
            if nonzero_inds[0][k] != nonzero_inds[1][k]:
                return False
    return True

def is_rotational_3D( m, rtol = 1e-6 ):
    """
    Test whether a numpy matrix is rotational O(3) or SO(3).

    Args:
        m (numpy.matrix|numpy.ndarray|list): The matrix.
        rtol (float): relative tolerance for elements of the matrix/tensor

    Returns:
        (bool): True | False.
    """
    # ensure the numpy type
    m = np.array( m )
    if m.ndim == 2 and m.shape[0] == 3:
        identity_check = ( m.transpose() ).dot( m )
        expected = np.identity( m.shape[0] )
        det_check = det( m )
        if np.allclose( identity_check, expected, rtol = rtol ) and abs( abs( det_check ) - 1) < rtol:
            return True
        else:
            return False
    else:
        return False

def is_rotational_proper_3D( m, rtol = 1e-6 ):
    """
    Test whether a numpy matrix is proper SO(3).

    Args:
        m (numpy.matrix|numpy.ndarray|list): The matrix.
        rtol (float): relative tolerance for elements of the matrix/tensor

    Returns:
        (bool): True | False.
    """
    if not is_rotational_3D( m ):
        return False
    else:
        m = np.array( m )
        det_check = det( m )
        if abs( det_check - 1) < rtol:
            return True
        else:
            return False

def is_3D_vector( x ):
    """
    Test whether x is a 3D-space vector.

    Args:
        x (numpy.ndarray): an argument to test.

    Returns:
        (bool): True | False.
    """
    # ensure the numpy type
    x = np.array( x )
    if x.ndim == 1 and x.shape[0] == 3:
        return True
    else:
        return False

def is_symmetrical_tensor( x, rtol = 1e-6, atol = 1e-15 ):
    """
    Test whether x is a symmetrical tensor in 3D-space.

    Args:
        x (numpy.ndarray): an argument to test.
        rtol (float): relative tolerance for elements of the matrix/tensor
        atol (float): absolute tolerance for elements of the matrix/tensor

    Returns:
        (bool): True | False.
    """
    # ensure the numpy type
    x = np.array( x )
    return x.ndim == 2 and x.shape[0] == 3 and issymmetric( x, rtol = rtol, atol = atol )

def get_tensor_axis( x, rtol = 1e-6, atol = 1e-14 ):
    """
    Test whether x is an axial tensor in 3D-space. Return corresponding axis.

    Args:
        x (numpy.ndarray): an argument to test.
        rtol (float): relative tolerance for elements of the matrix/tensor
        atol (float): absolute tolerance for elements of the matrix/tensor

    Returns:
        (numpy.ndarray): its axis if axial. Otherwise, null-vector.
    """
    # ensure the numpy type
    x = np.array( x )
    eigvals, eigvecs_tmp = np.linalg.eigh( x )
    # represent the eigvecs as rows:
    eigvecs = eigvecs_tmp.transpose()
    eigvals_abs = np.zeros( 3 )
    for i in range( 3 ):
        eigvals_abs[ i ] = abs( eigvals[ i ] )
    max_val = np.max( eigvals_abs )
    dec_tol = int( np.log10( max_val * rtol ) )
    if dec_tol >= 0:
        dec_tol = 0
    qty_unique_vals = len( np.unique( np.round( eigvals, decimals = - dec_tol ) ) )
    # is the tensor axially symmetric?
    if qty_unique_vals == 2:
        three_indx = { 0, 1, 2 }
        for i in range( 2 ):
            for j in range( i + 1, 3 ):
                if isclose( eigvals[i], eigvals[j], rel_tol = rtol ):
                    # the remaining eigenvalue defines the axial vector of the given tensor
                    indx_axial = next( iter( three_indx - { i, j } ) )
                    tensor_axis = eigvecs[ indx_axial ]
    else:
        tensor_axis = np.zeros( 3 )
    return tensor_axis

def is_scalar( x ):
    """
    Test whether x is a scalar.

    Args:
        x (int|float|numpy.ndarray): an argument to test.

    Returns:
        (bool): True | False.
    """
    if x is None:
        return False
    # ensure the numpy type
    x = np.array( x )
    # both 0 or 1 dimension scalars
    # with unitary length in case of 1 dimension
    if x.ndim == 0:
        return True
    elif x.ndim == 1 and x.shape[0] == 1:
        return True
    else:
        return False

def is_scalar_extended( x, rtol = 1e-6 ):
    """
    Test whether x is a scalar.
    Including the case when x is a diagonal matrix representation of a scalar.

    Args:
        x (int|float|numpy.ndarray): an argument to test.
        rtol (float): relative tolerance for elements of the matrix/tensor

    Returns:
        (bool): True | False.
    """
    if x is None:
        return False
    # ensure the numpy type
    x = np.array( x )
    if is_scalar( x ):
        return True
    elif is_square( x ):
        if is_diagonal( x ):
            trace = np.trace( x )
            av_element = trace / x.shape[0]
            expected = np.identity( x.shape[0] ) * av_element
            return np.allclose( x, expected, rtol = rtol )
        else:
            return False
    else:
        return False

def make_0D_scalar( x ):
    """
    Transform scalar to the simplest 0-dimensional type.

    Args:
        x (numpy.matrix|numpy.ndarray|list|int|float): The scalar
        in different representations.

    Returns:
        (numpy.ndarray): 0-dimensional numpy array scalar.

    Raise:
        TypeError: non-scalar input
    """
    # ensure the numpy type
    x = np.array( x )
    if is_scalar( x ):
        if x.ndim == 1:
            # make 0D numpy array
            x = np.array( x[0] )
            return x
        else:
            return x
    # transform the diagonal matrix with constant diagonal elements into the 0D-scalar
    elif is_scalar_extended( x ):
        # all are equal, one returns the first diagonal elements
        return np.array( x[0,0] )
    #elif is_diagonal( x ):
    #    if all( x[i][i] == x[i + 1][i + 1] for i in range (x.shape[0]) ):
    #        if is_scalar( x[0,0] ):
    #            x = np.array( x[0,0] )
    #            return x
    #        else:
    #            raise TypeError( 'Not a scalar!' )
    #    else:
    #        raise TypeError( 'Not a scalar!' )
    else:
        raise TypeError( 'Not a scalar!' )

def flatten_list( this_list ):
    return [ item for sublist in this_list for item in sublist ]

def number_of_unique_permutations( seq ):
    """Calculate the number of unique permutations of a sequence seq.

    Args:
        seq (list): list of items.
        
    Returns:
        int: The number of unique permutations of seq
        
    """
    times_included = list( Counter( seq ).values() )
    factorials = list( map( factorial, times_included ) )
    return int( factorial( len( seq ) ) / reduce( mul, factorials ) )

def set_copy_assignment( value ):
        set_value = deepcopy( value )
        if isinstance( set_value, set ):
            # if the type is right
            return_set = set_value
        elif set_value is None:
            # empty set for None
            return_set = set()
        else:
            # in case it is a single-value reversal assignment
            # or another sequence type
            return_set = set( set_value )
        return return_set

def is_equal_2D( value_1, value_2 ):
    if not isinstance( value_1, np.ndarray ) or not isinstance( value_2, np.ndarray ):
        raise TypeError( 'numpy arrays are expected' )
    shape_1 = value_1.shape
    shape_2 = value_2.shape
    if shape_1 != shape_2:
        raise ValueError( 'numpy arrays must have the same shape and dimensions' )
    ndim = value_1.ndim
    if ndim == 0:
        eq_flag = ( value_1 == value_2 )
    elif ndim == 1:
        eq_flag = all( value_1[i] == value_2[i] for i in range( value_1.shape[0] ) )
    elif ndim == 2:
        eq_flag = all( ( value_1[i,j] == value_2[i,j] ) for i, j in np.ndindex( value_1.shape[0], value_1.shape[1] ) )
    else:
        raise ValueError('not implemented for higher dimensions as of now')
    return eq_flag

def rotation_matrix_based_on_vectors( v1, v2 ):
    """
    Calculate rotation matrix that connect 2 given vectors.

    Args:
        v1, v2 (numpy.matrix|numpy.ndarray|list): The 3D-vectors to be connected.

    Returns:
        (numpy.ndarray): rotation matrix.
    
    Raise:
        TypeError: non-3D-vector inputs
    """
    v1 = np.array( v1 )
    v2 = np.array( v2 )
    if not is_3D_vector( v1 ) or not is_3D_vector( v2 ):
        raise TypeError('Unsupported non-3D-vector inputs')
    c = np.cross( v1, v2 )
    theta = asin( norm( c ) / ( norm( v1 ) * norm( v2 ) ) )
    if np.dot( v1, v2 ) < 0:
        theta += ( pi / 2 - theta ) * 2
    rot = Rotation.from_rotvec( theta * c / norm( c ), degrees = False )
    return rot.as_matrix()
