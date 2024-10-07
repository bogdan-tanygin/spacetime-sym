#
# Copyright (C) 2024 Dr. Bogdan Tanygin <info@deeptech.business>
# Copyright (C) 2015, 2021 Benjamin J. Morgan
#
# This file is part of spacetime-sym.
#

import numpy as np
from collections import Counter
from math import factorial
from functools import reduce
from operator import mul

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
        raise TypeError( 'Not a 2D array (matrix)' )
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

    Raise:
        TypeError: non-square type
    """
    # ensure the numpy type
    m = np.array( m )
    if not is_square( m ):
        raise TypeError( 'Not square matrix!' )
    # indices of nonzero values
    nonzero_inds = np.nonzero(m)
    for k in range( len( nonzero_inds[0] ) ):
        # check whether indices of non-diagonal elements exist
        if nonzero_inds[0][k] != nonzero_inds[1][k]:
            return False
    return True

def is_scalar( x ):
    """
    Test whether x is a scalar.

    Args:
        x (int|float|numpy.ndarray): an argument to test.

    Returns:
        (bool): True | False.
    """
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
    elif is_diagonal( x ) and all( x[i][i] == x[i + 1][i + 1] for i in range (x.ndim) ):
        if is_scalar( x[0,0] ):
            x = np.array( x[0,0] )
            return x
        else:
            raise TypeError( 'Not a scalar!' )
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

def unique_permutations( seq ):
    """
    Yield only unique permutations of seq in an efficient way.

    A python implementation of Knuth's "Algorithm L", also known from the 
    std::next_permutation function of C++, and as the permutation algorithm 
    of Narayana Pandita.
   
    see http://stackoverflow.com/questions/12836385/how-can-i-interleave-or-create-unique-permutations-of-two-stings-without-recurs/12837695
    """
    # Precalculate the indices we'll be iterating over for speed
    i_indices = range(len(seq) - 1, -1, -1)
    k_indices = i_indices[1:]
    # The algorithm specifies to start with a sorted version
    seq = sorted(seq)
    while True:
        #yield list( seq )
        yield list( seq )
        # Working backwards from the last-but-one index,           k
        # we find the index of the first decrease in value.  0 0 1 0 1 1 1 0
        for k in k_indices:
            if seq[k] < seq[k + 1]:
                break
        else:
            # Introducing the slightly unknown python for-else syntax:
            # else is executed only if the break statement was never reached.
            # If this is the case, seq is weakly decreasing, and we're done.
            return
        # Get item from sequence only once, for speed
        k_val = seq[k]
        # Working backwards starting with the last item,           k     i
        # find the first one greater than the one at k       0 0 1 0 1 1 1 0
        for i in i_indices:
            if k_val < seq[i]:
                break
        # Swap them in the most efficient way
        (seq[k], seq[i]) = (seq[i], seq[k])                #       k     i
                                                           # 0 0 1 1 1 1 0 0
        # Reverse the part after but not                           k
        # including k, also efficiently.                     0 0 1 1 0 0 1 1
        seq[k + 1:] = seq[-1:k:-1]

