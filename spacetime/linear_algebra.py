#
# Copyright (C) 2024 Dr. Bogdan Tanygin <info@tanygin-holding.com>
# Copyright (C) 2015, 2021 Benjamin J. Morgan
#
# This file is part of spacetime-sym.
#
# Spacetime-sym is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Spacetime-sym is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

import numpy as np

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
