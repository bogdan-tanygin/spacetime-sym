#
# Copyright (C) 2024 Dr. Bogdan Tanygin <info@deeptech.business>
# Copyright (C) 2015, 2021 Benjamin J. Morgan
#
# This file is part of spacetime-sym.
#

import numpy as np
from numpy.linalg import det
from numpy import abs
from scipy.spatial.transform import Rotation
from copy import deepcopy

from spacetime.linear_algebra import is_square, is_permutation_matrix, is_scalar, make_0D_scalar
from spacetime.physical_quantity import PhysicalQuantity
  
class SymmetryOperation:
    """
    `SymmetryOperation` class.
    """

    def __init__( self, matrix, label=None, force_permutation = False ):
        """
        Initialise a `SymmetryOperation` object

        Args:
            matrix: square 2D vector as either a
            `numpy.matrix`, `numpy.ndarray`, `scipy.spatial.transform.Rotation` or `list`.
            for this symmetry operation.
            label (default=None) (str): optional string label for this `SymmetryOperation` object.
            force_permutation (default = True) (bool): whether permutation
            matrix is a requirement.

        Raises:
            TypeError: if matrix is not `numpy.matrix`, `numpy.ndarray`, or `list`.
            ValueError: if matrix is not square.
            ValueError: if matrix is not a `permutation matrix`_
              assuming force_permutation is kept True.

            .. _permutation_matrix: https://en.wikipedia.org/wiki/Permutation_matrix

        Notes:
            To construct a `SymmetryOperation` object from a vector of site mappings
            use the `SymmetryOperation.from_vector()` method.

        Returns:
            None
        """
        self._force_permutation = force_permutation
        self._matrix_check_and_assign( matrix )
        self.label = label
        self.index_mapping = np.array( [ np.array(row).tolist().index(1) for row in self.matrix 
                                         if 1 in np.array(row).tolist()] )
    
    def _matrix_check_and_assign( self, m0 ):
        """
        Checks the matrix associated with the symmetry transformation.
        Assigns the private property.

        Args:
            m: a square 2D vector as either a
            `numpy.ndarray`, `Rotation`, or `list`.
            for this symmetry operation.

        Raises:
            ValueError: if m is not a square matrix
            ValueError: if m is not a permutation matrix
            TypeError: unsupported matrix type

        Returns:
            None.
        """
        m = deepcopy( m0 )
        if isinstance( m, np.ndarray ):
            self._matrix = np.array( m )
        elif isinstance( m, list):
            self._matrix = np.array( m )
        elif isinstance( m, Rotation):
            self._matrix = m.as_matrix()
        else:
            raise TypeError
        if not is_square( self._matrix ):
            raise ValueError('Not a square matrix')
        if self._force_permutation and not is_permutation_matrix( self._matrix ):
            raise ValueError('Not a permutation matrix')

    @property
    def matrix( self ):
        """
        Matrix associated with the symmetry transformation.
        The basic class here defines it in a configurational space.
        See derived classes `SymmetryOperationO3` and `SymmetryOperationSO3` for Cartesian space.

        Args:
            None

        Returns:
            (numpy.array): matrix.
        """
        return self._matrix

    @matrix.setter
    def matrix( self, value ):
        """
        Matrix associated with the symmetry transformation.
        The basic class here defines it in a configurational space.
        See derived classes `SymmetryOperationO3` and `SymmetryOperationSO3` for Cartesian space.

        Args:
            value (numpy.matrix|numpy.ndarray|list): a square 2D vector as either a
            `numpy.matrix`, `numpy.ndarray`, or `list`.
            for this symmetry operation.

        Raises:
            None

        Returns:
            None.
        """
        self._matrix_check_and_assign( value )

    def __mul__( self, other ):
        """
        Multiply this `SymmetryOperation` matrix with another `SymmetryOperation`.

        Args:
            other (SymmetryOperation): the other symmetry operation
            for the matrix multiplication self * other.

        Raises:
            TypeError: in case of operands belong to incompatible types.

        Returns:
            (SymmetryOperation): a new `SymmetryOperation` instance with the resultant matrix.
        """
        if isinstance( other, SymmetryOperation ):
            return SymmetryOperation( self.matrix.dot( other.matrix ) )
        else:
            raise TypeError

    def invert( self, label=None ):
        """
        Invert this `SymmetryOperation` object.

        Args:
            None
 
        Returns:
            A new `SymmetryOperation` object corresponding to the inverse matrix operation.
        """
        return SymmetryOperation( np.linalg.inv( self.matrix ).astype( float ), label = label)

    @classmethod
    def from_vector( cls, vector, count_from_zero=False, label=None ):
        """
        Initialise a SymmetryOperation object from a vector of site mappings.

        Args:
            vector (list): vector of integers defining a symmetry operation mapping.
            count_from_zero (default = False) (bool): set to True if the site index counts from zero.
            label (default=None) (str): optional string label for this `SymmetryOperation` object.
   
        Returns:
            a new SymmetryOperation object
        """
        if not count_from_zero:
            vector = [ x - 1 for x in vector ]
        dim = len( vector )
        matrix = np.zeros( ( dim, dim ) )
        for index, element in enumerate( vector ):
            matrix[ element, index ] = 1
        new_symmetry_operation = cls( matrix, label=label )
        return new_symmetry_operation

    def similarity_transform( self, s, label=None ):
        """
        Generate the SymmetryOperation produced by a similarity transform S^{-1}.M.S

        Args:
            s: the symmetry operation or matrix S.
            label (:obj:`str`, optional): the label to assign to the new SymmetryOperation. Defaults to None.

        Returns:
            the SymmetryOperation produced by the similarity transform
        """
        s_new = s.invert() * ( self * s )
        if label:
            s_new.set_label( label )
        return s_new

    def character( self ):
        """
        Return the character of this symmetry operation (the trace of `self.matrix`).

        Args:
            none

        Returns:
            np.trace( self.matrix )
        """
        return np.trace( self.matrix )

    def as_vector( self, count_from_zero=False ):
        """
        Return a vector representation of this symmetry operation

        Args:
            count_from_zero (default = False) (bool): set to True if the vector representation counts from zero
      
        Returns:
            a vector representation of this symmetry operation (as a list)
        """
        offset = 0 if count_from_zero else 1
        return [ row.tolist().index( 1 ) + offset for row in self.matrix.T ]

    def set_label( self, label ):
        """
        Set the label for this symmetry operation.
  
        Args:
            label: label to set for this symmetry operation
        Returns:
            self 
        """
        self.label = label
        return self

    def pprint( self ):
        """
        Pretty print for this symmetry operation

        Args:
            None
        Returns:
            None
        """
        label = self.label if self.label else '---'
        print( label + ' : ' + ' '.join( [ str(e) for e in self.as_vector() ] ) )

    def __repr__( self ):
        label = self.label if self.label else '---'
        return 'SymmetryOperation\nlabel(' + label + ")\n" + self.matrix.__repr__()

class SymmetryOperationO3(SymmetryOperation):
    """
    `SymmetryOperationO3` class.
    """
    def __init__( self, matrix = np.identity( 3 ), dich_operations = set(),
                  label = None, force_permutation = False):
        """
        Initialise a `SymmetryOperationO3` object, that contains a symmetry
        transformation of O(3) group of proper and improper rotations.
        The latter implies the chirality-changing reflection/inversion
        transformation. This class supports Euclidean space only.

        Args:
            matrix (numpy.matrix|numpy.ndarray|list): square 2D array as either a
            `numpy.matrix`, `numpy.ndarray`, or `list` for this symmetry operation.
            The default is an identity matrix.
            dich_operations (default = {}): a set of dichromatic symmetry reversal
            operations marked by string names.
            label (default = None) (str): optional string label for this object.
            force_permutation (default = False) (bool): whether permutation
            matrix is a requirement. It is not for an Euclidean space matrices.
        Raises:
            None
        Returns:
            None
        """
        super( SymmetryOperationO3, self ).__init__( matrix = matrix, label = label,
                                                     force_permutation = force_permutation )
        self._dich_operations = deepcopy( dich_operations )
        self._det_check_and_init( matrix = self._matrix )

    def _det_check_and_init( self, matrix, det_rtol = 1e-6 ):
        """
        Checks whether matrix determinant has unitary modulus.
        Adds improper rotation flag and parity transformation when the determinant is -1.

        Args:
            matrix (numpy.matrix|numpy.ndarray|list): square 2D array as either a
            `numpy.matrix`, `numpy.ndarray`, or `list` for this symmetry operation.
            det_rtol (default = 1e-6) (float): determinant check relative tolerance.
        Raises:
            ValueError: if determinant's magnitude is not unitary
        Returns:
            None
        """
        det_check = det( matrix )
        # assuming unitary magnitude
        if abs( abs( det_check ) - 1) > det_rtol:
            raise ValueError('Not a rotation matrix')
        # set the improper location flag
        if det_check < 0:
            # the symmetry operation belongs to O(3)
            self._improper = True
            # automatic setting of the space parity
            self._dich_operations.add('P')
        else:
            # the symmetry operation belongs to SO(3)
            self._improper = False
            # automatic setting of the space parity
            if 'P' in self._dich_operations:
                self._dich_operations.remove('P')

    def invert( self, label=None ):
        """
        Invert this `SymmetryOperationO3` object.

        Args:
            None
 
        Returns:
            A new `SymmetryOperationO3` object corresponding to the inverse matrix operation.
        """
        so = super( SymmetryOperationO3, self ).invert( label = label )
        #dich operations are applied same way both directions
        so.dich_operations = self._dich_operations
        return so

    @property
    def matrix( self ):
        """
        Proper or improper rotational matrix associated with the symmetry transformation.

        Args:
            None

        Returns:
            (numpy.array): matrix.
        """
        return self._matrix
    
    @matrix.setter
    def matrix( self, value ):
        """
        Proper or improper rotational matrix associated with the symmetry transformation.

        Args:
            value (numpy.matrix|numpy.ndarray|list): a square 2D vector as either a
            `numpy.matrix`, `numpy.ndarray`, or `list`.
            for this symmetry operation.

        Raises:
            None

        Returns:
            None.
        """
        self._det_check_and_init( matrix = value )
        self._matrix_check_and_assign( value )

    @property
    def improper( self ):
        """
        Improper rotation flag of :any:`SymmetryOperationO3`.
        A read-only property: it is set automatically by the improper
        rotation matrix definition

        Args:
            None

        Returns:
            (bool): True | False.
        """
        return self._improper
    
    @property
    def dich_operations( self ):
        """
        A set of dichromatic symmetry reversal operations.

        Args:
            None

        Returns:
            (set):  {'dich_label1', 'dich_label1', ...}.
        """
        return self._dich_operations
    
    @dich_operations.setter
    def dich_operations( self, value ):
        """
        A setter for the set of dichromatic symmetry reversal operations.
        Each operation will be applied to a physical value depending on its
        dichromatic symmetry properties.

        Args:
            value (set): {'dich_label1', 'dich_label1', ...}.
        
        Raises:
            TypeError: if labels are not strings

        Returns:
            None
        """
        if isinstance( value, set ):
            # if the type is right
            dich_operations = value
        elif value is None:
            # empty set for None
            dich_operations = set()
        else:
            # in case it is a single-value reversal assignment
            # or another sequence type
            dich_operations = set( value )
        # let's validate its elements type
        for dich_label in dich_operations:
            if not isinstance( dich_label, str ):
                raise TypeError( 'Dichromatic symmetry reversal operation label must be string' )
        # to avoid reassignment of the read-only (automatic) space parity
        if self._improper:
            dich_operations.add('P')
        elif 'P' in dich_operations:
            dich_operations.remove('P')
        self._dich_operations = deepcopy( dich_operations )

    def __mul__( self, other ):
        """
        Multiply this `SymmetryOperationO3` matrix with
        other matrices or vectors.

        Args:
            (option 1)
            other (SymmetryOperationO3):
            other symmetry operation for the matrix multiplication
            self * other. Derived classes are also supported, e.g. (SymmetryOperationSO3)
            (option 2)
            other (PhysicalQuantity): a physical quantity of the proper
            dimension to act on.

        Returns:
            (option 1)
            (SymmetryOperationO3): a new symmetry
            operation instance with the resultant matrix.
            (option 2)
            (PhysicalQuantity): a new physical quantity after operating
            the symmetry operation on it.
        """
        if type( other ) is SymmetryOperationO3:
            # symmetric difference to exclude compensating dich operations
            do_united = self._dich_operations ^ other.dich_operations
            return SymmetryOperationO3( matrix = self._matrix.dot( other.matrix ),
                                        dich_operations = do_united )
        elif type( other ) is SymmetryOperation:
            do_united = self._dich_operations
            return SymmetryOperationO3( matrix = self._matrix.dot( other.matrix ),
                                         dich_operations = do_united )
        elif isinstance( other, PhysicalQuantity ):
            # Dot-multiplaying (acting on) the PhysicalQuantity
            return self.operate_on( other )
        else:
            raise TypeError

    def operate_on( self, pq ):
        """
        Return the PhysicalQuantity (scalar or vector) 
        generated by applying this symmetry operation

        Args:
            pq (PhysicalQuantity): the physical quantity to operate on

        Returns:
            (PhysicalQuantity): the new physical quantity obtained by
            operating on it with this symmetry operation. 
        
        Raises:
            TypeError: if the operand is not a PhysicalQuantity or its dimension
            doesn't match the operator's one
            ValueError: if the operand does not have the dichromatic symmetry specification
            required for the operator's action
        """
        if not isinstance( pq, PhysicalQuantity ):
            raise TypeError( 'Not a PhysicalQuantity' )
        else:
            # a physical quantity after this symmetry transformation is applied
            pq_res = deepcopy(pq)
        if is_scalar( pq.value ):
            # physical scalars (including pseudoscalars) will not be changed
            # by O3 operations here
            # until dich properties are accounted for below
            new_value = deepcopy(pq.value)
        else:
            # all other cases -- vectors and tensors
            # let's make a basic check of dimensions
            if self.matrix.shape[1] != pq.value.shape[0]:
                raise TypeError( 'Mismatch of dimensions' )
            # regular matrix to vector operating
            if pq.value.ndim == 1:
                new_value = deepcopy( self.matrix.dot( pq.value ) )
            # physical tensors
            else:
                # check the remaining dimensions
                if self.matrix.shape[0] != pq.value.shape[1]:
                    raise TypeError( 'Mismatch of dimensions' )
                # matrix transformation acting on tensor T:
                # M T M^-1
                new_value = deepcopy( ( self.matrix.dot( pq.value ) ).dot( ( self.invert() ).matrix ) )
        for dich_oper in self._dich_operations:
            if dich_oper in pq.dich.keys():
                # keep or reverse the physical quantity's value for each
                # dichromatic symmetry reversal operation depending on its
                # symmetry properties
                new_value *= pq.dich[dich_oper]
            else:
                raise ValueError( 'Physical quantity does not have the dichromatic symmetry specified: {}'.format(dich_oper) )
        pq_res.value = new_value
        return pq_res

class SymmetryOperationSO3(SymmetryOperationO3):
    """
    `SymmetryOperationSO3` class.
    """
    def __init__( self, matrix = np.identity( 3 ), dich_operations = set(),
                  label = None, force_permutation = False):
        """
        Initialise a `SymmetryOperationSO3` object, that contains a symmetry
        transformation of SO(3) group of proper rotations.
        This class supports Euclidean space only.

        Args:
            matrix (numpy.matrix|numpy.ndarray|list): square 2D array as either a
            `numpy.matrix`, `numpy.ndarray`, or `list` for this symmetry operation.
            The default is an identity matrix.
            dich_operations (default = {}): a set of dichromatic symmetry reversal
            operations marked by string names.
            label (default=None) (str): optional string label for this object.
            force_permutation (default = False) (bool): whether permutation
            matrix is a requirement. It is not for an Euclidean space matrices.
        Raises:
            None

        Returns:
            None
        """
        super(SymmetryOperationSO3, self).__init__(matrix = matrix, dich_operations = dich_operations,
                                                   label = label, force_permutation = force_permutation)
        self._ensure_proper_rotation( matrix = self._matrix )

    def __mul__( self, other ):
        """
        Multiply this `SymmetryOperationSO3` matrix with
        other matrices or vectors.

        Args:
            (option 1)
            other (SymmetryOperationO3) or (SymmetryOperationSO3):
            other symmetry operation for the matrix multiplication
            self * other.
            (option 2)
            other (PhysicalQuantity): a physical quantity of the proper
            dimension to act on.

        Returns:
            (option 1)
            (SymmetryOperationO3) or (SymmetryOperationSO3): a new symmetry
            operation instance with the resultant matrix. If operand has
            the type SymmetryOperationO3, the return is always SymmetryOperationO3.
            (option 2)
            (PhysicalQuantity): a new physical quantity after operating
            the symmetry operation on it.
        """
        if type( other ) is SymmetryOperationO3:
            # Multiplying proper and improper rotation might give an improper one
            do_united = self._dich_operations ^ other.dich_operations
            return SymmetryOperationO3( matrix = self._matrix.dot( other.matrix ),
                                        dich_operations = do_united )
        elif type( other ) is SymmetryOperationSO3:
            # Multiplying proper rotations gives a proper one
            do_united = self._dich_operations ^ other.dich_operations
            return SymmetryOperationSO3( matrix = self._matrix.dot( other.matrix ),
                                         dich_operations = do_united )
        elif type( other ) is SymmetryOperation:
            # Multiplying proper and unknown rotation might give an improper one
            do_united = self._dich_operations
            return SymmetryOperationO3( matrix = self._matrix.dot( other.matrix ),
                                         dich_operations = do_united )
        elif isinstance( other, PhysicalQuantity ):
            # Dot-multiplaying (acting on) the PhysicalQuantity
            return self.operate_on( other )
        else:
            raise TypeError
    
    @property
    def matrix( self ):
        """
        Proper rotational matrix associated with the symmetry transformation.

        Args:
            None

        Returns:
            (numpy.array): matrix.
        """
        return self._matrix
    
    @matrix.setter
    def matrix( self, value ):
        """
        Proper rotational matrix associated with the symmetry transformation.

        Args:
            value (numpy.matrix|numpy.ndarray|list): a square 2D vector as either a
            `numpy.matrix`, `numpy.ndarray`, or `list`.
            for this symmetry operation.

        Raises:
            None

        Returns:
            None.
        """
        self._ensure_proper_rotation( matrix = value )
        self._matrix_check_and_assign( value )

    def _ensure_proper_rotation( self, matrix, det_rtol = 1e-6 ):
        """
        Checks whether matrix determinant has unitary value.
        Adds the proper rotation flag and even parity transformation.

        Args:
            matrix (numpy.matrix|numpy.ndarray|list): square 2D array as either a
            `scipy.spatial.transform.Rotation`, `numpy.matrix`, `numpy.ndarray`, 
            or `list` for this symmetry operation.
            det_rtol (default = 1e-6) (float): determinant check relative tolerance.
        Raises:
            ValueError: if determinant is not unitary
        Returns:
            None
        """
        det_check = det( matrix )
        # assuming unitary value
        if abs(det_check - 1) > det_rtol:
            raise ValueError('Not a proper rotation matrix')
        else:
            self._det_check_and_init( matrix = matrix )
