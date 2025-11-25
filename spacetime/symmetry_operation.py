#
# Copyright (C) 2024 Dr. Bogdan Tanygin <info@deeptech.business>
# Copyright (C) 2015, 2021 Benjamin J. Morgan
#
# This file is part of spacetime-sym.
#

import numpy as np
from numpy.linalg import det, norm
from numpy import abs
from scipy.spatial.transform import Rotation
from copy import deepcopy

from spacetime.linear_algebra import is_square, is_permutation_matrix, is_scalar, is_scalar_extended,\
                                     set_copy_assignment, is_rotational_3D, is_rotational_proper_3D
from spacetime.physical_quantity import PhysicalQuantity

class SymmetryOperation:
    """
    `SymmetryOperation` class.
    """
    def __init__( self, matrix = np.identity( 2 ), label = "", force_permutation = False,
                  matrix_precision = 0 ):
        """
        Initialise a `SymmetryOperation` object

        Args:
            matrix: square 2D vector as either a
            `numpy.ndarray`, `scipy.spatial.transform.Rotation` or `list`.
            for this symmetry operation.
            label (default = None) (str): optional string label for this `SymmetryOperation` object.
            force_permutation (default = True) (bool): whether permutation
            matrix is a requirement.
            matrix_precision (defaul = 0): minimal magnitude of the matrix' elements

        Raises:
            TypeError: if matrix is not of the expected type.
            ValueError: if matrix is not square.
            ValueError: if matrix is not a `permutation matrix`_
              assuming force_permutation is kept True.

        Returns:
            None
        """
        self._force_permutation = force_permutation
        self._matrix_check_and_assign( matrix )
        self.label = label
        self.matrix_precision = matrix_precision
    
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
    def label( self ):
        """
        Label of the symmetry transformation.

        Args:
            None.

        Returns:
            (str): label.
        """
        return self._label

    @label.setter
    def label( self, value ):
        """
        Label of the symmetry transformation.

        Args:
            label (str): optional string label for this `SymmetryOperation` object.

        Returns:
            None.
        """
        if not isinstance( value, str):
            raise ValueError('Label must be a string')
        else:
            self._label = value

    @property
    def matrix( self ):
        """
        Matrix associated with the symmetry transformation.
        The basic class here defines it in a general configurational space.
        See derived classes `SymmetryOperationO3` and `SymmetryOperationSO3` for Euclidean 3D space.

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
        See derived classes `SymmetryOperationO3` and `SymmetryOperationSO3` for Euclidean 3D space.

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

    def invert( self, label = "" ):
        """
        Invert this `SymmetryOperation` object.

        Args:
            None
 
        Returns:
            A new `SymmetryOperation` object corresponding to the inverse matrix operation.
        """
        return SymmetryOperation( np.linalg.inv( self.matrix ).astype( float ), label = label)

    def similarity_transform( self, s, label = "" ):
        """
        Generate the SymmetryOperation produced by a similarity transform S^{-1}.M.S

        Args:
            s: the symmetry operation or matrix S.
            label (:obj:`str`, optional): the label to assign to the new SymmetryOperation. Defaults to "".

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

    def __repr__( self ):
        label = self.label if self.label else '---'
        output_matrix = deepcopy( self.matrix )
        shape_0 = self.matrix.shape
        for i,j in np.ndindex( shape_0 ):
            if norm( output_matrix[i,j] ) <= self.matrix_precision:
                output_matrix[i,j] = 0
        output = 'SymmetryOperation\nlabel(' + label + ")\n{}".format(output_matrix)
        return output

class SymmetryOperationO3(SymmetryOperation):
    """
    `SymmetryOperationO3` class.
    """
    def __init__( self, matrix = np.identity( 3 ), dich_operations = set(),
                  label = "", force_permutation = False, matrix_precision = 1e-14):
        """
        Initialise a `SymmetryOperationO3` object, that contains a symmetry
        transformation of O(3) group of proper and improper rotations.
        The latter implies the chirality-changing reflection/inversion
        transformation. This class supports Euclidean space only.

        Args:
            matrix (numpy.matrix|numpy.ndarray|list): square 2D array as either a
            `numpy.ndarray`, or `list` for this symmetry operation.
            The default is an identity matrix.
            dich_operations (default = {}): a set of dichromatic symmetry reversal
            operations marked by string names.
            label (default = "") (str): optional string label for this object.
            force_permutation (default = False) (bool): whether permutation
            matrix is a requirement. It is not for an Euclidean space matrices.
            matrix_precision (defaul = 1e-14): minimal magnitude of the matrix' elements
        Raises:
            None
        Returns:
            None
        """
        super( SymmetryOperationO3, self ).__init__( matrix = matrix, label = label,
                                                     force_permutation = force_permutation )
        self.dich_operations = dich_operations
        self._det_check_and_init( matrix = self._matrix )
        self.matrix_precision = matrix_precision

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
        if not is_rotational_3D( matrix, rtol = det_rtol * 1):
            raise ValueError('Not a rotation matrix')
        # set the improper location flag
        if not is_rotational_proper_3D( matrix, rtol = det_rtol * 1):
            # the symmetry operation belongs to O(3)
            self._improper = True
            # automatic setting of the space parity
            self._dich_operations.add('P')
        else:
            # the symmetry operation belongs to SO(3)
            self._improper = False
            # automatic setting of the space parity
            if 'P' in self._dich_operations:
                raise ValueError('Mismatch: proper rotation matrix and parity reversal among dichromatic reversals')

    def invert( self, label = "" ):
        """
        Invert this `SymmetryOperationO3` object.

        Args:
            None
 
        Returns:
            A new `SymmetryOperationO3` object corresponding to the inverse matrix operation.
        """
        #dich operations are applied same way both directions
        so = SymmetryOperationO3( np.linalg.inv( self.matrix ).astype( float ), dich_operations = self._dich_operations, label = label)
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
            TypeError: if the input is not a set

        Returns:
            None
        """
        self._dich_operations_setter_impl( value )

    def _dich_operations_setter_impl( self, value ):
        dich_oper = set_copy_assignment( value )
        if not isinstance( dich_oper, set ):
            raise TypeError('Must be a set of dichromatic reversal string labels')
        # let's validate its elements type
        for dich_label in dich_oper:
            if not isinstance( dich_label, str ):
                raise TypeError( 'Dichromatic symmetry reversal operation label must be string' )
        # to avoid reassignment of the read-only (automatic) space parity
        if hasattr( self, '_improper' ): # avoid at the init stage, it's handled by _det_check_and_init()
            if self._improper:
                dich_oper.add('P')
            elif 'P' in dich_oper:
                raise ValueError('Mismatch: proper rotation matrix and parity reversal among dichromatic reversals')
        self._dich_operations = dich_oper

    def add_dich_operations( self, new_dich_operations ):
        """
        A setter to add extra dichromatic symmetry reversal operations.

        Args:
            value (set): {'dich_label1', 'dich_label1', ...}.
        
        Raises:
            TypeError: if labels are not strings

        Returns:
            None
        """
        new_dich_operations = set_copy_assignment( new_dich_operations )
        dich_operations = self.dich_operations
        dich_operations |= new_dich_operations
        self.dich_operations = dich_operations

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
        if isinstance( other, SymmetryOperationO3 ):
            # symmetric difference to exclude compensating dich operations
            do_united = self._dich_operations ^ other.dich_operations
            return SymmetryOperationO3( matrix = self._matrix.dot( other.matrix ),
                                        dich_operations = do_united )
            # one returns a more complex class
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
        if is_scalar_extended( pq.value ):
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
    
    def _dich_output( self ):
        if len( self.dich_operations ) > 0:
            dich_oper_print = self.dich_operations
            dich_oper_print = list( dich_oper_print )
            dich_oper_print.sort()
        else: dich_oper_print = ''
        return '\nDichromatic reversals: {}'.format( dich_oper_print )

    def __repr__( self ):
        output = super(SymmetryOperationO3, self).__repr__()
        output += self._dich_output()
        return output

class SymmetryOperationSO3(SymmetryOperationO3):
    """
    `SymmetryOperationSO3` class.
    """
    def __init__( self, matrix = np.identity( 3 ), dich_operations = set(),
                  label = "", force_permutation = False):
        """
        Initialise a `SymmetryOperationSO3` object, that contains a symmetry
        transformation of SO(3) group of proper rotations.
        This class supports Euclidean space only.

        Args:
            matrix (numpy.matrix|numpy.ndarray|list|Rotation): square 2D array as either a
            `numpy.matrix`, `numpy.ndarray`, or `list` for this symmetry operation.
            The default is an identity matrix.
            dich_operations (default = {}): a set of dichromatic symmetry reversal
            operations marked by string names. The spatial inversion 'P'
            cannot be among them.
            label (default = "") (str): optional string label for this object.
            force_permutation (default = False) (bool): whether permutation
            matrix is a requirement. It is not for an Euclidean space matrices.
        Raises:
            ValueError: if spatial inversion 'P' presents among dichromatic reversals

        Returns:
            None
        """
        super(SymmetryOperationSO3, self).__init__(matrix = matrix, dich_operations = dich_operations,
                                                   label = label, force_permutation = force_permutation)
        if 'P' in dich_operations:
            raise ValueError('SO3 cannot be an improper rotation')
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
            raise TypeError('Unsupported class of an operand')

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

    def invert( self, label = "" ):
        """
        Invert this `SymmetryOperationSO3` object.

        Args:
            None
 
        Returns:
            A new `SymmetryOperationSO3` object corresponding to the inverse matrix operation.
        """
        #dich operations are applied same way both directions
        so = SymmetryOperationSO3( np.linalg.inv( self.matrix ).astype( float ), dich_operations = self._dich_operations, label = label)
        return so

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
        # assuming unitary value
        if not is_rotational_proper_3D( matrix, rtol = det_rtol * 1 ):
            raise ValueError('Not a proper rotation matrix')
        else:
            self._det_check_and_init( matrix = matrix )

class LimitingSymmetryOperationO3(SymmetryOperationO3):
    """
    `LimitingSymmetryOperationO3` class.
    """
    def __init__( self, axis = [ 0, 0, 1 ], dich_operations = set()):
        """
        Initialise a `LimitingSymmetryOperationO3` object, that represents a symmetry
        transformation of limiting Curie subgroup of O(3) group of proper and
        improper rotations. It is based on the infinity-fold rotation axis.
        As a subclass of O3, it support dichromatic symmetry reversal operations.
        One of them, 'P', introduces spatial inversion that makes this rotation
        an improper one.

        Args:
            axis (list): 3D direction of the infinity-fold rotational axis.
            The default is Z.
            dich_operations (default = {}): a set of dichromatic symmetry reversal
            operations marked by string names. The special name 'P' forces spatial
            inversion (all coordinates multiply -1).
        Raises:
            ValueError/TypeError: if axis is not a non-zero 3D vector
        Returns:
            None
        """
        # to be reassigned below
        self._axis = axis
        self.dich_operations = dich_operations
        self._axis_check_and_init( axis = axis )
    
    def _axis_check_and_init( self, axis, atol = 1e-6 ):
        axis = deepcopy( axis )
        if axis is not None:
            if isinstance( axis, np.ndarray ):
                axis = np.array( axis )
            elif isinstance( axis, list):
                axis = np.array( axis )
            else:
                raise TypeError('Must be 3D non-zero vector')
            if len( axis ) == 3 and np.linalg.norm( axis ) > atol:
                self._axis = axis
                label = '∞'
                if 'P' in self._dich_operations:
                    label += '-'
                if 'T' in self._dich_operations:
                    label += "'"
                if 'C' in self._dich_operations:
                    label += '*'
                self.label = label
            else:
                raise ValueError('Must be 3D non-zero vector')
        else:
            raise TypeError('Must be 3D non-zero vector')

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
        do_value = set_copy_assignment( value )
        if 'P' in do_value:
            self._improper = True
        else:
            self._improper = False
        super( LimitingSymmetryOperationO3, self )._dich_operations_setter_impl( do_value )
        self._axis_check_and_init( self._axis )

    @property
    def axis( self ):
        """
        Returns:
            A vector that defines a symmetry rotation axis of order ∞.
        """
        return self._axis

    @axis.setter
    def axis( self, value ):
        """
        A vector that defines a symmetry rotation axis of order ∞.
        The symmetry operation is an infinitesimal rotation (differential rotation matrix).
        
        Raises:
            ValueError: if axis is not a non-zero 3D vector
        """
        self._axis_check_and_init( value )

    @property
    def matrix( self ):
        raise ValueError( 'The matrix is not defined for an infinitesimal rotation' )
        return None
    
    @matrix.setter
    def matrix( self, value ):
        raise ValueError( 'The matrix is not defined for an infinitesimal rotation' )

    def __repr__( self ):
        output = 'SymmetryOperation\nlabel(' + self.label + ")"
        output += '\nAxis: {}'.format(self.axis)
        output += self._dich_output()
        return output

class LimitingSymmetryOperationSO3(LimitingSymmetryOperationO3):
    """
    `LimitingSymmetryOperationSO3` class.
    """
    def __init__( self, axis = [ 0, 0, 1 ], dich_operations = set()):
        """
        Initialise a `LimitingSymmetryOperationSO3` object, that represents a symmetry
        transformation of limiting Curie subgroup of SO(3) group of proper rotations.
        It is based on the infinitfold rotation axis.

        Args:
            axis (list): 3D direction of the infinitfold rotational axis.
            The default is Z.
            dich_operations (default = {}): a set of dichromatic symmetry reversal
            operations marked by string names.
        Raises:
            ValueError/TypeError: if axis is not a 3D vector.
            ValueError: if dich_operations contains the spatial reversal operation 'P'.
        Returns:
            None
        """
        if 'P' in dich_operations:
            raise ValueError('SO3 cannot be an improper rotation')
        else: self._improper = False
        super( LimitingSymmetryOperationSO3, self ).__init__( axis = axis, dich_operations = dich_operations )
    
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
            ValueError: if dich_operations contains the spatial reversal operation 'P'.

        Returns:
            None
        """
        do_value = set_copy_assignment( value )
        if 'P' in do_value:
            raise ValueError('SO3 cannot be an improper rotation')
        else: super( LimitingSymmetryOperationSO3, self )._dich_operations_setter_impl( do_value )
        self._axis_check_and_init( self._axis )
