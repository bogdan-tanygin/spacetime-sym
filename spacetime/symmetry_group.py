#
# Copyright (C) 2024 Dr. Bogdan Tanygin <info@deeptech.business>
# Copyright (C) 2015, 2021 Benjamin J. Morgan
#
# This file is part of spacetime-sym.
#

import numpy as np
from spacetime import SymmetryOperation, SymmetryOperationO3, SymmetryOperationSO3
from spacetime.linear_algebra import is_scalar
from itertools import product
from copy import deepcopy

class SymmetryGroup:
    """
    :any:`SymmetryGroup` class.

    A :any:`SymmetryGroup` object contains a list of :any:`SymmetryOperation` objects. Their order can be arbitrary.

    e.g.::

        SymmetryGroup( symmetry_operations=[ s1, s2, s3 ] )

    where `s1`, `s2`, and `s3` are :any:`SymmetryOperation` objects.

    :any:`SymmetryGroup` objects can also be created from files using the class methods::

        SymmetryGroup.read_from_file( filename )

    and::

        SymmetryGroup.read_from_file_with_labels( filename )
    """

    class_str = 'SymmetryGroup'

    def __init__( self, symmetry_operations=[] ):
        """
        Create a :any:`SymmetryGroup` object.

        Args:
            symmetry_operations (list): A list of :any:`SymmetryOperation` objects.

        Returns:
            None
        """
        self._symmetry_operations_check_and_init( symmetry_operations )
        self._validate_and_correct()

    @property
    def symmetry_operations( self ):
        """
        List of symmetry transformations forming the group.

        Args:
            None

        Returns:
            List of symmetry operations.
        """
        return self._symmetry_operations

    @symmetry_operations.setter
    def symmetry_operations( self, value ):
        self._symmetry_operations_check_and_init( value )
        self._validate_and_correct()

    """
    Get an order of the group. Forces the group validation before that.
    Args:
        group_atol (float): absolute tolerance of the matrix elements comparison.
    Returns:
        Order of the symmetry group.
    """
    def order( self, group_atol = 1e-4 ):
        self._validate_and_correct(group_atol = group_atol)
        return len( self._symmetry_operations )

    def _add_so_if_new( self, so, group_atol ):
        if not any( np.allclose( so.matrix, so_0_.matrix, atol = group_atol ) for so_0_ in self._symmetry_operations ):
            self._symmetry_operations.append( so )
            self._gen_flag = True
    
    #TODO add dich functionality to _validate_and_correct, _add_so_if_new, and add_and_generate

    def _validate_and_correct( self, group_atol = 1e-4 ):
        # identity check / add if needed
        if not any( np.allclose( so.matrix, self._e_0.matrix, atol = group_atol) for so in self._symmetry_operations):
            self.add_and_generate( self._e_0 )
        # first, let's deduplicate the group
        # group order (as of here and now):
        g_order = len( self._symmetry_operations )
        if g_order > 1:
            indx_for_remove = list()
            for i in range( g_order ):
                if i not in indx_for_remove:
                    for j in range( g_order ):
                        if i != j:
                            so = self._symmetry_operations[i]
                            so_1 = self._symmetry_operations[j]
                            if np.allclose( so.matrix, so_1.matrix, atol = group_atol ):
                                indx_for_remove.append( j )
            for i in sorted(indx_for_remove, reverse = True):
                del self._symmetry_operations[i]
        # second, let's regenerate it. To do it, let's remove the symmetry operations & add them again.
        for so in self._symmetry_operations:
            self._symmetry_operations.remove( so )
            self.add_and_generate( so )

    def add_and_generate( self, so, group_atol = 1e-4  ):
        """
        Add a :any:`SymmetryOperation` to this :any:`SymmetryGroup` and generate
        all the remaining group operations based on consideration of the current list of
        symmetry operations as a group generator.
        Args:
            symmetry_operation (:any:`SymmetryOperation`): The :any:`SymmetryOperation` to add.
            group_atol (float): absolute tolerance of the matrix elements comparison.
        Raises:
            ValueError: in case of numbers of dimensions mismatch.
        Returns:
            None
        """
        # dimensions must match. They already match inside group.
        # Also, the matrices are already square in case of SymmetryOperation objects
        if any( so_0.matrix.shape[0] != so.matrix.shape[0] for so_0 in self._symmetry_operations ):
            ValueError('Wrong dimensions of the input symmetry operation')
        # Adding part
        self._add_so_if_new( so, group_atol )        
        # Generating part
        if self._gen_flag:
            # the flag is used to track whether group continues to generate new elements
            # it means that we need one more cycle unless it stops
            self._gen_flag = False
            so_list_0 = deepcopy( self._symmetry_operations )
            for so_0 in so_list_0:
                so_1 = so_0 * so
                so_1_inv = so_1.invert()
                self.add_and_generate( so_1 )
                self.add_and_generate( so_1_inv )
                so_2 = so * so_0
                so_2_inv = so_2.invert()
                self.add_and_generate( so_2 )
                self.add_and_generate( so_2_inv )

    def _save_basic_identity_so( self, dim_0 ):
        #identity is a must by the group definition
        e = SymmetryOperation( matrix = np.identity( dim_0 ))
        self._e_0 = e
    
    def _symmetry_operations_check_and_init( self, symmetry_operations={} ):
        """
        Checks the consistency of symmetry operations list provided.
        If everything is correct, assigns it.

        Args:
            symmetry_operations: a list of :any:`SymmetryOperation` objects.

        Raises:
            TypeError: symmetry operations do not belong to :any:`SymmetryOperation` or its derived classes
            ValueError: dimensions of symmetry operations do not match
        Returns:
            None.
        """
        # make sure, it is a list
        symmetry_operations = list( symmetry_operations )
        # number of symmetry operations given at the init stage
        n_0 = len( symmetry_operations )
        if n_0 > 0:
            # type check
            if any( not isinstance( so, SymmetryOperation ) for so in symmetry_operations ):
                raise TypeError('symmetry_operations must contain SymmetryOperation objects only')
            so_random = next( iter( symmetry_operations ) )
            # it is already checked in the SymmetryOperation class that the matrix is square
            dim_0 = so_random.matrix.shape[0]
            # dimensions matching: matrix/vector case
            if dim_0 > 1:
                if not all( dim_0 == dim_0 for so in symmetry_operations ):
                    raise ValueError('Different dimensions of input symmetry operations')
            # dimensions matching: scalar case
            else:
                if not all( is_scalar( so.matrix ) for so in symmetry_operations ):
                    raise ValueError('Different dimensions of input symmetry operations')
            # set the number of dimensions of the group's transformation
            self._symmetry_operations = symmetry_operations
            self._save_basic_identity_so( dim_0 )
        else:
            # default 3-dimensional group. Can be changed after reassignment of symmetry_operations
            dim_0 = 3
            self._save_basic_identity_so( dim_0 )
            self._symmetry_operations = { self._e_0 }

    @classmethod
    def read_from_file( cls, filename ):
        """
        Create a :any:`SymmetryGroup` object from a file.
       
        The file format should be a series of numerical mappings representing each symmetry operation.

        e.g. for a pair of equivalent sites::

            # example input file to define the spacegroup for a pair of equivalent sites
            1 2
            2 1

        Args:
            filename (str): Name of the file to be read in.

        Returns:
            spacegroup (SymmetryGroup)	
        """
        data = np.loadtxt( filename, dtype=int )
        symmetry_operations = [ SymmetryOperation.from_vector( row.tolist() ) for row in data ]
        return( cls( symmetry_operations = symmetry_operations ) )

    @classmethod
    def read_from_file_with_labels( cls, filename ):
        """
        Create a :any:`SymmetryGroup` object from a file, with labelled symmetry operations.

        The file format should be a series of numerical mappings representing each symmetry operation, prepended with a string that will be used as a label.

        e.g. for a pair of equivalent sites::

            # example input file to define the spacegroup for a pair of equivalent sites
            E  1 2
            C2 2 1

        Args:
            filename (str): Name of the file to be read in.

        Returns:
            spacegroup (SymmetryGroup)
        """
        data = np.genfromtxt( filename, dtype=str )
        labels = [ row[0] for row in data ]
        vectors = [ [ float(s) for s in row[1:] ] for row in data ]
        symmetry_operations = [ SymmetryOperation.from_vector( v ) for v in vectors ]
        [ so.set_label( l ) for (l, so) in zip( labels, symmetry_operations ) ]
        return( cls( symmetry_operations=symmetry_operations ) )

    def save_symmetry_operation_vectors_to( self, filename ):
        """
        Save the set of vectors describing each symmetry operation in this :any:`SymmetryGroup` to a file.

        Args:
            filename (str): Name of the file to save to.

        Returns:
            None
        """
        operation_list = []
        for symmetry_operation in self._symmetry_operations:
            operation_list.append( symmetry_operation.as_vector() )
        np.savetxt( filename, np.array( operation_list ), fmt='%i' )

    def by_label( self, label ):
        """
        Returns the :any:`SymmetryOperation` with a matching label.

        Args:
            label (str): The label identifying the chosen symmetry operation.

        Returns:
            (:any:`SymmetryOperation`): The symmetry operation that matches this label.
        """ 
        return next((so for so in self._symmetry_operations if so.label == label), None)

    @property
    def labels( self ):
        """
        A list of labels for each :any:`SymmetryOperation` in this spacegroup.

        Args:
            None

        Returns:
            (list): A list of label strings.
        """
        return [ so.label for so in self._symmetry_operations ] 

    def __repr__( self ):
        to_return = '{}\n'.format( self.__class__.class_str )
        for so in self._symmetry_operations:
            to_return += "{}\t{}\n".format( so.label, so.as_vector() )
        return to_return

    @property
    def size( self ):
        return len( self._symmetry_operations )

    def __mul__( self, other ):
        """
        Direct product.
        """
        return SymmetryGroup( [ s1 * s2 for s1, s2 in product( self._symmetry_operations, other.symmetry_operations ) ] )
