#
# Copyright (C) 2024 Dr. Bogdan Tanygin <info@deeptech.business>
#
# This file is part of spacetime-sym.
#

import numpy as np
from copy import deepcopy
from spacetime.linear_algebra import is_scalar, is_diagonal, make_0D_scalar, is_scalar, is_scalar_extended, is_equal_2D

class PhysicalQuantity(object):
    """
    This class describes a physical quantity supplemented by its 
    dichromatic symmetry properties (hereinafter, "dich") against
    selected symmetry operations. E.g. dis tuple defines whether
    the physical quantity should be inverted under operating upon
    it such symmetry operations as charge conjugation (C-symmetry),
    spatial mirror reflection or inversion operation (parity, P), and
    the time reversal symmetry (T-symmetry).
    """
    def __init__( self, value = None, dich = { 'C':1, 'P':1, 'T':1 }, bidirector = False, label = None):
        """
        Initialise a `PhysicalQuantity` object.

        Args:
            value: the value of the physical quantity (always as a NumPy array).
            dich: dichromatic symmetry properties in a key-value form.
                  The key is a string label. The value is an actual change when the
                  corresponding symmetry operation
                  is applied to the value: a multiplying +1 or -1.
            bidirector: the flag whether sign/direction makes sense for its value.
                        For instance, if value is an axis, not a directed vector,
                        then True. It also applies to scalars and tensors.
            label: a name of the physical quantity
        Raises:
            TypeError: if the dichromatic symmetry keys are not textual
            ValueError: if the dichromatic symmetry values are not +1 or -1.
            TypeError: if the label is not textual

        Returns:
            None
        """
        if dich is not None:
            self._check_dich_format( dich )
        if label is not None:
            self._check_label_format( label )
        # regardless of the value's type and dimension, it is stored
        # as an numpy array
        if not is_scalar( value ):
            self._value = np.array( deepcopy( value ) )
        else:
            self._value = np.array( value )
        self._dich = deepcopy( dich )
        self._label = label
        self._check_and_set_bidirector_flag( bidirector = bidirector )
    
    def __eq__(self, other):
        """
        Compare `PhysicalQuantity` objects (redefines the operator "==" ).

        Raises:
            TypeError: attempt to compare with non-PhysicalQuantity
            TypeError: attempt to compare different types of values (e.g. vectors and scalars)
        """
        if not isinstance( other, PhysicalQuantity ):
            raise TypeError( 'Must be compared with other PhysicalQuantity' )
        # Compare dichromatic symmetry properties
        if self._dich != other.dich:
            return False
        # Compare values
        shape_1 = self._value.shape
        shape_2 = other.value.shape
        if shape_1 == shape_2:
            value_1 = self._value
            value_2 = other.value
        # compare scalar and diagonal matrix
        elif is_scalar_extended( self._value ) and is_scalar_extended( other.value ):
            value_1 = make_0D_scalar( self._value )
            value_2 = make_0D_scalar( other.value )
        else:
            raise TypeError( 'Dimensions or types of values of physical quantities do not match' )
        if self.bidirector and other.bidirector:
            # bidirectors comparison ignores their signs
            return is_equal_2D( value_1, value_2 ) or is_equal_2D( value_1, np.array( - value_2 ) )
        elif self.bidirector or other.bidirector:
            # if ONLY one of them is a bidirector, comparison depends on the theory
            raise TypeError("""Comparison of bidirector non-bidirector is ambiguous and depends on the theory
                         Feel free to implement it here for your case""")
        else:
            # both are non-bidirectors
            return is_equal_2D( value_1, value_2 )

    @property
    def label( self ):
        """
        A physical quantity's name.

        Args:
            None

        Returns:
            a name of the physical quantity.
        """
        return self._label

    @label.setter
    def label( self, value ):
        self._check_label_format( value )
        self._label = value

    @property
    def value( self ):
        """
        A physical quantity's value.

        Args:
            None

        Returns:
            a value of the physical quantity.
        """
        return self._value
    
    @value.setter
    def value( self, value ) :
        # array-like representation of vectors, scalars, and tensors
        self._value = np.array( deepcopy( value ) )

    def _check_dich_format( self, dich ):
        if not isinstance( dich, dict ):
            raise TypeError( 'dich must be a dictionary' )
        dis_types = dich.keys()
        for dis_type in dis_types:
            if not isinstance( dis_type, str ):
                raise TypeError( 'The key must be a string' )
            if dich[dis_type] not in {1, -1}:
                raise ValueError( 'Wrong {}\'s dichromatic symmetry property value!'.format( dis_type ) )
    
    def _check_label_format( self, label ):
        if not isinstance( label, str ):
            raise TypeError( 'The label must be a string' )

    def _check_and_set_bidirector_flag( self, bidirector ):
        if bidirector not in { True, False }:
            raise ValueError( 'bidirector must be boolean' )
        self._bidirector = bidirector

    @property
    def bidirector( self ):
        return self._bidirector
    
    @bidirector.setter
    def bidirector( self, value ):
        """
        The flag whether sign/direction makes sense for its value.

        Raises:
            ValueError: if not boolean
        """
        self._check_and_set_bidirector_flag( bidirector = value )


    @property
    def dich( self ):
        """
        Dichromatic symmetry properties of :any:`PhysicalQuantity`
        with respect to defined symmetry operations.

        Args:
            None

        Returns:
            (dict):  {'key1': 'value1', ...}, where
            values are 1 for symmetry and -1 for asymmetry,
            and key label the dichromatic reversal symmetry 
            operation.
        """
        return self._dich
    
    @dich.setter
    def dich(self, value):
        self._check_dich_format( value )
        self._dich = deepcopy( value )
