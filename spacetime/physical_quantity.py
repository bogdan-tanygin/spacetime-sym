# Copyright (C) 2024-2026 Dr. Bogdan Tanygin <info@deeptech.business>

import numpy as np
from copy import deepcopy
from spacetime.linear_algebra import is_scalar, is_diagonal, make_0D_scalar, is_scalar, is_scalar_extended, is_equal_2D

class PhysicalQuantity(object):
    """
    Represent a scalar, vector, or rank-two tensor with configurable reversal characters.

    The ``dich`` mapping associates symmetry-reversal labels with +1 or -1
    factors. These factors describe the quantity's response to operations
    such as parity (P), time reversal (T), and the symbolic charge-conjugation
    label (C), as well as other user-defined binary labels. Spatial O(3)
    matrices act on the value separately.
    """
    def __init__( self, value = None, dich = { 'C':1, 'P':1, 'T':1 }, bidirector = False, label = ""):
        """
        Initialise a `PhysicalQuantity` object.

        Args:
            value: Value of the physical quantity, stored as a NumPy array.
            dich: Mapping from symmetry-reversal labels to +1 or -1 factors.
                These factors are applied when the corresponding operation
                acts on the quantity.
            bidirector: If True, equality treats a value and its negative as
                equivalent (for example, for an unoriented axis). This flag
                can also be used with scalars and tensors.
            label: Name of the physical quantity.
        Raises:
            TypeError: If a dichromatic symmetry key is not a string.
            ValueError: If a dichromatic symmetry factor is not +1 or -1.
            TypeError: If the label is not a string.

        Returns:
            None
        """
        if dich is not None:
            self._check_dich_format( dich )
        if label is not None:
            self._check_label_format( label )
        # regardless of the value's type and dimension, it is stored
        # as a NumPy array
        if not is_scalar( value ):
            self._value = np.array( deepcopy( value ) )
        else:
            self._value = np.array( value )
        self._dich = deepcopy( dich )
        self._label = label
        self._check_and_set_bidirector_flag( bidirector = bidirector )
    
    def __eq__(self, other):
        """
        Compare two `PhysicalQuantity` objects using the ``==`` operator.

        Raises:
            TypeError: If the other object is not a PhysicalQuantity.
            TypeError: If the value types or dimensions are incompatible.
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
            # Bidirector comparison ignores the overall sign.
            return is_equal_2D( value_1, value_2 ) or is_equal_2D( value_1, np.array( - value_2 ) )
        elif self.bidirector or other.bidirector:
            # Comparing a bidirector with a non-bidirector is ambiguous.
            raise TypeError("""Comparison of bidirector and non-bidirector is ambiguous and depends on the theory
                         Feel free to implement it here for your case""")
        else:
            # Neither quantity is a bidirector.
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
        Flag indicating whether the overall sign of the value is disregarded.

        Raises:
            ValueError: If the flag is not boolean.
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
            dict: Mapping from reversal labels to +1 (even) or -1 (odd)
            transformation factors.
        """
        return self._dich
    
    @dich.setter
    def dich(self, value):
        self._check_dich_format( value )
        self._dich = deepcopy( value )
