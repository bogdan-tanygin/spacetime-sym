#
# Copyright (C) 2024 Dr. Bogdan Tanygin <info@tanygin-holding.com>
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
from copy import deepcopy
from spacetime.linear_algebra import is_scalar, is_diagonal, make_0D_scalar

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
    def __init__( self, value = None, dich = None, label = None):
        """
        Initialise a `PhysicalQuantity` object.

        Args:
            value: the value of the physical quantity (always as a NumPy array).
            dich: dichromatic symmetry properties in a key-value form.
            The key is a string label. The value is an actual change when the
            corresponding symmetry operation
            is applied to the value: a multiplying +1 or -1.
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
        self._value = np.array( deepcopy( value ) )
        self._dich = deepcopy( dich )
        self._label = label
    
    def __eq__(self, other):
        """
        Compare `PhysicalQuantity` objects (redefines the operator "==" ).

        Raises:
            TypeError: attempt to compare with non-PhysicalQuantity
            TypeError: attempt to compare different types of values (e.g. vectors and scalars)
        """
        if not isinstance( other, PhysicalQuantity ):
            TypeError( 'Must be compared with other PhysicalQuantity' )
        # Compare dichromatic symmetry properties
        if self._dich != other.dich:
            return False
        # Compare values
        shape_1 = self._value.shape
        shape_2 = other.value.shape
        if shape_1 == shape_2:
            return self._value == other.value
        # compare scalar and diagonal matrix
        elif is_scalar( self._value ) and is_diagonal( other.value ):
            return self._value == make_0D_scalar( other.value )
        elif is_scalar( other.value ) and is_diagonal( self._value ):
            return other.value == make_0D_scalar( self._value )
        else:
            TypeError( 'Dimensions or types of values of physical quantities do not match' )

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

    @property
    def dich(self):
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
