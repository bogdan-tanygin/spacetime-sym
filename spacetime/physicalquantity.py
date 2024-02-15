#
# Copyright (C) 2024 Dr. Bogdan Tanygin <info@tanygin-holding.com>
#
# This file is part of Spacetime-sym.
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

class PhysicalQuantity:

    class_str = 'Spacetime Physical Quantity'

    def __init__(self, parities = {"C":0, "P":0, "T":0}):
        self.parities = parities

    @property
    def value( self ):
        """
        A physical quantity's value.

        Args:
            None

        Returns:
            the value of the physical quantity.
        """
        return self._value
    
    @value.setter
    def value(self, value):
        self._value = value

    @property
    def parities( self ):
        """
        Parities of :any:`PhysicalQuantity` with respect to
        defined conjugations.

        Args:
            None

        Returns:
            (dict):  {'key1': 'value1', ...}, where
            values are 1 for even, -1 for odd, and 0 for undefined.
        """
        return self._parities
    
    @parities.setter
    def parities(self, value):
        parity_types = list(value.keys())
        for parity in parity_types:
            if value[parity] not in {1, -1, 0}:
                raise ValueError('Wrong {} parity value!'.format(parity))
        self._parities = value
