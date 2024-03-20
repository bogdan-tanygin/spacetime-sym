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

class PhysicalQuantity(object):
    """
    This class describes a physical quantity supplemented by its 
    dichromatic symmetry properties (hereinafter, "dis") against
    selected symmetry operations. The default dis are specified
    against charge conjugation (C-symmetry), spatial mirror or
    inversion operation (parity, P), and the time reversal symmetry
    (T-symmetry).
    """

    class_str = 'Spacetime Physical Quantity'

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
        self._value = np.array(value)

    @property
    def dis(self):
        """
        Dichromatic symmetry properties of :any:`PhysicalQuantity`
        with respect to defined symmetry operations.

        Args:
            None

        Returns:
            (dict):  {'key1': 'value1', ...}, where
            values are 1 for symmetry, -1 for asymmetry, and 0 for undefined.
        """
        return self._dis
    
    @dis.setter
    def dis(self, value):
        dis_types = list(value.keys())
        for dis_type in dis_types:
            if value[dis_type] not in {1, -1, 0}:
                raise ValueError('Wrong {}\'s dichromatic symmetry property value!'.format(dis_type))
        self._dis = value
