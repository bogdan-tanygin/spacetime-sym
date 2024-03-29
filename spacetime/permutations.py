#
# Copyright (C) 2015, 2021 Benjamin J. Morgan
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

from collections import Counter
from math import factorial
from functools import reduce
from operator import mul

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
