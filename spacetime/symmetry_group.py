#
# Copyright (C) 2024-2025 Dr. Bogdan Tanygin <info@deeptech.business>
# Copyright (C) 2015, 2021 Benjamin J. Morgan
#
# This file is part of spacetime-sym.
#

from math import pi, isclose
import numpy as np
from spacetime import SymmetryOperation, SymmetryOperationO3, SymmetryOperationSO3
from spacetime.physical_quantity import PhysicalQuantity
from spacetime.linear_algebra import is_scalar, is_scalar_extended, is_3D_vector, is_symmetrical_tensor, \
                                     get_tensor_axis
from itertools import product
from copy import deepcopy
from numpy.linalg import eig, norm, det
from scipy.spatial.transform import Rotation

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
        group_rtol (float): relative tolerance of the matrix elements comparison.
    Returns:
        Order of the symmetry group.
    """
    def order( self, group_rtol = 1e-4 ):
        self._validate_and_correct(group_rtol = group_rtol)
        return len( self._symmetry_operations )

    def _add_so_if_new( self, so, group_rtol ):
        self._gen_flag = False
        # if the matrix is new, we add symmetry operation anyway
        if not any( np.allclose( so.matrix, so_0_.matrix, rtol = group_rtol ) for so_0_ in self._symmetry_operations ):
            self._symmetry_operations.append( so )
            self._gen_flag = True
        # if there is already such matrix, we need to compare dich properties
        elif isinstance( so, SymmetryOperationO3 ):
            # default inside this logical branch
            self._gen_flag = True
            for so_0_ in self._symmetry_operations:
                # this must be true for some operations if we are here
                if np.allclose( so.matrix, so_0_.matrix, rtol = group_rtol ):
                    # if the existing symmetry operation is not dichromatic, we add the new one anyway
                    # note: we enabled dichromatic properties for the group O3 and its subgroups only
                    # allowing us automatic (spatial) parity control, etc.
                    if isinstance( so_0_, SymmetryOperationO3 ):
                        if ( so.dich_operations == so_0_.dich_operations ) or ( len( so.dich_operations ) == 0 
                                                                                and len( so_0_.dich_operations ) == 0 ):
                            # at least one match is enough to skip adding a new one
                            self._gen_flag = False
                            break
            if ( self._gen_flag ):
                self._symmetry_operations.append( so )

    def _validate_and_correct( self, group_rtol = 1e-4 ):
        # identity check / add if needed
        if not self.identity_w_dich_flag:
            if not any( np.allclose( so.matrix, self._e_0.matrix, rtol = group_rtol) for so in self._symmetry_operations):
                self.add_and_generate( self._e_0 )
        else:
            # identity operation must contain empty list of dichromatic reversals
            if not any( np.allclose( so.matrix, self._e_0.matrix, rtol = group_rtol) and len( so.dich_operations ) == 0 
                       for so in self._symmetry_operations):
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
                            if np.allclose( so.matrix, so_1.matrix, rtol = group_rtol ):
                                if type(so) is SymmetryOperation and type(so_1) is SymmetryOperation:
                                    indx_for_remove.append( j )
                                # for SymmetryOperationO3 and its subclasses, hence, isinstance(), not type()
                                elif isinstance( so, SymmetryOperationO3 ) and isinstance( so_1, SymmetryOperationO3 ):
                                    if ( so.dich_operations == so_1.dich_operations ) or ( len( so.dich_operations ) == 0 
                                                                                and len( so_1.dich_operations ) == 0 ):
                                        indx_for_remove.append( j )
            for i in sorted(indx_for_remove, reverse = True):
                del self._symmetry_operations[i]
        # second, let's regenerate it. To do it, let's remove the symmetry operations & add them again.
        for so in self._symmetry_operations:
            self._symmetry_operations.remove( so )
            self.add_and_generate( so )

    def add_and_generate( self, so, group_rtol = 1e-4  ):
        """
        Add a :any:`SymmetryOperation` to this :any:`SymmetryGroup` and generate
        all the remaining group operations based on consideration of the current list of
        symmetry operations as a group generator.
        Args:
            symmetry_operation (:any:`SymmetryOperation`): The :any:`SymmetryOperation` to add.
            group_rtol (float): relative tolerance of the matrix elements comparison.
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
        self._add_so_if_new( so, group_rtol )        
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

    def _save_basic_identity_so( self, dim_0, identity_w_dich_flag ):
        #identity is a must by the group definition
        if not identity_w_dich_flag:
            e = SymmetryOperation( matrix = np.identity( dim_0 ))
            e.label = 'E'
        else:
            e = SymmetryOperationSO3( matrix = np.identity( dim_0 ), dich_operations = set() )
            e.label = 'E'
        self._e_0 = e
    
    def _symmetry_operations_check_and_init( self, symmetry_operations={} ):
        """
        Checks the consistency of symmetry operations list provided.
        If everything is correct, assigns it.

        Args:
            symmetry_operations: a list or a set of :any:`SymmetryOperation` objects.

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
        self.identity_w_dich_flag = False
        if n_0 > 0:
            # type check
            if any( not isinstance( so, SymmetryOperation ) for so in symmetry_operations ):
                raise TypeError('symmetry_operations must contain SymmetryOperation objects only')
            so_random = next( iter( symmetry_operations ) )
            # it is already checked in the SymmetryOperation class that the matrix is square
            dim_0 = so_random.matrix.shape[0]
            # dimensions matching: matrix/vector case
            # TODO UT
            if dim_0 > 1:
                if not all( dim_0 == so.matrix.shape[0] for so in symmetry_operations ):
                    raise ValueError('Different dimensions of input symmetry operations')
            # dimensions matching: scalar case
            else:
                if not all( is_scalar( so.matrix ) for so in symmetry_operations ):
                    raise ValueError('Different dimensions of input symmetry operations')
            # set the number of dimensions of the group's transformation
            self._symmetry_operations = symmetry_operations
            if any(  isinstance( so, SymmetryOperationO3 ) for so in symmetry_operations ):
                self.identity_w_dich_flag = True
            self._save_basic_identity_so( dim_0, self.identity_w_dich_flag )
        else:
            # default 3-dimensional group. Can be changed after reassignment of symmetry_operations
            dim_0 = 3
            self._save_basic_identity_so( dim_0, False )
            self._symmetry_operations = [ self._e_0 ]

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

    #TODO UT
    def is_invariant( self, physical_quantity, rtol = 1e-6, atol = 1e-14 ):
        """
        Check whether the given physical_quantity is an invariant of the given symmetry group transformations.

        Args:
            physical_quantity (PhysicalQuantity): a physical quantity to check.
            rtol (float): a relative tolerance of the comparing
            atol (float): an absolute tolerance of the comparing

        Raises:
            TypeError: if physical_quantity does not belong to the class PhysicalQuantity

        Return:
            (bool): True | False
        """
        if not isinstance( physical_quantity, PhysicalQuantity ):
            raise TypeError('physical_quantity must belongs to the class PhysicalQuantity')
        invariant_flag = True
        #TODO UT diagonal bidirectional asymmetric tensor test with mirror mx, my, mz
        for so in self.symmetry_operations:
            pq_updated = so * physical_quantity
            if not np.allclose( pq_updated.value, physical_quantity.value, rtol = rtol ):
                if not physical_quantity.bidirector:
                    invariant_flag = False
                elif not np.allclose( pq_updated.value, - physical_quantity.value, rtol = rtol ):
                    invariant_flag = False
        return invariant_flag

    def __repr__( self ):
        to_return = '{}\n'.format( self.__class__.class_str )
        for so in self._symmetry_operations:
            to_return += "{}\n".format( so )
        return to_return

    def __mul__( self, other ):
        """
        Direct product.
        """
        return SymmetryGroup( [ s1 * s2 for s1, s2 in product( self._symmetry_operations, other.symmetry_operations ) ] )

#TODO
#TODO UT for LimitingSymmetryGroupAxial similar to LimitingSymmetryGroupScalar
#TODO functional UTs to cover ∞2, ∞/m, ∞mm, or ∞/mm + dich
#TODO UTs: init through decorator assignments
class LimitingSymmetryGroupAxial(SymmetryGroup):
    """
    `LimitingSymmetryGroupAxial` class.
    """

    class_str = 'LimitingSymmetryGroupAxial'

    def __init__( self, axis = [ 1, 0, 0 ], symmetry_operations = [ SymmetryOperationSO3( ) ] ):
        """
        Create a :any:`LimitingSymmetryGroupAxial` object of a symmetry group of
        an axis/vector/straight line/plane. Using Hermann-Mauguin notation,
        it is one of the following limiting Curie groups:
        ∞ (default), ∞2, ∞/m, ∞mm, or ∞/mm. First two describe symmetry of chiral physical objects.

        Args:
            axis: the direction of the infinity-fold rotational symmetry axis.
                Must be either `numpy.ndarray` or `list`.
            symmetry_operations (list): a list of symmetry operations which supplement ∞.
                They can be different orientations of 2, m, and, also, identity (always) and a spatial inversion 1-.
                with (optionally) dichromatic reversals which make the group "grey" for the given property.
                For instance, 'T' for time-reversal can make ∞/m1'.
                If an only set of dichromatic reversals form a combined
                reversals with an improper rotation, it defines a noninvariant chirality.
                For instance, 'T' with 'P' makes makes m' and the whole group looks like ∞/m'.
        Raises:
            ValueError: if the axis is not an invariant of the rest symmetry operations
        Returns:
            None
        """
        self._check_and_set_axis( axis = axis )
        #TODO UTs - extending some of init test cases - for axis and [so] reassignemnt
        super(LimitingSymmetryGroupAxial, self).__init__( symmetry_operations = symmetry_operations)
        self._axial_symmetry_operations_check()
        # we can generate and assign lavel after the axis and symmetry operations have been initialised
        self._assign_label()
    
    def _check_and_set_axis( self, axis ):
        axis = deepcopy( axis )
        if isinstance( axis, np.ndarray ) or isinstance( axis, list):
            self._axis = np.array( axis ) / norm( np.array( axis ) )
        else:
            raise TypeError('Not a vector')
        if not is_3D_vector( self._axis ):
            raise TypeError('Not a 3D vector')
        # the 2-fold rotational axis as a part of the axial limiting group
        rot_vec_2 =  self.axis * pi / norm( self.axis )
        rot_2 = Rotation.from_rotvec( rot_vec_2, degrees = False )
        so_2 = SymmetryOperationSO3( matrix = rot_2 )
        self._so_2 = so_2

    @property
    def axis( self ):
        return self._axis
    
    #TODO UT
    @axis.setter
    def axis( self, value ):
        """
        The direction of the infinity-fold rotational symmetry axis.

        Args:
            value: the axis' direction (3D vector). Must be either `numpy.ndarray` or `list`.

        Raises:
            TypeError/ValueError: in cases of not a vector / 3D vector.

        Returns:
            None
        """
        self._check_and_set_axis( axis = value )
        # we need to reinit the group if the axis has been changed
        self._axial_symmetry_operations_check()
        # the 2-fold rotational axis is a part of the axial limiting group anyway. It is practical to have it
        # added implicitly for the generation of the rest 
        self._symmetry_operations.append( self._so_2 )
        # now, the group should be regenerated (new axis)
        super(LimitingSymmetryGroupAxial, self).__init__( symmetry_operations = self.symmetry_operations)
        # we can generate and assign lavel after the axis and symmetry operations have been initialised
        self._assign_label()

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

    #TODO UT
    @symmetry_operations.setter
    def symmetry_operations( self, value ):
        so_list = list( value )
        so_list.append( self._so_2 )
        self._symmetry_operations_check_and_init( so_list )
        self._validate_and_correct()
        self._assign_label()
        # with new symmetry operations, the axis should be revalidated
        self._axial_symmetry_operations_check()

    #TODO UT
    def _axial_symmetry_operations_check( self ):
        # let's check that the given axis is an invariant of the rest symmetry operations
        # assumption: we have already initialised the set of symmetry operations
        physical_quantity = PhysicalQuantity( value = self.axis, bidirector = True )
        invariant_flag = super( LimitingSymmetryGroupAxial, self ).is_invariant( physical_quantity = physical_quantity )
        if not invariant_flag:
            raise ValueError( 'The provided axis is not an invariant of the rest symmetry operations' )

    def is_invariant( self, physical_quantity, rtol = 1e-6, atol = 1e-14 ):
        """
        Check whether the given physical_quantity is an invariant of the given symmetry group transformations.

        Args:
            physical_quantity (PhysicalQuantity): a physical quantity to check.
            rtol (float): a relative tolerance of the comparing
            atol (float): an absolute tolerance of the comparing
            decimal (int): number of decimal places to round eigen values to

        Raises:
            TypeError: if physical_quantity does not belong to the class PhysicalQuantity

        Return:
            (bool): True | False
        """
        invariant_flag = super( LimitingSymmetryGroupAxial, self ).is_invariant( physical_quantity = physical_quantity )
        if invariant_flag:
            if is_3D_vector( physical_quantity.value ):
                # collinearity test for the vector
                cross_product = np.cross( self.axis, physical_quantity.value )
                if not np.allclose( cross_product, np.array( np.zeros( (3) ) ), atol = atol):
                    invariant_flag = False
            elif is_symmetrical_tensor( physical_quantity.value, rtol = rtol, atol = atol ):
                if not is_scalar_extended( physical_quantity.value ):
                    eigvals, eigvecs_tmp = np.linalg.eigh( physical_quantity.value )
                    # represent the eigvecs as rows:
                    eigvecs = eigvecs_tmp.transpose()
                    eigvals_abs = np.zeros( 3 )
                    for i in range( 3 ):
                        eigvals_abs[ i ] = abs( eigvals[ i ] )
                    max_val = np.max( eigvals_abs )
                    dec_tol = int( np.log10( max_val * rtol ) )
                    if dec_tol >= 0:
                        dec_tol = 0
                    qty_unique_vals = len( np.unique( np.round( eigvals, decimals = - dec_tol ) ) )
                    # is the tensor axially symmetric?
                    if qty_unique_vals == 2:
                        three_indx = { 0, 1, 2 }
                        for i in range( 2 ):
                            for j in range( i + 1, 3 ):
                                if isclose( eigvals[i], eigvals[j], rel_tol = rtol ):
                                    # the remaining eigenvalue defines the axial vector of the given tensor
                                    indx_axial = next( iter( three_indx - { i, j } ) )
                                    tensor_axis = eigvecs[ indx_axial ]
                        cross_product = np.cross( self.axis, tensor_axis )
                        if not np.allclose( cross_product, np.array( np.zeros( (3) ) ), atol = atol):
                            invariant_flag = False
                    elif qty_unique_vals == 3:
                        # anisotropic tensor. Hence, no axial symmetry
                        invariant_flag = False
                    #elif qty_unique_vals == 1:
                        # isotropic tensor, nothing to do -- keep the defined invariance flag
            elif not is_scalar_extended( physical_quantity.value ):
                raise ValueError('Non-symmetrical tensor or another unsupporte (in this version) type')
        
        return invariant_flag

    #TODO prio
    def _assign_label( self , atol = 1E14 ):
        #label = '∞'
        label = ''
        for so in self.symmetry_operations:
            # for O3 and its subgroups
            if isinstance( so, SymmetryOperationO3 ):
                matrix_axis = get_tensor_axis( so.matrix )
                
                is_axial = True
                if norm( matrix_axis ) < atol:
                    is_axial = False
                det_val = det( so.matrix )
                
                is_proper_rot = True
                if is_axial < 0:
                    is_proper_rot = False
                
                is_collinear = False
                c = np.cross( matrix_axis, self.axis )
                if norm( c ) < atol:
                    is_collinear = True
                
                is_perpend = False
                d = np.dot( matrix_axis, self.axis )
                if abs( d ) < atol:
                    is_perpend = True

                if not is_proper_rot and not is_axial:
                    label += '\\'
                
                if is_proper_rot and is_axial and is_perpend:
                    label += '2'
                
                if not is_proper_rot and is_axial:
                    label += 'm'

                if 'P' in so.dich_operations:
                    remaining_dich_set = so.dich_operations - { 'P' }
                else:
                    #label += '1'
                    remaining_dich_set = so.dich_operations
                remaining_dich_set = list( remaining_dich_set )
                remaining_dich_set.sort()
                for dich in remaining_dich_set:
                    if dich == 'T':
                        label += "'"
                    elif dich == 'C':
                        label += '*'
                    else:
                        label += dich
        if '\\' in label:
            label = label.replace("\\", "")
            label = '\\' + label
        label = '∞' + label
        self.label = label
    
    def __repr__( self ):
        to_return = '{}\n'.format( self.label )
        to_return += super( LimitingSymmetryGroupAxial, self ).__repr__()
        return to_return

#TODO init through decorator assignments, UTs
class LimitingSymmetryGroupScalar(LimitingSymmetryGroupAxial):
    """
    `LimitingSymmetryGroupScalar` class.
    """

    class_str = 'LimitingSymmetryGroupScalar'

    def __init__( self, scalar_symmetry_operations = [ SymmetryOperationSO3( ) ] ):
        """
        Create a :any:`LimitingSymmetryGroupPoint` object of a symmetry group of a point or a (pseudo)scalar.
        Using Hermann-Mauguin notation, it is one of two limiting Curie groups:
        ∞∞ (default) or ∞∞m. In terms of a physical quantity that is invariant under these groups'
        transformations, ∞∞m describes a scalar and ∞∞ describes a pseudoscalar.

        Args:
            scalar_symmetry_operations (list): a list of identity- or inversion-based symmetry operations
                with (optionally) dichromatic reversals which make the group "grey" for the given property.
                For instance, 'T' for time-reversal makes ∞∞1' or ∞∞m1'
                The group ∞∞m is defined by 'P' reversal, i.e. a spatial inversion.
                If an only set of dichromatic reversals form a combined
                reversals with an improper rotation, it defines a noninvariant chirality.
                For instance, 'T' with 'P' makes makes m' and the whole group looks like ∞∞m'.
        Raises:

        Returns:
            None
        """
        #TODO call this during [ so ] assignment here as well. Local decorator is needed
        #TODO duplicate UTs - using init test cases
        self._scalar_symmetry_operations_check( scalar_symmetry_operations = scalar_symmetry_operations)
        super(LimitingSymmetryGroupScalar, self).__init__( symmetry_operations = scalar_symmetry_operations)
        self._assign_label()
    
    def _scalar_symmetry_operations_check( self, scalar_symmetry_operations, rtol = 1e-6 ):
        if not isinstance( scalar_symmetry_operations, list ):
            raise TypeError('Must be a list of SymmetryOperation objects')
        for so in scalar_symmetry_operations:
            if not isinstance( so, SymmetryOperation ):
                #TODO UT
                raise TypeError('The objects in the list must belong to SymmetryOperation or its subclasses')
            n_dim = so.matrix.shape[0]
            if not ( np.allclose( so.matrix,   np.identity( n_dim ), rtol = rtol) or
                     np.allclose( so.matrix, - np.identity( n_dim ), rtol = rtol) ):
                raise ValueError('Must be an identity or inversion matrix')

    def is_invariant( self, physical_quantity, rtol = 1e-6 ):
        """
        Check whether the given physical_quantity is an invariant of the given symmetry group transformations.

        Args:
            physical_quantity (PhysicalQuantity): a physical quantity to check.
            rtol (float): a relative tolerance of the comparing

        Raises:
            TypeError: if physical_quantity does not belong to the class PhysicalQuantity

        Return:
            (bool): True | False
        """
        invariant_flag = super( LimitingSymmetryGroupScalar, self ).is_invariant( physical_quantity = physical_quantity, rtol = rtol )
        if not is_scalar_extended( physical_quantity.value ):
            invariant_flag = False
        return invariant_flag

    def _assign_label( self ):
        #TODO UT: more cases
        label = '∞∞'
        for so in self.symmetry_operations:
            # for O3 and its subgroups
            if isinstance( so, SymmetryOperationO3 ):
                if 'P' in so.dich_operations:
                    label += 'm'
                    remaining_dich_set = so.dich_operations - { 'P' }
                else:
                    if len( so.dich_operations ) > 0:
                        label += '1'
                    remaining_dich_set = so.dich_operations
                remaining_dich_set = list( remaining_dich_set )
                remaining_dich_set.sort()
                for dich in remaining_dich_set:
                    if dich == 'T':
                        label += "'"
                    elif dich == 'C':
                        label += '*'
                    else:
                        label += dich
        self.label = label
    
    def __repr__( self ):
        to_return = '{}\n'.format( self.label )
        to_return += super( LimitingSymmetryGroupScalar, self ).__repr__()
        return to_return

