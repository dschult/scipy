"""Sparse 1D array format using compressed storage sparsetools."""
__all__ = ['uni_array']

import operator

import numpy as np

from ._base import _spbase, issparse, sparray
from ._data import _data_matrix
from . import _sparsetools
from ._sparsetools import (get_csr_submatrix, csr_sample_values, csr_row_index,
                           csr_row_slice, csr_column_index1, csr_column_index2,
                           csr_matvec, csr_matvecs)
from ._compressed import _cs_matrix, _process_slice
from ._sputils import (upcast, upcast_char, isshape,
                       getdtype, downcast_intp_index,
                       check_shape, validateaxis)


class uni_array(_cs_matrix, sparray):
    _format = 'uni'

#    @property
#    def indptr(self):
#        self._indptr = self._indptr.astype(self.indices.dtype, copy=False)
#        return self._indptr
#
#    @indptr.setter
#    def indptr(self, val):
#        val = np.asarray(val)
#        if val.shape != (2,):
#            raise ValueError("'uni' format indptr must have shape (2,)")
#        if val[0] != 0:
#            raise ValueError("'uni' format indptr must have first entry 0")
#        self._indptr = val

    def _swap(self, x):
        """swap 2-tuple if csc format otherwise leave untouched."""
        return x

    @property
    def _shape_as_2d(self):
        s = self._shape
        return (1, s[-1]) if len(s) == 1 else s

    def __init__(self, arg1, shape=None, dtype=None, copy=False):
        _data_matrix.__init__(self)

        self.indptr = np.zeros(2, dtype='i')

        if issparse(arg1):
            if arg1.format == self.format:
                if copy:
                    arg1 = arg1.copy()
                self._set_self(arg1.data, arg1.indices, arg1.indptr, arg1.shape)
            elif arg1.ndim == 1 or 1 in arg1.shape:
                dim = tuple(i for i, s in enumerate(arg1.shape) if s != 1)
                if len(dim) != 1:
                    raise  ValueError("input shape doesn't match 1d")
                d = dim[0]
                argfmt = arg1.format
                if argfmt == 'coo':
                    indices = arg1.indices[d]
                elif argfmt in ('csr', 'csc'):
                    if d == arg1._swap((1, 0))[0]:
                        indices = arg1.indices
                    else:
                        idx_dtype = self._get_index_dtype(maxval=arg1.shape[0])
                        indices = np.diff(arg1.indptr).nonzero()[0].astype(idx_dtype)
                elif argfmt == 'dok':
                    indices = np.fromiter(arg1._dict)
                else:
                    if d == 0:
                        arg1 = arg1.tocsr()
                    else:
                        arg1 = arg1.tocsc()
                    indices = arg1.indices

                indptr = np.array([0, len(indices)], dtype=indices.dtype)

                if dtype is None:
                    dtype = arg1.dtype
                if argfmt == 'dok':
                    data = np.fromiter(arg1._dict.values()).astype(dtype, copy=False)
                else:
                    data = arg1.data.astype(dtype, copy=False)
                self._set_self(data, indices, indptr, (arg1.shape[d],))
            else:
                raise ValueError("uni_array input not convertable to 1d")

        elif isinstance(arg1, tuple):
            if isshape(arg1, allow_ndim=True):
                # Its a tuple of matrix dimensions (N,)
                # create empty matrix
                shape = check_shape(arg1, allow_ndim=True)
                if len(shape) != 1:
                    raise NotImplementedError("uni_array input shape must be 1d")
                # Select index dtype large enough to pass array and
                # scalar parameters to sparsetools
                idx_dtype = self._get_index_dtype(maxval=shape[0])
                self.indices = np.zeros(0, idx_dtype)
                self.indptr = np.zeros(2, idx_dtype)
                self.data = np.zeros(0, getdtype(dtype, default=float))
                self._shape = shape
            else:
                if len(arg1) == 2:
                    # (data, indices) format
                    try:
                        data, indices = arg1
                    except (TypeError, ValueError) as e:
                        raise TypeError('invalid input format') from e

                    if shape is None:
                        if any(len(idx) == 0 for idx in indices):
                            raise ValueError('cannot infer shape. input has 0 size')
                        if len(indices) != 1:
                            raise ValueError('uni_array (data, indices) input not 1d')
                        shape = (operator.index(np.max(indices[0])) + 1,)
                    self._shape = check_shape(shape, allow_ndim=True)

                    idx_dtype = self._get_index_dtype(indices, maxval=max(self.shape))
                    self.indices = np.array(indices[0], copy=copy, dtype=idx_dtype)
                    self.indptr = np.array([0, len(self.indices)], idx_dtype)
                    self.data = np.array(data, copy=copy, dtype=dtype)

                elif len(arg1) == 3:
                    # (data, indices, indptr) format
                    try:
                        data, indices, indptr = arg1
                    except (TypeError, ValueError) as e:
                        raise TypeError('invalid input format') from e
                    if len(indptr) != 2:
                        if indices.count_nonzero() == 0:
                            # read as if csc format
                            indices = np.diff(indptr).nonzero()[0]
                        else:
                            raise ValueError('input does not represent a 1d array')

                    if shape is None:
                        if len(indices) == 0:
                            raise ValueError('cannot infer shape. input has 0 size')
                        shape = (operator.index(np.max(indices)) + 1,)

                    if 0 not in shape:
                        maxval = max(shape)
                    else:
                        maxval = None
                    idx_dtype = self._get_index_dtype(indices, maxval=maxval)

                    self.indices = np.array(indices, copy=copy, dtype=idx_dtype)
                    self.indptr = np.array(indptr, copy=copy, dtype=idx_dtype)
                    self.data = np.array(data, copy=copy, dtype=dtype)
                else:
                    raise ValueError("unrecognized uni_array input")

                self.has_canonical_format = False
        else:
            # must be dense
            try:
                arg1 = np.asarray(arg1)
            except Exception as e:
                raise ValueError("unrecognized uni_array input") from e
            if arg1.ndim > 1:
                arg1 = arg1.squeeze()
                if arg1.ndim != 1:
                    raise ValueError('uni_array input not 1d')
            if shape is None:
                shape = arg1.shape
            self.indices = np.atleast_1d(arg1).nonzero()[0]
            self.indptr = np.array([0, len(self.indices)], dtype=self.indices.dtype)
            self.data = arg1[self.indices]

        if shape is not None:
            self._shape = check_shape(shape, allow_ndim=True)
        elif self._shape is None:
            # shape not already set, try to infer dimensions
            try:
                N = self.indices.max() + 1
            except Exception as e:
                raise ValueError('unable to infer matrix dimensions') from e
            new_shape = (N,)
            self._shape = check_shape(new_shape, allow_ndim=True)

        if dtype is not None:
            self.data = self.data.astype(dtype, copy=False)

        self.check_format(full_check=False)

    def _set_self(self, data, indices, indptr, shape):
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self._shape = check_shape(shape, allow_ndim=True)

    def _getnnz(self, axis=None):
        if axis is None or axis in (-1, 0):
            return int(self.indptr[-1])
        raise ValueError('axis out of bounds')

    _getnnz.__doc__ = _spbase._getnnz.__doc__


    ###########################
    # Multiplication handlers #
    ###########################

    def _mul_vector(self, other):
        M, N = self._shape_as_2d

        # output array
        result = np.zeros(M, dtype=upcast_char(self.dtype.char, other.dtype.char))
        csr_matvec(M, N, self.indptr, self.indices, self.data, other, result)
        return result if self.ndim > 1 else result.reshape(())

    def _mul_multivector(self, other):
        M, N = self._shape_as_2d
        n_vecs = other.shape[-1]  # number of column vectors

        result = np.zeros((M, n_vecs),
                          dtype=upcast_char(self.dtype.char, other.dtype.char))
        csr_matvecs(M, N, n_vecs, self.indptr, self.indices, self.data,
                    other.ravel(), result.ravel())
        return result if self.ndim > 1 else result.reshape((n_vecs,))

    def _mul_sparse_matrix(self, other):
        """__matmul__ with another sparse matrix or array"""
        M, K1 = self._shape_as_2d
        K2, N = other.shape if other.ndim == 2 else (other.shape[-1], 1)
        if other.ndim == 2:
            new_shape = (N,)
        else:
            new_shape = ()

        if other.ndim == 1:
            if other.format != 'uni':
                other = self.__class__(other, copy=False)  # convert to this format
            # TODO uni @ uni requires converting one to csc format?
            Adata, Aind = self.data, self.indices
            Bdata, Bind = other.data, other.indices
            res = sum(
                Av*Bv
                for k, Av in zip(Aind, Adata)
                for j, Bv in zip(Bind, Bdata)
                if k == j
            )
            return np.array(res, dtype=self.dtype)
        else:
            other = self._csr_container(other, copy=False)

        idx_dtype = self._get_index_dtype((self.indptr, self.indices,
                                     other.indptr, other.indices))

        fn = getattr(_sparsetools, 'csr_matmat_maxnnz')
        nnz = fn(M, N,
                 np.asarray(self.indptr, dtype=idx_dtype),
                 np.asarray(self.indices, dtype=idx_dtype),
                 np.asarray(other.indptr, dtype=idx_dtype),
                 np.asarray(other.indices, dtype=idx_dtype))
        if nnz == 0:
            if new_shape == ():
                return np.array(0, dtype=upcast(self.dtype, other.dtype))
            return self.__class__(new_shape, dtype=upcast(self.dtype, other.dtype))

        idx_dtype = self._get_index_dtype((self.indptr, self.indices,
                                     other.indptr, other.indices),
                                    maxval=nnz)

        indptr = np.empty(2, dtype=idx_dtype)
        indices = np.empty(nnz, dtype=idx_dtype)
        data = np.empty(nnz, dtype=upcast(self.dtype, other.dtype))

        fn = getattr(_sparsetools, 'csr_matmat')
        fn(M, N, np.asarray(self.indptr, dtype=idx_dtype),
           np.asarray(self.indices, dtype=idx_dtype),
           self.data,
           np.asarray(other.indptr, dtype=idx_dtype),
           np.asarray(other.indices, dtype=idx_dtype),
           other.data,
           indptr, indices, data)

        if new_shape == ():
            return data.ravel()
        return self.__class__((data, indices, indptr), shape=new_shape)


    #####################
    # Reduce operations #
    #####################

    def sum(self, axis=None, dtype=None, out=None):
        validateaxis(axis)
        if axis not in (None, 0, -1):
            raise ValueError(f'axis not valid. {axis} is not None, 0 or -1')
        res = self.data.sum().astype(dtype)
        if out is not None:
            if np.prod(np.array(out.shape)) != 1:
                raise ValueError("dimensions do not match")
            out[...] = res
        return res

    sum.__doc__ = _spbase.sum.__doc__

    def _minor_reduce(self, ufunc, data=None):
        """Reduce nonzeros with a ufunc over the minor axis when non-empty

        Can be applied to a function of self.data by supplying data parameter.

        Warning: this does not call sum_duplicates()

        Returns
        -------
        major_index : array of ints
            Major indices where nonzero

        value : array of self.dtype
            Reduce result for nonzeros in each major_index
        """
        if data is None:
            data = self.data
        major_index = np.flatnonzero(np.diff(self.indptr))
        value = ufunc.reduceat(data,
                               downcast_intp_index(self.indptr[major_index]))
        return major_index, value

    #######################
    # Getting and Setting #
    #######################

    def _get_int(self, idx):
        if 0 <= idx <= self.shape[-1]:
            spot = np.flatnonzero(self.indices == idx)
            if spot.size:
                return self.data[spot[0]]
            return self.data.dtype.type(0)
        raise IndexError(f'index ({idx}) out of range')

    def _get_slice(self, idx):
        if idx == slice(None):
            return self.copy()
        if idx.step in (1, None):
            major, minor = 0, idx
            ret = self._get_submatrix(major, minor, copy=True)
            assert False, ("Got HERE (in uni._get_slice). ret: ", ret)
            return ret.reshape(ret.shape[-1])

        return self._minor_slice(idx)

    def _get_array(self, idx):
        idx = np.asarray(idx)
        idx_dtype = self.indices.dtype
        M, N = self._shape_as_2d
        row = np.zeros_like(idx, dtype=idx_dtype)
        major, minor = self._swap((row, idx))
        major = np.asarray(major, dtype=idx_dtype)
        minor = np.asarray(minor, dtype=idx_dtype)
        if minor.size == 0:
            return self.__class__([], dtype=self.dtype)
        new_shape = minor.shape if minor.shape[0] > 1 else (minor.shape[-1],)

        val = np.empty(major.size, dtype=self.dtype)
        csr_sample_values(M, N, self.indptr, self.indices, self.data,
                          major.size, major.ravel(), minor.ravel(), val)
        return self.__class__(val.reshape(new_shape))

    def _major_index_fancy(self, idx):
        """Index along the major axis where idx is an array of ints.
        """
        idx_dtype = self.indices.dtype
        indices = np.asarray(idx, dtype=idx_dtype).ravel()

        _, N = self._shape_as_2d
        M = len(indices)
        new_shape = (M,)
        if M == 0:
            return self.__class__(new_shape, dtype=self.dtype)

        row_nnz = self.indptr[indices + 1] - self.indptr[indices]
        idx_dtype = self.indices.dtype
        res_indptr = np.zeros(M + 1, dtype=idx_dtype)
        np.cumsum(row_nnz, out=res_indptr[1:])

        nnz = res_indptr[-1]
        res_indices = np.empty(nnz, dtype=idx_dtype)
        res_data = np.empty(nnz, dtype=self.dtype)
        csr_row_index(M, indices, self.indptr, self.indices, self.data,
                      res_indices, res_data)

        return self.__class__((res_data, res_indices, res_indptr),
                              shape=new_shape, copy=False)

    def _major_slice(self, idx, copy=False):
        """Index along the major axis where idx is a slice object.
        """
        if idx == slice(None):
            return self.copy() if copy else self

        M, N = self._shape_as_2d
        start, stop, step = idx.indices(M)
        M = len(range(start, stop, step))
        new_shape = (M,)
        if M == 0:
            return self.__class__(new_shape, dtype=self.dtype)

        # Work out what slices are needed for `row_nnz`
        # start,stop can be -1, only if step is negative
        start0, stop0 = start, stop
        if stop == -1 and start >= 0:
            stop0 = None
        start1, stop1 = start + 1, stop + 1

        row_nnz = self.indptr[start1:stop1:step] - \
            self.indptr[start0:stop0:step]
        idx_dtype = self.indices.dtype
        res_indptr = np.zeros(M+1, dtype=idx_dtype)
        np.cumsum(row_nnz, out=res_indptr[1:])

        if step == 1:
            all_idx = slice(self.indptr[start], self.indptr[stop])
            res_indices = np.array(self.indices[all_idx], copy=copy)
            res_data = np.array(self.data[all_idx], copy=copy)
        else:
            nnz = res_indptr[-1]
            res_indices = np.empty(nnz, dtype=idx_dtype)
            res_data = np.empty(nnz, dtype=self.dtype)
            csr_row_slice(start, stop, step, self.indptr, self.indices,
                          self.data, res_indices, res_data)

        return self.__class__((res_data, res_indices, res_indptr),
                              shape=new_shape, copy=False)

    def _minor_index_fancy(self, idx):
        """Index along the minor axis where idx is an array of ints.
        """
        idx_dtype = self.indices.dtype
        idx = np.asarray(idx, dtype=idx_dtype).ravel()

        M, N = self._shape_as_2d
        k = len(idx)
        new_shape = (k,)
        if k == 0:
            return self.__class__(new_shape, dtype=self.dtype)

        # pass 1: count idx entries and compute new indptr
        col_offsets = np.zeros(N, dtype=idx_dtype)
        res_indptr = np.empty_like(self.indptr)
        csr_column_index1(k, idx, M, N, self.indptr, self.indices,
                          col_offsets, res_indptr)

        # pass 2: copy indices/data for selected idxs
        col_order = np.argsort(idx).astype(idx_dtype, copy=False)
        nnz = res_indptr[-1]
        res_indices = np.empty(nnz, dtype=idx_dtype)
        res_data = np.empty(nnz, dtype=self.dtype)
        csr_column_index2(col_order, col_offsets, len(self.indices),
                          self.indices, self.data, res_indices, res_data)
        return self.__class__((res_data, res_indices, res_indptr),
                              shape=new_shape, copy=False)

    def _minor_slice(self, idx, copy=False):
        """Index along the minor axis where idx is a slice object.
        """
        if idx == slice(None):
            return self.copy() if copy else self

        M, N = self._shape_as_2d
        start, stop, step = idx.indices(N)
        newN = len(range(start, stop, step))
        if newN == 0:
            return self.__class__((newN,), dtype=self.dtype)
        if step == 1:
            return self._get_submatrix(minor=idx, copy=copy)
        # TODO: don't fall back to fancy indexing here
        return self._minor_index_fancy(np.arange(start, stop, step))

    def _get_submatrix(self, major=None, minor=None, copy=False):
        """Return a submatrix of this matrix.

        major, minor: None, int, or slice with step 1
        """
        M, N = self._shape_as_2d
        i0, i1 = _process_slice(major, M)
        j0, j1 = _process_slice(minor, N)

        if i0 == 0 and j0 == 0 and i1 == M and j1 == N:
            return self.copy() if copy else self

        indptr, indices, data = get_csr_submatrix(
            M, N, self.indptr, self.indices, self.data, i0, i1, j0, j1)

        shape = (j1 - j0,)
        return self.__class__((data, indices, indptr), shape=shape,
                              dtype=self.dtype, copy=False)

    def _set_int(self, idx, x):
        self._set_many(0, idx, x)

    def _set_array(self, idx, x):
        major, minor = self._swap((np.zeros_like(idx), idx))
        broadcast = x.shape[-1] == 1 and minor.shape[-1] != 1
        if broadcast:
            x = np.repeat(x.data, idx.shape[-1])
        self._set_many(major, minor, x)

    def _setdiag(self, values, k):
        raise ValueError("_setdiag requires an array of at least two dimensions")

    ######################
    # Conversion methods #
    ######################

    def tocoo(self, copy=True):
        if copy:
            return self._coo_container((self.data.copy(), (self.indices.copy(),)),
                                       self.shape)
        return self._coo_container((self.data, (self.indices,)), self.shape)

    tocoo.__doc__ = _spbase.tocoo.__doc__

#    def toarray(self, order=None, out=None):
#        # order and out are ignored here. Easy to include by using
#        # inherited method _compressed.toarray if it used shape_as_2d
#        out = np.zeros(self.shape, dtype=self.dtype, order=order)
#        M, N = self._shape_as_2d
#        csr_todense(M, N, self.indptr, self.indices, self.data, out)
#        return out
#
#    toarray.__doc__ = _spbase.toarray.__doc__
