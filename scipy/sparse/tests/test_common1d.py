"""Test of 1D aspects of sparse array classes"""

import operator
from pytest import raises as assert_raises

import numpy as np
from numpy import zeros, array, dot, asarray, ComplexWarning

from numpy.testing import (assert_equal, assert_array_equal,
        assert_array_almost_equal, assert_almost_equal, assert_,
        assert_allclose,suppress_warnings)

from scipy.sparse import coo_array, dok_array, sparray, issparse
from scipy.sparse._sputils import (supported_dtypes, isscalarlike,
                                   asmatrix, matrix)

import pytest


sup_complex = suppress_warnings()
sup_complex.filter(ComplexWarning)


def assert_array_equal_dtype(x, y, **kwargs):
    assert_(x.dtype == y.dtype)
    assert_array_equal(x, y, **kwargs)


def toarray(a):
    if isinstance(a, np.ndarray) or isscalarlike(a):
        return a
    return a.toarray()


class _TestCommon1D:
    """test common functionality shared by 1D sparse formats"""
    @classmethod
    def init_class(cls):
        # Canonical data.
        cls.dat1d = array([3, 0, 1, 0], 'd')
        cls.datsp = cls.spcreator(cls.dat1d)

        # Some sparse and dense matrices with data for every supported dtype.
        # This set union is a workaround for numpy#6295, which means that
        # two np.int64 dtypes don't hash to the same value.
        cls.checked_dtypes = set(supported_dtypes).union(cls.math_dtypes)
        cls.dat_dtypes = {}
        cls.datsp_dtypes = {}
        for dtype in cls.checked_dtypes:
            cls.dat_dtypes[dtype] = cls.dat1d.astype(dtype)
            cls.datsp_dtypes[dtype] = cls.spcreator(cls.dat1d.astype(dtype))

        # Check that the original data is equivalent to the
        # corresponding dat_dtypes & datsp_dtypes.
        assert_equal(cls.dat1d, cls.dat_dtypes[np.float64])
        assert_equal(cls.datsp.toarray(),
                     cls.datsp_dtypes[np.float64].toarray())

    def test_empty(self):
        # create empty matrices
        assert_equal(self.spcreator((3,)).toarray(), zeros(3))
        assert_equal(self.spcreator((3,)).nnz, 0)
        assert_equal(self.spcreator((3,)).count_nonzero(), 0)

    def test_invalid_shapes(self):
        assert_raises(ValueError, self.spcreator, (-3,))

    def test_repr(self):
        repr(self.datsp)

    def test_str(self):
        str(self.datsp)

    def test_empty_arithmetic(self):
        # Test manipulating empty matrices. Fails in SciPy SVN <= r1768
        shape = (5,)
        for mytype in [np.dtype('int32'), np.dtype('float32'),
                np.dtype('float64'), np.dtype('complex64'),
                np.dtype('complex128')]:
            a = self.spcreator(shape, dtype=mytype)
            b = a + a
            c = 2 * a
            d = a @ a.tocsc()
            e = a @ a.tocsr()
            f = a @ a.tocoo()
            for m in [a,b,c,d,e,f]:
                assert_equal(m.toarray(), a.toarray()@a.toarray())
                assert_equal(m.dtype, mytype)
                assert_equal(m.toarray().dtype, mytype)

    def test_abs(self):
        A = array([-1, 0, 17, 0, -5, 0, 1, -4, 0, 0, 0, 0], 'd')
        assert_equal(abs(A), abs(self.spcreator(A)).toarray())

    def test_round(self):
        decimal = 1
        A = array([-1.35, 0.56, 17.25, -5.98], 'd')
        assert_equal(np.around(A, decimals=decimal),
                     round(self.spcreator(A), ndigits=decimal).toarray())

    def test_elementwise_power(self):
        A = array([-4, -3, -2, -1, 0, 1, 2, 3, 4], 'd')
        assert_equal(np.power(A, 2), self.spcreator(A).power(2).toarray())

        #it's element-wise power function, input has to be a scalar
        assert_raises(NotImplementedError, self.spcreator(A).power, A)

    def test_neg(self):
        A = array([-1, 0, 17, 0, -5, 0, 1, -4, 0, 0, 0, 0], 'd')
        assert_equal(-A, (-self.spcreator(A)).toarray())

    def test_real(self):
        D = array([[1 + 3j, 2 - 4j]])
        A = self.spcreator(D)
        assert_equal(A.real.toarray(), D.real)

    def test_imag(self):
        D = array([[1 + 3j, 2 - 4j]])
        A = self.spcreator(D)
        assert_equal(A.imag.toarray(), D.imag)

    def test_reshape(self):
        x = self.spcreator([1, 0, 7, 0, 0, 0, 0, -3, 0, 0, 0, 5])
        y = x.reshape((4, 3))
        desired = [[1, 0, 7], [0, 0, 0], [0, -3, 0], [0, 0, 5]]
        assert_array_equal(y.toarray(), desired)

        y = x.reshape((12,))
        assert_(y is x)

        y = x.reshape(12)
        assert_array_equal(y.toarray(), x.toarray())

    def test_sum(self):
        np.random.seed(1234)
        dat_1 = array([0, 1, 2, 3, -4, 5, -6, 7, 9])
        dat_2 = np.random.rand(5)
        dat_3 = np.array([])
        dat_4 = np.zeros((40, ))
        matrices = [dat_1, dat_2, dat_3, dat_4]

        for m in matrices:
            dat = array(m)
            datsp = self.spcreator(dat)
            with np.errstate(over='ignore'):
                assert_(np.isscalar(datsp.sum()))
                assert_array_almost_equal(dat.sum(), datsp.sum())
                assert_array_almost_equal(dat.sum(axis=None), datsp.sum(axis=None))
                assert_array_almost_equal(dat.sum(axis=0), datsp.sum(axis=0))
                assert_array_almost_equal(dat.sum(axis=-1), datsp.sum(axis=-1))

        # test out parameter
        datsp.sum(axis=0, out=np.zeros(()))

    def test_sum_invalid_params(self):
        out = np.zeros((3,))  # wrong size for out
        dat = array([0, 1, 2])
        datsp = self.spcreator(dat)

        assert_raises(ValueError, datsp.sum, axis=1)
        assert_raises(TypeError, datsp.sum, axis=(0, 1))
        assert_raises(TypeError, datsp.sum, axis=1.5)
        assert_raises(ValueError, datsp.sum, axis=0, out=out)

    def test_numpy_sum(self):
        dat = array([0, 1, 2])
        datsp = self.spcreator(dat)

        dat_sum = np.sum(dat)
        datsp_sum = np.sum(datsp)

        assert_array_almost_equal(dat_sum, datsp_sum)

    def test_mean(self):
        keepdims = not isinstance(self, sparray)
        dat = array([0, 1, 2])
        datsp = self.spcreator(dat)

        assert_array_almost_equal(dat.mean(), datsp.mean())
        assert_(np.isscalar(datsp.mean(axis=None)))
        assert_array_almost_equal(
            dat.mean(axis=None, keepdims=keepdims), datsp.mean(axis=None)
        )
        assert_array_almost_equal(
            dat.mean(axis=0, keepdims=keepdims), datsp.mean(axis=0)
        )
        assert_array_almost_equal(
            dat.mean(axis=-1, keepdims=keepdims), datsp.mean(axis=-1)
        )

        with pytest.raises(ValueError, match='axis'):
            datsp.mean(axis=1)
        with pytest.raises(ValueError, match='axis'):
            datsp.mean(axis=-2)

    def test_mean_invalid_params(self):
        out = asarray(np.zeros((1, 3)))
        dat = array([[0, 1, 2],
                     [3, -4, 5],
                     [-6, 7, 9]])
        datsp = self.spcreator(dat)

        assert_raises(ValueError, datsp.mean, axis=3)
        assert_raises(TypeError, datsp.mean, axis=(0, 1))
        assert_raises(TypeError, datsp.mean, axis=1.5)
        assert_raises(ValueError, datsp.mean, axis=1, out=out)

    def test_sum_dtype(self):
        dat = array([0, 1, 2])
        datsp = self.spcreator(dat)
        print("datsp format:", datsp.format)

        def check_dtype(dtype):
            print("datsp dtype:", datsp.dtype, datsp)
            print("in check_dtype: ", dtype, "datsp dtype:", datsp.dtype, "dat dtype:", dat.dtype)
            dat_sum = dat.sum(dtype=dtype)
            datsp_sum = datsp.sum(dtype=dtype)
            print("dtype: ", dtype, "dat_sum dtype:", dat_sum.dtype, "datsp_sum dtype:", datsp_sum.dtype)

            assert_array_almost_equal(dat_sum, datsp_sum)
            assert_equal(dat_sum.dtype, datsp_sum.dtype)

        for dtype in self.checked_dtypes:
            check_dtype(dtype)

    def test_mean_dtype(self):
        dat = array([0, 1, 2])
        datsp = self.spcreator(dat)
        print("datsp format:", datsp.format)

        def check_dtype(dtype):
            print("datsp dtype:", datsp.dtype, datsp)
            print("in check_dtype: ", dtype, "datsp dtype:", datsp.dtype, "dat dtype:", dat.dtype)
            dat_mean = dat.mean(dtype=dtype)
            datsp_mean = datsp.mean(dtype=dtype)
            print("dtype: ", dtype, "dat_mean dtype:", dat_mean.dtype, "datsp_mean dtype:", datsp_mean.dtype)

            assert_array_almost_equal(dat_mean, datsp_mean)
            assert_equal(dat_mean.dtype, datsp_mean.dtype)

        for dtype in self.checked_dtypes:
            check_dtype(dtype)

    def test_mean_out(self):
        dat = array([0, 1, 2])
        datsp = self.spcreator(dat)

        dat_out = array([0])
        datsp_out = array([0])

        dat.mean(out=dat_out, keepdims=True)
        datsp.mean(out=datsp_out)
        assert_array_almost_equal(dat_out, datsp_out)

        dat.mean(axis=0, out=dat_out, keepdims=True)
        datsp.mean(axis=0, out=datsp_out)
        assert_array_almost_equal(dat_out, datsp_out)

    def test_numpy_mean(self):
        dat = array([0, 1, 2])
        datsp = self.spcreator(dat)

        dat_mean = np.mean(dat)
        datsp_mean = np.mean(datsp)

        assert_array_almost_equal(dat_mean, datsp_mean)
        assert_equal(dat_mean.dtype, datsp_mean.dtype)

    @sup_complex
    def test_from_array(self):
        A = array([2,3,4])
        assert_array_equal(self.spcreator(A).toarray(), A)

        A = array([1.0 + 3j, 0, -1])
        assert_array_equal(self.spcreator(A).toarray(), A)
        assert_array_equal(self.spcreator(A, dtype='int16').toarray(),A.astype('int16'))

    @sup_complex
    def test_from_list(self):
        A = [2,3,4]
        assert_array_equal(self.spcreator(A).toarray(), A)

        A = [1.0 + 3j, 0, -1]
        assert_array_equal(self.spcreator(A).toarray(), array(A))
        assert_array_equal(
            self.spcreator(A, dtype='int16').toarray(), array(A).astype('int16')
        )

    @sup_complex
    def test_from_sparse(self):
        D = array([1,0,0])
        S = coo_array(D)
        assert_array_equal(self.spcreator(S).toarray(), D)
        S = self.spcreator(D)
        assert_array_equal(self.spcreator(S).toarray(), D)

        D = array([1.0 + 3j, 0, -1])
        S = coo_array(D)
        assert_array_equal(self.spcreator(S).toarray(), D)
        assert_array_equal(self.spcreator(S, dtype='int16').toarray(), D.astype('int16'))
        S = self.spcreator(D)
        assert_array_equal(self.spcreator(S).toarray(), D)
        assert_array_equal(self.spcreator(S, dtype='int16').toarray(), D.astype('int16'))

    def test_toarray(self):
        # Check C- or F-contiguous (default).
        dat = asarray(self.dat1d)
        chk = self.datsp.toarray()
        assert_array_equal(chk, dat)
        assert_(chk.flags.c_contiguous == chk.flags.f_contiguous)
        # Check C-contiguous (with arg).
        chk = self.datsp.toarray(order='C')
        assert_array_equal(chk, dat)
        assert_(chk.flags.c_contiguous)
        assert_(chk.flags.f_contiguous)
        # Check F-contiguous (with arg).
        chk = self.datsp.toarray(order='F')
        assert_array_equal(chk, dat)
        assert_(chk.flags.c_contiguous)
        assert_(chk.flags.f_contiguous)

        # Check with output arg.
        out = np.zeros(self.datsp.shape, dtype=self.datsp.dtype)
        self.datsp.toarray(out=out)
        assert_array_equal(chk, dat)
        # Check that things are fine when we don't initialize with zeros.
        out[...] = 1.
        self.datsp.toarray(out=out)
        assert_array_equal(chk, dat)
        a = array([1., 2., 3., 4.])
        dense_dot_dense = dot(a, dat)
        check = dot(a, self.datsp.toarray())
        assert_array_equal(dense_dot_dense, check)
        b = array([1., 2., 3., 4.])
        dense_dot_dense = dot(dat, b)
        check2 = dot(self.datsp.toarray(), b)
        assert_array_equal(dense_dot_dense, check2)

        # Check bool data works.
        spbool = self.spcreator(dat, dtype=bool)
        arrbool = dat.astype(bool)
        assert_array_equal(spbool.toarray(), arrbool)

    def test_mul_scalar(self):
        def check(dtype):
            dat = self.dat_dtypes[dtype]
            datsp = self.datsp_dtypes[dtype]

            assert_array_equal(dat*2, (datsp*2).toarray())
            assert_array_equal(dat*17.3, (datsp*17.3).toarray())

        for dtype in self.math_dtypes:
            check(dtype)

    def test_rmul_scalar(self):
        def check(dtype):
            dat = self.dat_dtypes[dtype]
            datsp = self.datsp_dtypes[dtype]

            assert_array_equal(2*dat, (2*datsp).toarray())
            assert_array_equal(17.3*dat, (17.3*datsp).toarray())

        for dtype in self.math_dtypes:
            check(dtype)

    def test_add(self):
        def check(dtype):
            dat = self.dat_dtypes[dtype]
            datsp = self.datsp_dtypes[dtype]

            a = dat.copy()
            a[0] = 2.0
            b = datsp
            c = b + a
            assert_array_equal(c, b.toarray() + a)

            # test broadcasting
            # Note: cdnt add nonzero scalar to sparray. Can add len 1 array
            c = b + a[0:1]
            assert_array_equal(c, b.toarray() + a[0])

        for dtype in self.math_dtypes:
            check(dtype)

    def test_radd(self):
        def check(dtype):
            dat = self.dat_dtypes[dtype]
            datsp = self.datsp_dtypes[dtype]

            a = dat.copy()
            a[0] = 2.0
            b = datsp
            c = a + b
            assert_array_equal(c, a + b.toarray())

        for dtype in self.math_dtypes:
            check(dtype)

    def test_sub(self):
        def check(dtype):
            dat = self.dat_dtypes[dtype]
            datsp = self.datsp_dtypes[dtype]

            assert_array_equal((datsp - datsp).toarray(), np.zeros(4))
            assert_array_equal((datsp - 0).toarray(), dat)

            A = self.spcreator([1, -4, 0, 2], dtype='d')
            assert_array_equal((datsp - A).toarray(), dat - A.toarray())
            assert_array_equal((A - datsp).toarray(), A.toarray() - dat)

            # test broadcasting
            assert_array_equal(datsp - dat[0], dat - dat[0])

        for dtype in self.math_dtypes:
            if dtype == np.dtype('bool'):
                # boolean array subtraction deprecated in 1.9.0
                continue

            check(dtype)

    def test_rsub(self):
        def check(dtype):
            dat = self.dat_dtypes[dtype]
            datsp = self.datsp_dtypes[dtype]

            assert_array_equal((dat - datsp),[0,0,0,0])
            assert_array_equal((datsp - dat),[0,0,0,0])
            assert_array_equal((0 - datsp).toarray(), -dat)

            A = self.spcreator([1, -4, 0, 2], dtype='d')
            assert_array_equal((dat - A), dat - A.toarray())
            assert_array_equal((A - dat), A.toarray() - dat)
            assert_array_equal(A.toarray() - datsp, A.toarray() - dat)
            assert_array_equal(datsp - A.toarray(), dat - A.toarray())

            # test broadcasting
            assert_array_equal(dat[:1] - datsp, dat[:1] - dat)

        for dtype in self.math_dtypes:
            if dtype == np.dtype('bool'):
                # boolean array subtraction deprecated in 1.9.0
                continue

            check(dtype)

    def test_add0(self):
        def check(dtype):
            dat = self.dat_dtypes[dtype]
            datsp = self.datsp_dtypes[dtype]

            # Adding 0 to a sparse matrix
            assert_array_equal((datsp + 0).toarray(), dat)
            # use sum (which takes 0 as a starting value)
            sumS = sum([k * datsp for k in range(1, 3)])
            sumD = sum([k * dat for k in range(1, 3)])
            assert_almost_equal(sumS.toarray(), sumD)

        for dtype in self.math_dtypes:
            check(dtype)

    def test_elementwise_multiply(self):
        # real/real
        A = array([4,0,9])
        B = array([0,7,-1])
        Asp = self.spcreator(A)
        Bsp = self.spcreator(B)
        assert_almost_equal(Asp.multiply(Bsp).toarray(), A*B)  # sparse/sparse
        assert_almost_equal(Asp.multiply(B).toarray(), A*B)  # sparse/dense

        # complex/complex
        C = array([1-2j,0+5j,-1+0j])
        D = array([5+2j,7-3j,-2+1j])
        Csp = self.spcreator(C)
        Dsp = self.spcreator(D)
        assert_almost_equal(Csp.multiply(Dsp).toarray(), C*D)  # sparse/sparse
        assert_almost_equal(Csp.multiply(D).toarray(), C*D)  # sparse/dense

        # real/complex
        assert_almost_equal(Asp.multiply(Dsp).toarray(), A*D)  # sparse/sparse
        assert_almost_equal(Asp.multiply(D).toarray(), A*D)  # sparse/dense

    def test_elementwise_multiply_broadcast(self):
        A = array([4])
        B = array([[-9]])
        C = array([1,-1,0])
        D = array([[7,9,-9]])
        E = array([[3],[2],[1]])
        F = array([[8,6,3],[-4,3,2],[6,6,6]])
        G = [1, 2, 3]
        H = np.ones((3, 4))
        J = H.T
        K = array([[0]])
        L = array([[[1,2],[0,1]]])

        # Some arrays can't be cast as spmatrices (A,C,L) so leave
        # them out.
        Asp = self.spcreator(A)
        Bsp = self.spcreator(B)
        Csp = self.spcreator(C)
        Dsp = self.spcreator(D)
        Esp = self.spcreator(E)
        Fsp = self.spcreator(F)
        Gsp = self.spcreator(G)
        Hsp = self.spcreator(H)
        Hspp = self.spcreator(H[0,None])
        Jsp = self.spcreator(J)
        Jspp = self.spcreator(J[:,0,None])
        Ksp = self.spcreator(K)

        matrices = [A, B, C, D, E, F, G, H, J, K, L]
        spmatrices = [Asp, Bsp, Csp, Dsp, Esp, Fsp, Gsp, Hsp, Hspp, Jsp, Jspp, Ksp]
        sp1dmatrices = [Asp, Csp, Gsp]

        # sparse/sparse
        for i in sp1dmatrices:
            for j in spmatrices:
                try:
                    dense_mult = i.toarray() * j.toarray()
                except ValueError:
                    assert_raises(ValueError, i.multiply, j)
                    continue
                sp_mult = i.multiply(j)
                assert_almost_equal(sp_mult.toarray(), dense_mult)

        # sparse/dense
        for i in sp1dmatrices:
            for j in matrices:
                try:
                    dense_mult = i.toarray() * j
                except TypeError:
                    continue
                except ValueError:
                    assert_raises(ValueError, i.multiply, j)
                    continue
                sp_mult = i.multiply(j)
                if issparse(sp_mult):
                    assert_almost_equal(sp_mult.toarray(), dense_mult)
                else:
                    assert_almost_equal(sp_mult, dense_mult)

    def test_elementwise_divide(self):
        expected = [1, np.nan, 1, np.nan]
        assert_array_equal((self.datsp / self.datsp).toarray(), expected)

        denom = self.spcreator([1, 0, 0, 4],dtype='d')
        expected = [3, np.nan, np.nan, 0]
        assert_array_equal(toarray(self.datsp / denom), expected)

        # complex
        A = array([1-2j, 0+5j, -1+0j])
        B = array([5+2j, 7-3j, -2+1j])
        Asp = self.spcreator(A)
        Bsp = self.spcreator(B)
        assert_almost_equal(toarray(Asp / Bsp), A/B)

        # integer
        A = array([1, 2, 3])
        B = array([0, 1, 2])
        Asp = self.spcreator(A)
        Bsp = self.spcreator(B)
        with np.errstate(divide='ignore'):
            assert_array_equal(toarray(Asp / Bsp), A / B)

        # mismatching sparsity patterns
        A = array([0, 1])
        B = array([1, 0])
        Asp = self.spcreator(A)
        Bsp = self.spcreator(B)
        with np.errstate(divide='ignore', invalid='ignore'):
            assert_array_equal(np.array(toarray(Asp / Bsp)), A / B)

    def test_pow(self):
        A = array([1, 0, 2, 0])
        B = self.spcreator(A)

        # unusual exponents
        with pytest.raises(ValueError, match='negative integer powers'):
            B ** -1
        with pytest.raises(NotImplementedError, match='zero power'):
            B ** 0

        for exponent in [1, 2, 3, 2.2]:
            ret_sp = B**exponent
            ret_np = A**exponent
            assert_array_equal(ret_sp.toarray(), ret_np)
            assert_equal(ret_sp.dtype, ret_np.dtype)

    def test_rmatvec(self):
        M = self.datsp
        assert_array_almost_equal([1,2,3,4] @ M, dot([1,2,3,4], M.toarray()))
        row = array([[1,2,3,4]])
        assert_array_almost_equal(row @ M, row @ M.toarray())

    def test_small_multiplication(self):
        # test that A*x works for x with shape () (1,) (1,1) and (1,0)
        A = self.spcreator([1,2,3])

        assert_(issparse(A * array(1)))
        assert_equal((A * array(1)).toarray(), [1, 2, 3])

        assert_equal(A @ array([1]), array([1, 2, 3]))
        assert_equal(A @ array([[1]]), array([[1], [2], [3]]))
        assert_equal(A @ np.ones((1, 1)), array([[1], [2], [3]]))
        assert_equal(A @ np.ones((1, 0)), np.ones((3, 0)))

    def test_dot_scalar(self):
        M = self.datsp
        scalar = 10
        actual = M.dot(scalar)
        expected = M @ scalar

        assert_allclose(actual.toarray(), expected.toarray())

    def test_matmul(self):
        M = self.spcreator([2,0,3.0])
        B = self.spcreator(array([[0,1],[1,0],[0,2]],'d'))
        col = array([[1,2,3]]).T

        matmul = operator.matmul
        # check matrix-vector
#        print("col: ", col)
#        print("M.toarray",M.toarray())
#        print("M",M._dict,"done")
#        print("M@col: ",M.shape, col.shape, matmul(M, col))
#        print("M.toarray @ col: ",M.toarray().shape, col.shape, matmul(M.toarray(), col))
        assert_array_almost_equal(matmul(M, col), M.toarray() @ col)
#        print("made it past M@col test")
#        print("M.shape", M.shape)
#        print("B.shape", B.shape)

        # check matrix-matrix
        print("M@B: ",M.shape, B.shape)
        assert_array_almost_equal(matmul(M, B).toarray(), (M @ B).toarray())
        assert_array_almost_equal(matmul(M.toarray(), B), (M @ B).toarray())
        assert_array_almost_equal(matmul(M, B.toarray()), (M @ B).toarray())

        # check error on matrix-scalar
        assert_raises(ValueError, matmul, M, 1)
        assert_raises(ValueError, matmul, 1, M)

    def test_matvec(self):
        M = self.spcreator([2,0,3.0])
        col = array([[1,2,3]]).T

        assert_array_almost_equal(M @ col, M.toarray() @ col)

        # check result type
        assert np.ndim(M @ array([1,2,3])) == 0
        assert isinstance(M @ matrix([[1,2,3]]).T, np.ndarray)

        # ensure exception is raised for improper dimensions
        bad_vecs = [array([1,2]), array([1,2,3,4]), array([[1],[2]]),
                    matrix([1,2,3]), matrix([[1],[2]])]
        for x in bad_vecs:
            assert_raises(ValueError, M.__mul__, x)

        # The current relationship between sparse matrix products and array
        # products is as follows:
        assert_array_almost_equal(M@array([1,2,3]), dot(M.toarray(),[1,2,3]))
        assert_array_almost_equal(M@[[1],[2],[3]], asmatrix(dot(M.toarray(),[1,2,3])).T)
        # Note that the result of M * x is dense if x has a singleton dimension.

    def test_transpose(self):
        dat_1 = self.dat1d
        dat_2 = np.array([])
        matrices = [dat_1, dat_2]

        def check(dtype, j):
            dat = array(matrices[j], dtype=dtype)
            datsp = self.spcreator(dat)

            a = datsp.transpose()
            b = dat.transpose()

            assert_array_equal(a.toarray(), b)
            assert_array_equal(a.transpose().toarray(), dat)
            assert_equal(a.dtype, b.dtype)

        for dtype in self.checked_dtypes:
            for j in range(len(matrices)):
                check(dtype, j)

    def test_add_dense(self):
        def check(dtype):
            dat = self.dat_dtypes[dtype]
            datsp = self.datsp_dtypes[dtype]

            # adding a dense matrix to a sparse matrix
            sum1 = dat + datsp
            assert_array_equal(sum1, dat + dat)
            sum2 = datsp + dat
            assert_array_equal(sum2, dat + dat)

        for dtype in self.math_dtypes:
            check(dtype)

    def test_sub_dense(self):
        # subtracting a dense matrix to/from a sparse matrix
        def check(dtype):
            dat = self.dat_dtypes[dtype]
            datsp = self.datsp_dtypes[dtype]

            # Manually add to avoid upcasting from scalar
            # multiplication.
            sum1 = (dat + dat + dat) - datsp
            assert_array_equal(sum1, dat + dat)
            sum2 = (datsp + datsp + datsp) - dat
            assert_array_equal(sum2, dat + dat)

        for dtype in self.math_dtypes:
            if dtype == np.dtype('bool'):
                # boolean array subtraction deprecated in 1.9.0
                continue

            check(dtype)

    # test that __iter__ is compatible with NumPy matrix
    def test_iterator(self):
        B = array(np.arange(5))
        A = self.spcreator(B)

        if A.format not in ['coo', 'dia', 'bsr']:
            for x, y in zip(A, B):
                assert_equal(x, y)

    def test_size_zero_matrix_arithmetic(self):
        # Test basic matrix arithmetic with shapes like (0,0), (10,0),
        # (0, 3), etc.
        mat = array([])
        a = mat.reshape(0)
        d = mat.reshape((1, 0))
        f = np.ones([5, 5])

        asp = self.spcreator(a)
        dsp = self.spcreator(d)

        # matrix product.
        assert_array_equal(asp.dot(asp).toarray(), np.dot(a, a))

        # bad matrix products
        assert_raises(ValueError, asp.dot, f)

        # elemente-wise multiplication
        assert_array_equal(asp.multiply(asp).toarray(), np.multiply(a, a))

        assert_array_equal(asp.multiply(a).toarray(), np.multiply(a, a))

        assert_array_equal(asp.multiply(6).toarray(), np.multiply(a, 6))

        # bad element-wise multiplication
        assert_raises(ValueError, asp.multiply, f)

        # Addition
        assert_array_equal(asp.__add__(asp).toarray(), a.__add__(a))

        # bad addition
        assert_raises(ValueError, asp.__add__, dsp)

    def test_resize(self):
        # resize(shape) resizes the matrix in-place
        D = np.array([1, 0, 3, 4])
        S = self.spcreator(D)
        assert_(S.resize((3,)) is None)
        assert_array_equal(S.toarray(), [1, 0, 3])
        S.resize((5,))
        assert_array_equal(S.toarray(), [1, 0, 3, 0, 0])


class TestDOK1D(_TestCommon1D):
    spcreator = dok_array
    math_dtypes = [np.int_, np.float_, np.complex_]


TestDOK1D.init_class()


class TestCOO1D(_TestCommon1D):
    spcreator = coo_array
    math_dtypes = [np.int_, np.float_, np.complex_]


TestCOO1D.init_class()
