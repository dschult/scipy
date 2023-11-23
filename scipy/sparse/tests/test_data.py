import numpy as np
from numpy.testing import assert_equal, assert_array_equal
from scipy.sparse import coo_array, csr_array, csc_array
from pytest import raises as assert_raises

# limit to valid 1d formats that also support minmax
formats = ['coo', 'csr', 'csc']
creators = [coo_array, csr_array, csc_array]


def test_minmax():
    for spcreator in creators:
        D = np.arange(20)

        X = spcreator(D)
        assert_equal(X.min(), 0)
        assert_equal(X.max(), 19)

        D *= -1
        X = spcreator(D)
        assert_equal(X.min(), -19)
        assert_equal(X.max(), 0)

        D += 5
        X = spcreator(D)
        assert_equal(X.min(), -14)
        assert_equal(X.max(), 5)

        # try a fully dense matrix
        X = spcreator(np.arange(1, 10))
        assert_equal(X.min(), 1)
        assert_equal(X.min().dtype, X.dtype)

        X = -X
        assert_equal(X.max(), -1)

        # and a fully sparse matrix
        Z = spcreator(np.zeros(3))
        assert_equal(Z.min(), 0)
        assert_equal(Z.max(), 0)
        assert_equal(Z.max().dtype, Z.dtype)

        # another test
        D = np.arange(20, dtype=float)
        D[2:5] = 0
        X = spcreator(D)
        assert_equal(X.min(), 0)
        assert_equal(X.max(), 19)


def test_minmax_axis():
    for spcreator in creators:
        D = np.arange(5)
        X = spcreator(D)

        axes = [-1, 0]
        for axis in axes:
            assert_array_equal(X.max(axis=axis).toarray(), D.max(axis=axis))
            assert_array_equal(X.min(axis=axis).toarray(), D.min(axis=axis))

        for axis in [-2, 1]:
            assert_raises(ValueError, X.min, axis=axis)
            assert_raises(ValueError, X.max, axis=axis)


def test_numpy_minmax():
    dat = np.array([3, -4, 5, 0])
    for spcreator in creators:
        datsp = spcreator(dat)
        assert_array_equal(np.min(datsp), np.min(dat))
        assert_array_equal(np.max(datsp), np.max(dat))


def test_argmax():
    D1 = np.array([-1, 5, 2, 3])   # min/max not zero
    D2 = np.array([0, 0, -1, -2])  # max is zero
    D3 = np.array([-1, -2, -3, -4]) # completely dense neg
    D4 = np.array([1, 2, 3, 4])    # completely dense pos
    D5 = np.array([1, 2, 0, 0])    # min is zero

    for spcreator in creators:
        for D in [D1, D2, D3, D4, D5]:
            mat = spcreator(D)

            assert_equal(mat.argmax(), np.argmax(D))
            assert_equal(mat.argmin(), np.argmin(D))

            assert_equal(mat.argmax(axis=0), np.argmax(D, axis=0))
            assert_equal(mat.argmin(axis=0), np.argmin(D, axis=0))

            assert_equal(mat.argmax(axis=-1), np.argmax(D, axis=-1))
            assert_equal(mat.argmin(axis=-1), np.argmin(D, axis=-1))
