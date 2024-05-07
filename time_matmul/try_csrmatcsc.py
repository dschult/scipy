import scipy as sp; import numpy as np
upcast = sp.sparse._sputils.upcast


def csr_matmul(A, B):
    M, K1 = A.shape if A.ndim==2 else (1, A.shape[0])
    K2, N = B.shape if B.ndim==2 else (B.shape[0], 1)
    # prep
#    A.eliminate_zeros()
#    A.sum_duplicates()
#    A.sort_indices()
#    B.eliminate_zeros()
#    B.sum_duplicates()
#    B.sort_indices()
    assert A.format == 'csr'
    if B.format == 'csc':
        mat_maxnnz = sp.sparse._sparsetools.csr_matcsc_maxnnz
        matmul = sp.sparse._sparsetools.csr_matcsc
    elif B.format == 'csr':
        mat_maxnnz = sp.sparse._sparsetools.csr_matmat_maxnnz
        matmul = sp.sparse._sparsetools.csr_matmat
    else:
        assert B.format in ("csc", "csr")

    nnz = mat_maxnnz(M, N, A.indptr, A.indices, B.indptr, B.indices)
    idx_dtype = A._get_index_dtype(
        (A.indptr, A.indices, B.indptr, B.indices), maxval=nnz
    )

    Cptr= np.empty(M + 1, dtype=idx_dtype)
    Cind= np.empty(nnz, dtype=idx_dtype)
    Cdat = np.empty(nnz, dtype=upcast(A.dtype, B.dtype))
    matmul(
        M, N, A.indptr, A.indices, A.data, B.indptr, B.indices, B.data, Cptr, Cind, Cdat
    )
    result = sp.sparse.csr_array((Cdat, Cind, Cptr), shape=(M, N))
    return result

#def csr_matmul_csc(A, B):
#
#
#def csr_matmul_csr(A, B):
#    M, K1 = A.shape if A.ndim==2 else (1, A.shape[0])
#    K2, N = B.shape if B.ndim==2 else (B.shape[0], 1)
#    # prep
#    A.eliminate_zeros()
#    A.sum_duplicates()
#    A.sort_indices()
#    B.eliminate_zeros()
#    B.sum_duplicates()
#    B.sort_indices()
#    assert A.format == 'csr'
#    assert B.format == 'csr'
#    nnz = sp.sparse._sparsetools.csr_matmat_maxnnz(
#        M, N, A.indptr, A.indices, B.indptr, B.indices
#    )
#    idx_dtype = A._get_index_dtype((A.indptr, A.indices, B.indptr, B.indices), maxval=nnz)
#
#    Cptr= np.empty(M + 1, dtype=idx_dtype)
#    Cind= np.empty(nnz, dtype=idx_dtype)
#    Cdat = np.empty(nnz, dtype=upcast(A.dtype, B.dtype))
#    sp.sparse._sparsetools.csr_matmat(
#        M, N, A.indptr, A.indices, A.data, B.indptr, B.indices, B.data, Cptr, Cind, Cdat
#    )
#    result = sp.sparse.csr_array((Cdat, Cind, Cptr), shape=(M,N))
#    return result
