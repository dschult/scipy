import scipy as sp; import numpy as np
upcast = sp.sparse._sputils.upcast

A = np.arange(15).reshape((3,5))
Asp = sp.sparse.csr_array(A)
Bsp = Asp.T

M, N = Asp.shape


nnz = sp.sparse._sparsetools.csr_matcsc_maxnnz(M,M,Asp.indptr, Asp.indices, Bsp.indptr,Bsp.indices)
#Cptr,Cind,Cdat = np.empty(M+1, dtype=np.int32), np.empty(nnz, dtype=np.int32), np.empty(nnz, dtype=Asp.dtype)
Cptr= np.empty(M+1, dtype=np.int32)
Cind= np.empty(nnz, dtype=np.int32)
Cdat = np.empty(nnz, dtype=Asp.dtype)
sp.sparse._sparsetools.csr_matcsc(M,M,
                    Asp.indptr, Asp.indices,Asp.data,
                    Bsp.indptr,Bsp.indices,Bsp.data,
                    Cptr,Cind,Cdat)

#print("Cptr: ",Cptr)
#print("Cind: ",Cind)
#print("Cdat: ",Cdat)

def csr_matmul_csc(A, B):
    M,K1 = A.shape if A.ndim==2 else (1,A.shape[0])
    K2,N = B.shape if B.ndim==2 else (B.shape[0],1)
    # prep
    A.eliminate_zeros()
    A.sum_duplicates()
    A.sort_indices()
    B.eliminate_zeros()
    B.sum_duplicates()
    B.sort_indices()
    assert A.format == 'csr'
    assert B.format == 'csc'
    nnz = sp.sparse._sparsetools.csr_matcsc_maxnnz(M,N,A.indptr, A.indices, B.indptr,B.indices)

    idx_dtype = A._get_index_dtype((A.indptr, A.indices, B.indptr, B.indices), maxval=nnz)

#    print("M: ", M, " A.shape, B.shape: ",A.shape, B.shape)
    Cptr= np.empty(M+1, dtype=idx_dtype)
    Cind= np.empty(nnz, dtype=idx_dtype)
    Cdat = np.empty(nnz, dtype=upcast(A.dtype, B.dtype))
    sp.sparse._sparsetools.csr_matcsc(
        M,N,A.indptr, A.indices, A.data, B.indptr, B.indices, B.data, Cptr, Cind, Cdat
    )
#    print("M,N): ",M, N)
#    print("Cptr[-1] and len(Cind) and len(Cdat): ", Cptr[-1], len(Cind), len(Cdat))
#    print(Cptr)
    result = sp.sparse.csr_array((Cdat, Cind, Cptr), shape=(M,N))
#    result.prune()
    return result

def csr_matmul_csr(A, B):
    M,K1 = A.shape if A.ndim==2 else (1,A.shape[0])
    K2,N = B.shape if B.ndim==2 else (B.shape[0],1)
    # prep
    A.eliminate_zeros()
    A.sum_duplicates()
    A.sort_indices()
    B.eliminate_zeros()
    B.sum_duplicates()
    B.sort_indices()
    assert A.format == 'csr'
    assert B.format == 'csr'
    nnz = sp.sparse._sparsetools.csr_matmat_maxnnz(M,N,A.indptr, A.indices, B.indptr,B.indices)

    idx_dtype = A._get_index_dtype((A.indptr, A.indices, B.indptr, B.indices), maxval=nnz)

#    print("M: ", M, " A.shape, B.shape: ",A.shape, B.shape)
    Cptr= np.empty(M+1, dtype=idx_dtype)
    Cind= np.empty(nnz, dtype=idx_dtype)
    Cdat = np.empty(nnz, dtype=upcast(A.dtype, B.dtype))
    sp.sparse._sparsetools.csr_matmat(
        M,N,A.indptr, A.indices, A.data, B.indptr, B.indices, B.data, Cptr, Cind, Cdat
    )
#    print("M,N): ",M, N)
#    print("Cptr[-1] and len(Cind) and len(Cdat): ", Cptr[-1], len(Cind), len(Cdat))
#    print(Cptr)
    result = sp.sparse.csr_array((Cdat, Cind, Cptr), shape=(M,N))
#    result.prune()
    return result

M, N, K, density, seed = 3, 4, 5, 0.1, 42
rng = np.random.default_rng(seed)
A = sp.sparse.random(M, K, density=density, random_state=rng).tocsr()
B = sp.sparse.random(K, N, density=density, random_state=rng).tocsc()
Bcsr = B.tocsr()
#print("A shape,nnz: ",A.shape,A.nnz,"B shape, nnz: ",B.shape,B.nnz)

Cnew = csr_matmul_csc(A, B)
#print(Cnew.indptr)
#print(Cnew.indices)
#print(Cnew.data)
Ctot = A.toarray() @ B.toarray()
Cold = csr_matmul_csr(A, Bcsr)
#print(Cold.toarray())
#print(Cnew.toarray())
assert np.array_equal(Ctot, Cnew.toarray())
assert np.array_equal(Cold.toarray(), Cnew.toarray())

#%timeit csr_matmul_csc(A, B)  # time csr*csc method
#%timeit A@B  # time current sparse multiply
#%timeit A@B.toarray()  # time csr sparse multiply dense
#%timeit A.toarray()@B.toarray()  # time dense
