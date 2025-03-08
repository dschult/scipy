import time

import numpy as np
import numpy.random as rng
import scipy as sp


def time_csctools(A, B, rounds=7, numb=100):
    times = []
    for r in range(rounds):
        tottime = 0
        for _ in range(numb):
            a, b = A.copy(), B.copy()
            start = time.perf_counter()
#            a.sort_indices()
#            b.sort_indices()
            a._binopt(b, '_plus_')
            stop = time.perf_counter()
            tottime += stop - start
        times.append(tottime / numb)
    return np.array(times)


def time_csrtools(A, B, rounds=7, numb=100):
    times = []
    for r in range(rounds):
        tottime = 0
        for _ in range(numb):
            a, b = A.copy(), B.copy()
            start = time.perf_counter()
#            a.sort_indices()
#            b.sort_indices()
            a._binopt2(b, '_plus_')
            stop = time.perf_counter()
            tottime += stop - start
        times.append(tottime / numb)
    return np.array(times)


def make_sorted_indices(M=2 ** 10, N=1, density=0.1, rng=rng):
    A = sp.sparse.random_array((M, N), density=density, format='csc',rng=rng)
    return A


def make_shuffled_indices(M=2 ** 10, N=1, density=0.1):
    A = sp.sparse.random_array((M, N), density=density, format='csc')
    for ptr in range(len(A.indptr) - 1):
        start, end = A.indptr[ptr:ptr + 2]
        if end > start:
            inds = A.indices[start:end]
            rng.shuffle(inds)
            A.indices[start:end] = inds
    return A


all_start = time.time()
##  actually do the timing
rounds = 20
number = 100
#density = 1e-5
results = []
for i in range(14, 10, -2):
#    for k in range(4, -1, -1):
    for k in range(14, 9, -4):
        for j in range(2, min(i,10)+1, 3): # density 2 ** -j
            M = 2 ** i
            density = 2 ** -j
            N = 2 ** k
#            nnz = j   # dens=nnz/MN
#            density = j / (M*N)
#            if density > 1:
#                print(f"WARNING: density is {density} > 1. {M=} {N=} {nnz=}")
#                continue
            nnz = round(M * N * density)
            rng=np.random.default_rng(2373648)
            A = make_sorted_indices(M, N, density, rng=rng)
            B = make_sorted_indices(M, N, density, rng=rng)
            #A = make_shuffled_indices(M, N, density)
            #B = make_shuffled_indices(M, N, density)
            err = np.abs(A._binopt2(B, '_plus_') - A._binopt(B, '_plus_')).sum()
#            print(f"Error check: {err=}")
            if err > 1e-7:
                assert False

            print(f"time_csctools: {M=} =2**{i} {N=} = 2**{k} (nonzero={A.nnz})", end=' ')
            t_c = time_csctools(A, B, rounds, number).min()
            print(t_c) #, end=' ')

            print(f"time_csrtools: {M=} =2**{i} {N=} = 2**{k} (nonzero={A.nnz})", end=' ')
            t_r = time_csrtools(A, B, rounds, number).min()
            print(t_r) #, end=' ')

            results.append((M, N, nnz, t_r, t_c))
            #print()
            del A, B

all_time = time.time() - all_start
print(f"done timing after {all_time}s\n")

print("---------- results ----------")
print(f"shape: (M, N)")
print(f"----- {rounds=} {number=} -----")
print(f"        M    N        nnz   t_csrtools t_csctools ratio(csctools/csrtools)")
for M, N, nnz, t_r, t_c in results:
    print(f"{M:10d},{N:3d},{nnz:10d},{t_r:.6f},{t_c:.6f},{t_c / t_r:.2f},11")

