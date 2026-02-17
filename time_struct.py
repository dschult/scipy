import subprocess
import time

import numpy as np
import scipy as sp


br_name = subprocess.run('git rev-parse --abbrev-ref HEAD', capture_output=True, shell=True, text=True).stdout[:-1]
print("Branch " + br_name)

# Not really the "main" branch. Just the branch we copy/paste to compare others to.
main_times = [
    0.00282547919778,
    0.00286793750711,
    0.00276560835773,
    0.00287323750090,
    0.00288349374896,
    0.00278888960602,
]

# store results in dict so can change order of loops
min_result = {
    'csr_add': 1e9,
    'csr_sub': 1e9,
    'csr_mul': 1e9,
    'csc_add': 1e9,
    'csc_sub': 1e9,
    'csc_mul': 1e9
}

# choose these to make the ratios near 1 when running in "main" branch
rounds=20
numb=20
N = 1000

A = sp.sparse.random_array((N, N), density=0.3, rng=97)
B = A.T

for r in range(rounds):
    # put loop over `r` outside to spread out the rounds over time.
    for format in ['csr', 'csc']:
        AA = A.asformat(format).copy()
        BB = B.asformat(format).copy()

        for op in ['__add__', '__sub__', '__mul__']:
            name = format + '_' + op[2:5]

            # print(" ".join([format, op, str(r), time.ctime()]))
            # timeit
            a, b = AA.copy(), BB.copy()
            fn = getattr(a, op)
            # warmup
            fn(b)

            start = time.perf_counter()
            for _ in range(numb):
                fn(b)
            stop = time.perf_counter()
            min_result[name] = min((stop - start) / numb, min_result[name])

# print results that can be copy/pasted in above as the new "main" baseline
# Needed in any new environment
print()
for format in ['csr', 'csc']:
    for op in ['__add__', '__sub__', '__mul__']:
        name = format + '_' + op[2:5]
        print(f"{min_result[name]:.14f},")

# summary output
print()
print("Summary:")
print(" format  op     main    struct  s/m ")
msg = "  {f}   {o}  {m:8.6f} {s:8.6f} {r:4.2f} "
for (name, s), m in zip(min_result.items(), main_times):
    f = name[:3]
    o = name[-3:]
    r = s/m
    print(msg.format(f=f, o=o, m=m, s=s, r=r))
