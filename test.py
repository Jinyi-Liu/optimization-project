import numpy as np
m = 100
n = 50
A = np.random.randn(m, n)
b = np.random.rand(m, )
z0 = np.random.rand(m, )
c = -A.T@z0
# linprog
from scipy.optimize import linprog
res = linprog(c, A_ub=A, b_ub=b, method='highs')
x = res.x
s = (1-c.T@x)/np.linalg.norm(c)**2
b = b + s * A @ c
x0 = s * c

MAXITERS = 500
TOL = 1e-8
RESTOL = 1e-8
MU = 10
ALPHA = 0.01
BETA = 0.5

[m, n] = A.shape
gaps = []
resdls = []
x = x0
s = b - A @ x
z = 1 / s

for iters in range(MAXITERS):
    gap = s.T @ z
    gaps.append(gap)
    res = A.T @ z + c
    resdls.append(np.linalg.norm(res))

    if gap < TOL and np.linalg.norm(res) < RESTOL:
        break

    tinv = gap / (m * MU)
    sol = -np.linalg.solve(
        np.block([[np.zeros((n, n)), A.T], [A, np.diag(-s / z)]]),
        np.concatenate((A.T @ z + c.reshape(-1), -s + tinv * (1 / z)))
    )

    dx = sol[:n]
    dz = sol[n:(n + m)]
    ds = -A @ dx

    # backtracking line search
    r = np.concatenate((c + A.T @ z, z * s - tinv))
    step = min(1.0, 0.99 / np.max(-dz / z))

    while np.min(s + step * ds) <= 0:
        step = BETA * step

    newz = z + step * dz
    newx = x + step * dx
    news = s + step * ds
    newr = np.concatenate((c + A.T @ newz, newz * news - tinv))

    while np.linalg.norm(newr) > (1 - ALPHA * step) * np.linalg.norm(r):
        step = BETA * step
        newz = z + step * dz
        newx = x + step * dx
        news = s + step * ds
        newr = np.concatenate((c + A.T @ newz, newz * news - tinv))

    x = newx
    z = newz
    s = b - A @ x
