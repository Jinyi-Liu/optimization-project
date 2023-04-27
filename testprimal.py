import numpy as np

# m: number of constraints
# n: number of variables
m = 50
n = 100

A = np.random.randn(m, n)
x0 = np.random.random(n)
b = A.dot(x0)
z = np.random.randn(m)
s = np.random.randn(n)

c = A.T.dot(z) + s

def get_res(x, lambda_, v, tinv):
    r_dual = Df.dot(lambda_) + A.T.dot(v)+c
    r_cent = -np.diag(lambda_)@f(x) - tinv
    r_pri = A.dot(x)-b
    return r_dual, r_cent, r_pri

def get_res_norm(x, lambda_, v, tinv):
    r_dual, r_cent, r_pri = get_res(x, lambda_, v, tinv)
    return np.linalg.norm(r_dual), np.linalg.norm(r_cent), np.linalg.norm(r_pri)


MAXITERS = 500
TOL = 1e-8
RESTOL = 1e-8
mu = 10
alpha = 0.01
beta = 0.5
gaps = []
resdls = []
x = x0
f = lambda x: -x
Df = -np.eye(n)
lambda_ = -1 / (f(x))
v = np.zeros(m)

for iters in range(MAXITERS):
    # Surrogate duality gap
    ita = - f(x).dot(lambda_)
    gaps.append(ita)
    # residual
    # r_dual = Df.dot(lambda_) + A.T.dot(v)+c
    # r_cent = -np.diag(lambda_)@f(x) - tinv
    # r_pri = A.dot(x)-b
    r_dual, r_cent, r_pri = get_res(x, lambda_, v, tinv)
    resdls.append(np.linalg.norm(np.hstack([r_dual, r_pri])))
    # stopping criterion
    if ita < TOL and np.linalg.norm(np.hstack([r_dual, r_pri])) < RESTOL:
        break
    tinv = ita / (m * mu)
    sol = -np.linalg.solve(
        np.block([[np.zeros((n, n)), Df.T, A.T],
                  [-np.diag(lambda_) @ Df, -np.diag(f(x)), np.zeros((n, m))],
                  [A, np.zeros((m, m)), np.zeros((m, n))]]),
        np.hstack([r_dual, r_cent, r_pri])
    )
    dx = sol[:n]
    dlambda_ = sol[n:n + n]
    dv = sol[-m:]

    # backtracking line search
    # The maximum step such that the new lambda_ is positive, i.e., feasible.
    step = min(1, 0.99 / np.max(-dlambda_ / lambda_))
    while True:
        x_new = x + step * dx
        if np.all(f(x_new) < 0):
            break
        step *= beta
        
    new_x = x + step * dx
    new_lambda_ = lambda_ + step * dlambda_
    new_v = v + step * dv

    # Old residual norm
    old_norm = sum(get_res_norm(x, lambda_, v, tinv))

    while sum(get_res_norm(new_x, new_lambda_, new_v, tinv)) > (1 - alpha * step) * old_norm:
        step *= beta
        new_x = x + step * dx
        new_lambda_ = lambda_ + step * dlambda_
        new_v = v + step * dv
        print('step = ', step)

    x = new_x
    lambda_ = new_lambda_
    v = new_v

