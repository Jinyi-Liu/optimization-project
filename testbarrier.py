import numpy as np
# m: number of constraints
# n: number of variables
m = 50
n = 100

A = np.random.randn(m, n)
x0 = np.random.random(n)
b = A.dot(x0)
z = np.random.randn(m)
s = np.random.random(n)

# This is to make sure that the dual problem is feasible in interior since A.T @ z < c.
c = A.T.dot(z) + s


MAXITERS = 500
TOL = 1e-8
RESTOL = 1e-8
mu = 10
alpha = 0.01
beta = 0.5


from newton_eq import newton_eq

#%%
t = 10
x_inner = x0
while m/t > TOL:
    f = lambda x: t*c.dot(x) - np.sum(np.log(x))
    grad_f = lambda x: t*c - 1/x
    nabla_f = lambda x: np.diag(1/x**2)
    x = newton_eq(f, grad_f, nabla_f, x_inner, A, b, MAXITERS=50, TOL=1e-5, alpha=0.01, beta=0.8)
    x_inner = x
    t *= mu