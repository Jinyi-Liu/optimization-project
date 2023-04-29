import numpy as np


def quasi_newton_LBFGS(f, grad_f, x0, A, b, dom_f, MAXITERS=100, TOL=1e-8, alpha=0.01, beta=0.8, print_iter=False, M=1,
                       diag_only=False, decrement_func=None):
    MAXITERS = MAXITERS
    TOL = TOL
    alpha = alpha
    beta = beta
    f, grad_f, dom_f = f, grad_f, dom_f

    x = x0.copy()
    n = len(x)

    if not decrement_func:
        decrement = lambda dx: np.linalg.norm(dx)
    else:
        decrement = decrement_func

    decrement_value_list = []
    obj_list = [f(x)]
    x_list = [x]

    B = np.eye(n)
    B_inv = np.eye(n)
    grad = grad_f(x)

    s_queue = []
    y_queue = []

    dx = -B_inv.dot(grad)
    t = 1
    for iters in range(1, MAXITERS):

        decrement_value = decrement(t*dx)
        print(decrement_value)
        decrement_value_list.append(decrement_value)
        if decrement_value < TOL:
            if print_iter:
                print("Iteration: %d, decrement: %.10f" % (iters, decrement_value))
            break

        t = 1
        # Check if t*dx is in still in the domain.
        while not dom_f(x + t * dx):
            t *= beta
        # Backtracking line search.
        while f(x + t * dx) > f(x) - alpha * t * decrement_value:
            t *= beta

        # x_{t+1} - x_t = t*dx
        s = t * dx
        x_new = x + s
        grad_new = grad_f(x_new)
        y = grad_new - grad

        if len(s_queue) == M:
            s_queue.pop(0)
            y_queue.pop(0)
        s_queue.append(s)
        y_queue.append(y)

        if iters >= M:
            alpha_queue = []
            q = - grad_new
            for s, y in zip(s_queue[::-1], y_queue[::-1]):
                alpha = np.dot(s, q) / np.dot(y, s)
                q = q - alpha * y
                alpha_queue.insert(0, alpha)
            p = q
            for s, y, alpha in zip(s_queue, y_queue, alpha_queue):
                beta_ = np.dot(y, p) / np.dot(y, s)
                p = p + (alpha - beta_) * s
            dx_new = p
        else:
            dx_new = -B_inv.dot(grad_new)

        grad = grad_new
        x = x_new
        dx = dx_new

        x_list.append(x.copy())
        obj_list.append(f(x))

        if print_iter:
            print("Iteration: %d, decrement: %.10f" % (iters, decrement_value))
    return np.array(x_list), obj_list


# m: number of constraints
# n: number of variables
m = 50
n = 200
A = np.random.randn(m, n)
b = np.ones(m)

f = lambda x: -np.sum(np.log(1 - A @ x)) - np.sum(np.log(1 - x ** 2))
grad_f = lambda x: A.T @ (1 / (1 - A @ x)) + 2 * x / (1 - x ** 2)
nabla_f = lambda x: A.T @ np.diag(1 / (1 - A @ x) ** 2) @ A + np.diag(2 / (1 - x ** 2) ** 2)


def dom_f(x, A=A, b=b):
    cons_1 = A @ x - b < 0
    cons_2 = np.abs(x) - 1 < 0
    return np.all(cons_1) and np.all(cons_2)


x_list_quasi, obj_list_quasi = quasi_newton_LBFGS(f, grad_f, np.zeros(n), A, b,
                                                  dom_f, MAXITERS=1000, TOL=1e-8, alpha=0.01, print_iter=False,
                                                  M=5, diag_only=False, decrement_func=None)
print(obj_list_quasi)