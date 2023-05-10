import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

def gradient_descent(f, grad_f, x0, A, b, dom_f, MAXITERS=100, TOL=1e-8, alpha=0.01, beta=0.8, print_iter=False):
    MAXITERS = MAXITERS
    TOL = TOL
    alpha = alpha
    beta = beta
    f, grad_f, dom_f = f, grad_f, dom_f
    A, b = A, b
    m, n = A.shape
    x = x0
    decrement = lambda  dx: np.sum(dx**2)
    obj_list = [f(x0)]
    decrement_value_check = 0
    alpha_check = 0
    for iters in range(MAXITERS):
        dx = -grad_f(x)
        decrement_value = decrement(dx)

        if print_iter:
            print("Iteration: %d, decrement: %.10f" % (iters, decrement_value),alpha_check)
        if decrement_value**2< TOL:
            break
        
        t = 1
        while not dom_f(x + t*dx):
            t *= beta
        while f(x + t*dx) > f(x) - alpha * t * decrement_value:
            t *= beta

        x += t*dx
        obj_list.append(f(x))

    return obj_list


# Newton's method for inequality constrained problem
def newton(f, grad_f, nabla_f, x0, A, b, dom_f, MAXITERS=100, TOL=1e-8,alpha = 0.01, beta = 0.8, print_iter=False, N=1, diag_only=False, decrement_func=None, eq=False):
    MAXITERS = MAXITERS
    TOL = TOL
    alpha = alpha
    beta = beta
    f, grad_f, nabla_f, dom_f = f, grad_f, nabla_f, dom_f
    A, b = A, b
    m, n = A.shape
    x = x0.copy()
    
    if not decrement_func:
        # Newton decrement
        decrement = lambda dx, x: (dx.dot(nabla_f(x).dot(dx)))
    else:
        decrement = decrement_func

    obj_list = [f(x)]
    nabla = nabla_f(x)
    
    for iters in range(1, MAXITERS):
        if iters % N == 0:
            nabla = nabla_f(x)
            if diag_only:
                nabla = np.diag(np.diag(nabla))
        if eq:
            dx = np.linalg.solve(
                np.block([[nabla, A.T], [A, np.zeros((m, m))]]),
                np.block([-grad_f(x), np.zeros(m)]))[0:n]
        else:
            dx = np.linalg.solve(nabla, -grad_f(x))
        if print_iter:
            print("Iteration: %d, decrement: %.10f" % (iters, decrement_value))

        decrement_value = decrement(dx, x)
        if decrement_value < TOL:
    
            break
        t = 1
        while not dom_f(x + t*dx):
            t *= beta
            
        # Backtracking line search
        while f(x + t*dx) > f(x) - alpha * t * decrement_value:
            t *= beta

        x += t*dx
        obj_list.append(f(x))

    return obj_list
    



def quasi_newton(f, grad_f, x0, A, b, dom_f, MAXITERS=100, TOL=1e-8,alpha = 0.01, beta = 0.8, print_iter=False, N=1, diag_only=False, decrement_func=None):
    MAXITERS = MAXITERS
    TOL = TOL
    alpha = alpha
    beta = beta
    f, grad_f, dom_f = f, grad_f, dom_f
    
    x = x0.copy()
    n = len(x)
    
    if not decrement_func:
        decrement = lambda dx, B: (dx.dot(B.dot(dx)))
    else:
        decrement = decrement_func
        
    decrement_value_list = []
    obj_list = [f(x)]
    
    B = np.eye(n)
    B_inv = np.eye(n)
    grad = grad_f(x)
    B_new = B.copy()
    for iters in range(0, MAXITERS):
        dx = -B_inv.dot(grad)

        decrement_value = decrement(dx, B)
        decrement_value_list.append(decrement_value)
        
        if print_iter:
            print("Iteration: %d, decrement: %.10f" % (iters, decrement_value))
        if decrement_value < TOL:
            break
        
        t = 1
        while not dom_f(x + t*dx):
            t *= beta
        # Backtracking line search.
        # while f(x + t*dx) > f(x) - alpha * t * decrement_value*2:
        #     t *= beta
        f_x = f(x)
        f_x_new = f(x + t * dx)
        grad_new = grad_f(x + t * dx)
        while f_x_new > f_x - alpha * t * decrement_value:
            t *= beta
            f_x_new = f(x + t * dx)
            grad_new = grad_f(x + t * dx)
            
        x += t*dx
        grad_new = grad_f(x)
        s = t * dx
        y = grad_new - grad
        
        if iters % N == 0:
            denom = np.dot(y, s)
            B_new = B - np.outer(B.dot(s), B.dot(s))/np.dot(s, B.dot(s)) + np.outer(y, y)/denom
            B_inv = (np.eye(n)-np.outer(s,y)/denom).dot(B_inv).dot(np.eye(n)-np.outer(y,s)/denom) + np.outer(s,s)/denom 
            B = B_new
            
        grad = grad_new

        # x_list.append(x.copy())
        obj_list.append(f(x))
        
    return obj_list



def plot_error_iter(fx, cvx_solution, label, color='blue'):
    plt.semilogy(np.arange(len(fx)), fx-cvx_solution, color=color, label=label)
    
def solve_by_cvx(A, b):
    # Use CVXPY to solve the problem with CVXOPT
    n = A.shape[1]
    x = cp.Variable(n)
    objective = cp.Minimize(-cp.sum(cp.log(1 - A @ x)) - cp.sum(cp.log(1 - x**2)))
    constraints = [A @ x <= b, cp.abs(x) <= 1]

    # Form and solve problem.
    prob = cp.Problem(objective, constraints)
    cvx_solution = prob.solve()
    return cvx_solution