import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

def gradient_descent(f, grad_f, x0, A, b, dom_f, MAXITERS=100, TOL=1e-8,alpha=0.01, beta=0.8, print_iter=False):
    MAXITERS = MAXITERS
    TOL = TOL
    alpha = alpha
    beta = beta
    f, grad_f, dom_f = f, grad_f, dom_f
    A, b = A, b
    m, n = A.shape
    x = x0
    decrement = lambda  dx: np.sum(dx**2)
    decrement_value_list = []
    obj_list = [f(x0)]
    x_list = [x0]
    for iters in range(MAXITERS):
        dx = -grad_f(x)
        decrement_value = decrement(dx)
        decrement_value_list.append(decrement_value)
        if decrement_value < TOL:
            if print_iter:
                print("Iteration: %d, decrement: %.10f" % (iters, decrement_value))
            break
        t = 1
        while not dom_f(x + t*dx):
            t *= beta

        while f(x + t*dx) > f(x) - alpha * t * decrement_value:
            t *= beta
        # print(dx)
        x += t*dx
        x_list.append(x)
        obj_list.append(f(x))
        # print("Iteration: %d, decrement: %.10f" % (iters, decrement_value))
    return x_list, obj_list


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
        decrement = lambda dx, x: (dx.dot(nabla_f(x).dot(dx)))/2
    else:
        decrement = decrement_func
    decrement_value_list = []
    obj_list = [f(x)]
    x_list = [x]
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

        decrement_value = decrement(dx, x)
        decrement_value_list.append(decrement_value)
        if decrement_value < TOL:
            if print_iter:
                print("Iteration: %d, decrement: %.10f" % (iters, decrement_value))
            break
        
        t = 1
        while not dom_f(x + t*dx):
            t *= beta
            
        # Backtracking line search
        while f(x + t*dx) > f(x) - alpha * t * decrement_value:
            t *= beta

        x += t*dx
        x_list.append(x.copy())
        obj_list.append(f(x))
        if print_iter:
            print("Iteration: %d, decrement: %.10f" % (iters, decrement_value))
    return np.array(x_list), obj_list
    

# Solve the equality constrained problem using Newton's method
# def newton_eq(f, grad_f, nabla_f, x0, A, b, dom_f, MAXITERS=100, TOL=1e-8,alpha = 0.01, beta = 0.8, print_iter=False):
    
#     MAXITERS = MAXITERS
#     TOL = TOL
#     alpha = alpha
#     beta = beta
#     f, grad_f, nabla_f, dom_f = f, grad_f, nabla_f, dom_f
#     A, b = A, b
#     m, n = A.shape
#     x = x0
#     # Newton decrement
#     stopping_criteria = lambda dx, x: (dx.dot(nabla_f(x).dot(dx)))/2
    
#     for iters in range(MAXITERS):
#         dx = np.linalg.solve(
#             np.block([[nabla_f(x), A.T], [A, np.zeros((m, m))]]),
#             np.block([-grad_f(x), np.zeros(m)]))
#         dx = dx[:n]
#         decrement_value = stopping_criteria(dx, x)
#         if print_iter:
#             print("Iteration: %d, decrement: %.8f" % (iters, decrement_value))
#         if decrement_value < TOL:
#             break
        
#         t = 1
#         while not dom_f(x + t*dx):
#             t *= beta
            
#         # Backtracking line search
#         while f(x + t*dx) > f(x) - alpha * t * decrement_value:
#             t *= beta
        
#         x += t*dx
        
#     return x



def quasi_newton(f, grad_f, x0, A, b, dom_f, MAXITERS=100, TOL=1e-8,alpha = 0.01, beta = 0.8, print_iter=False, N=1, diag_only=False, decrement_func=None):
    MAXITERS = MAXITERS
    TOL = TOL
    alpha = alpha
    beta = beta
    f, grad_f, dom_f = f, grad_f, dom_f
    
    x = x0.copy()
    n = len(x)
    
    if not decrement_func:
        decrement = lambda dx, B: (dx.dot(B.dot(dx)))/2
    else:
        decrement = decrement_func
        
    decrement_value_list = []
    obj_list = [f(x)]
    x_list = [x]
    
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
        while f(x + t*dx) > f(x) - alpha * t * decrement_value:
            t *= beta

        x += t*dx
        grad_new = grad_f(x)
        # x_{t+1} - x_t = t*dx
        s = t * dx
        y = grad_new - grad
        
        if iters % N == 0:
            denom = np.dot(y, s)
            B_new = B - np.outer(B.dot(s), B.dot(s))/np.dot(s, B.dot(s)) + np.outer(y, y)/denom
            # Woodbury identity
            # B_inv_new = (np.eye(n)-np.outer(s,y)/denom).dot(B_inv).dot(np.eye(n)-np.outer(y,s)/denom) + np.outer(s,s)/denom
            # B_inv = B_inv_new
            B_inv = (np.eye(n)-np.outer(s,y)/denom).dot(B_inv).dot(np.eye(n)-np.outer(y,s)/denom) + np.outer(s,s)/denom 
            B = B_new
            
        grad = grad_new

        x_list.append(x.copy())
        obj_list.append(f(x))
        
    return np.array(x_list), obj_list

# write a quasi newton BFGS with limited memory


# def quasi_newton_LBFGS(f, grad_f, x0, A, b, dom_f, MAXITERS=100, TOL=1e-8,alpha = 0.01, beta = 0.8, print_iter=False, M=1, diag_only=False, decrement_func=None):
#     MAXITERS = MAXITERS
#     TOL = TOL
#     alpha = alpha
#     beta = beta
#     f, grad_f, dom_f = f, grad_f, dom_f
    
#     x = x0.copy()
#     n = len(x)
    
#     if not decrement_func:
#         decrement = lambda dx, B: (dx.dot(B.dot(dx)))/2
#     else:
#         decrement = decrement_func
        
#     decrement_value_list = []
#     obj_list = [f(x)]
#     x_list = [x]
    
#     B = np.eye(n)
#     B_inv = np.eye(n)
#     grad = grad_f(x)
    
#     s_queue = []
#     y_queue = []
    
#     dx = -B_inv.dot(grad)
    
#     for iters in range(1, MAXITERS):
        
#         decrement_value = decrement(dx, B)
#         print(decrement_value)
#         decrement_value_list.append(decrement_value)
#         if decrement_value < TOL:
#             if print_iter:
#                 print("Iteration: %d, decrement: %.10f" % (iters, decrement_value))
#             break
        
#         t = 1
#         # Check if t*dx is in still in the domain.
#         while not dom_f(x + t*dx):
#             t *= beta
#         # Backtracking line search.
#         while f(x + t*dx) > f(x) - alpha * t * decrement_value:
#             t *= beta
            
#         # x_{t+1} - x_t = t*dx
#         s = t * dx
#         x_new = x + s
#         grad_new = grad_f(x)
#         y = grad_new - grad
        
#         if len(s_queue) == M:
#             s_queue.pop(0)
#             y_queue.pop(0)
#         s_queue.append(s)
#         y_queue.append(y)

        
#         if iters >= M:
#             alpha_queue = []
#             for s, y in zip(s_queue[::-1], y_queue[::-1]):
#                 alpha = np.dot(s, y)/np.dot(y, s)
#                 grad_new = grad_new - alpha * y
#                 alpha_queue.insert(0, alpha)
#             p = B_inv.dot(grad_new)
#             for s, y, alpha in zip(s_queue, y_queue, alpha_queue):
#                 beta = np.dot(y, p)/np.dot(y, s)
#                 p = p + (alpha-beta)*s
#             dx_new = -p
#         else:
#             dx_new = dx    
        
#         grad = grad_new
#         x = x_new
#         dx = dx_new
        
#         x_list.append(x.copy())
#         obj_list.append(f(x))
        
#         if print_iter:
#             print("Iteration: %d, decrement: %.10f" % (iters, decrement_value))
#     return np.array(x_list), obj_list


def plot_error_iter(x, fx, cvx_solution, label, color='blue'):
    plt.semilogy(np.arange(len(x)), fx-cvx_solution, color=color, label=label)
    
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