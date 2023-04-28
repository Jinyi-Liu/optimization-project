import numpy as np

def newton_eq(f, grad_f, nabla_f, x0, A, b, MAXITERS=100, TOL=1e-8,alpha = 0.01, beta = 0.8):
    MAXITERS = MAXITERS
    TOL = TOL
    alpha = alpha
    beta = beta
    f, grad_f, nabla_f = f, grad_f, nabla_f
    A, b = A, b
    m, n = A.shape
    x = x0
    decrement = lambda dx, x: (dx.dot(nabla_f(x).dot(dx)))
    
    for iters in range(MAXITERS):
        dx = np.linalg.solve(
            np.block([[nabla_f(x), A.T], [A, np.zeros((m, m))]]),
            np.block([-grad_f(x), np.zeros(m)]))
        dx = dx[:n]
        decrement_value = decrement(dx, x)/2
        if decrement_value < TOL:
            break
        t = 1

        while f(x + t*dx) > f(x) - alpha * t * decrement_value:
            t *= beta
        
        x += t*dx
        print("Iteration: %d, decrement: %f" % (iters, decrement_value))
    return x

def newton(f, grad_f, nabla_f, x0, A, b, domf, MAXITERS=100, TOL=1e-8,alpha = 0.01, beta = 0.8, print_iter=False, N=1, diag_only=False, decrement_func=None):
    MAXITERS = MAXITERS
    TOL = TOL
    alpha = alpha
    beta = beta
    f, grad_f, nabla_f, domf = f, grad_f, nabla_f, domf
    A, b = A, b
    m, n = A.shape
    x = x0.copy()
    if not decrement_func:
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

        dx = np.linalg.solve(nabla, -grad_f(x))
        dx = dx[:n]
        decrement_value = decrement(dx, x)
        decrement_value_list.append(decrement_value)
        if decrement_value < TOL:
            print("Iteration: %d, decrement: %.10f" % (iters, decrement_value))
            break
        t = 1
        while not domf(x + t*dx):
            print("This t is not in domain: %f" % t)
            t *= beta

        while f(x + t*dx) > f(x) - alpha * t * decrement_value:
            t *= beta

        x += t*dx
        x_list.append(x.copy())
        obj_list.append(f(x))
        if print_iter:
            print("Iteration: %d, decrement: %.10f" % (iters, decrement_value))
    return np.array(x_list), obj_list
    

def gradient_descent(f, grad_f, x0, A, b, domf, MAXITERS=100, TOL=1e-8,alpha=0.01, beta=0.8, print_iter=False):
    MAXITERS = MAXITERS
    TOL = TOL
    alpha = alpha
    beta = beta
    f, grad_f, domf = f, grad_f, domf
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
        while not domf(x + t*dx):
            t *= beta

        while f(x + t*dx) > f(x) - alpha * t * decrement_value:
            t *= beta
        # print(dx)
        x += t*dx
        x_list.append(x)
        obj_list.append(f(x))
        # print("Iteration: %d, decrement: %.10f" % (iters, decrement_value))
    return x_list, obj_list