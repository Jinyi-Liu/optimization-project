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

def newton(f, grad_f, nabla_f, x0, A, b, MAXITERS=100, TOL=1e-8,alpha = 0.01, beta = 0.8):
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
        dx = np.linalg.solve(nabla_f(x), -grad_f(x))
        dx = dx[:n]
        decrement_value = decrement(dx, x)/2
        if decrement_value < TOL:
            break
        t = 1
        while np.any(A@(x + t*dx) - b>0):
            t *= beta

        while f(x + t*dx) > f(x) - alpha * t * decrement_value:
            t *= beta
        # print(dx)
        x += t*dx
        # print("Iteration: %d, decrement: %f" % (iters, decrement_value))
    return x
