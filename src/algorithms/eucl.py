import cvxpy as cp
import numpy as np

def eucl_new_target(X, W, t):
    #print("X", X)
    #print("W", W)
    #print("l", l)
    z = cp.Variable(len(t)) # vector variable with shape (5,)
    #obj = cp.sum_squares(A @ z - t)
    obj = cp.norm(z - t, 2)
    constraints = [ ]
    
    for k in range(len(X)):
        constraints.append(np.dot(W[k], X[k]) >= W[k] @ z)

    #print(constraints)
    
    prob = cp.Problem(cp.Minimize(obj), constraints)
    #print(prob)
    prob.solve()
    print("status", prob.status)
    #print("optimal value", prob.value)
    #print("optimal var", z.value)
    return z.value