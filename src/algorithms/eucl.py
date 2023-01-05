import cvxpy as cp
import numpy as np

def eucl_new_target(X, W, t, n, C):
    # C is the maximum percentage deviation away from the target vector
    #print("X", X)
    #print("W", W)
    #print("l", l)
    z = cp.Variable(len(t)) # vector variable with shape (5,)
    #obj = cp.sum_squares(A @ z - t)
    obj = cp.norm(t - z, 2)
    constraints = []
    for k in range(len(t)):
        if k < n:
            constraints.append(z[k] >= (1 + C[k]) * t[k])
            constraints.append(z[k] <= t[k])
        else:
            constraints.append(z[k] >= (1 - C[k]) * t[k]) 
            constraints.append(z[k] <= t[k]) 
    
    for k in range(len(X)):
        constraints.append(np.dot(W[k], X[k]) >= W[k] @ z)
        #a = np.dot(W[k], X[k])
        #b = np.dot(W[k], t)
        #print(f"C{k}", a, b, a < b)

    #print(constraints)
    
    prob = cp.Problem(cp.Minimize(obj), constraints)
    #print(prob)
    prob.solve()
    #print("status", prob.status)
    #print("optimal value", prob.value)
    #print("optimal var", z.value)
    #for k in range(len(X)):
    #    #constraints.append(np.dot(W[k], X[k]) >= W[k] @ z.value)
    #    a = np.dot(W[k], X[k])
    #    b = np.dot(W[k], z.value)
    #    print(f"Sol C{k}", a, b, a < b)
    return z.value