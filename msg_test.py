import hybrid

msg_env = hybrid.MessageSender()

words = ["", "i", "r", "e", "s"]

def message_sending_task(num_msgs):
    task = hybrid.DFA(
        list(range(1 + num_msgs + 3)), 
        0, 
        [num_msgs + 1], 
        [num_msgs + 3], 
        [num_msgs + 2]
    )
    task.add_transition(0, "i", 1)
    for w in set(words).difference(set(["i"])):
        task.add_transition(0, w, 0)
    for r in range(num_msgs):
        task.add_transition(1 + r, "s", 1 + r + 1)
        task.add_transition(1 + r, "e", num_msgs + 3)
        for w in set(words).difference(set(["s", "e"])):
            task.add_transition(1 + r, w, 1 + r)
    for w in words:
        task.add_transition(num_msgs + 1, w, num_msgs + 2)
        task.add_transition(num_msgs + 2, w, num_msgs + 2)
        task.add_transition(num_msgs + 3, w, num_msgs + 3)
    return task

mission = hybrid.Mission()

dfa = message_sending_task(1)
mission.add_task(dfa)
scpm = hybrid.SCPM(mission, 1, list(range(2)))

P, R = hybrid.test_build(scpm, msg_env)

# convert the transition matrix and the rewards matrix into scipy matrices.
from scipy.sparse import csr_matrix
print("CSR TRANSITION MATRIX TO ARRAY")
sciP = csr_matrix((P.x, P.p, P.i), shape=(P.m, P.n)).toarray()
for r in range(P.m):
    for c in range(P.n):
        print(f" {sciP[r, c]:.2} ", end=" ")
    print()

sciR = csr_matrix((R.x, R.p, R.i), shape=(R.m, R.n)).toarray()
print("CSR REWARDS MATRIX TO ARRAY")
for r in range(R.m):
    for c in range(R.n):
        print(f" {sciR[r, c]:.2} ", end=" ")
    print()

#print("THREAD TEST")
#mission = hybrid.Mission()
#
#dfa = message_sending_task(100)
#mission.add_task(dfa)
#scpm = hybrid.SCPM(mission, 1, list(range(2)))
#hybrid.thread_test(scpm, msg_env)

print("MKL TEST BLAS")
test_output = hybrid.mkl_test()
print(f"MKL BLAS TEST: {test_output == 10.}")

print("MKL SPARSE MATRIX CREATE TEST")
#hybrid.csr_impl_test()

import numpy as np

A = np.array([
    [1, -1, 0, -3, 0],
    [-2, 5, 0, 0, 0],
    [0, 0, 4, 6, 4],
    [-4, 0, 2, 7, 0],
    [0, 8, 0, 0, -5]
])

x = [3, 2, 5, 4, 1]

print("A.x \n", A @ x)

A = np.array([
    [10, 20, 0, 0, 0, 0],
    [0, 30, 0, 40, 0, 0],
    [0, 0, 50, 60, 70, 0],
    [0, 0, 0, 0, 0, 80]
])

x = [1, 2, 3, 4, 5, 6]

print("A.x \n", A @ x)

epsilon = 0.0001
w = [1.0, 0.0]
initP, initR, init_x, init_pi = hybrid.test_initial_policy(scpm, msg_env, w, epsilon)

print("CSR TRANSITION MATRIX TO ARRAY")
sciP = csr_matrix((initP.x, initP.p, initP.i), shape=(initP.m, initP.n)).toarray()
for r in range(initP.m):
    for c in range(initP.n):
        print(f" {sciP[r, c]:.2} ", end=" ")
    print()

sciR = csr_matrix((initR.x, initR.p, initR.i), shape=(initR.m, initR.n)).toarray()
print("CSR REWARDS MATRIX TO ARRAY")
for r in range(initR.m):
    for c in range(initR.n):
        print(f" {sciR[r, c]:.2} ", end=" ")
    print()


sciP = csr_matrix((P.x, P.p, P.i), shape=(P.m, P.n)).toarray()

w = [0.5, 0.5]

hybrid.test_policy_optimisation(
    scpm, msg_env, w, epsilon, init_x, init_pi
)

