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

#mission = hybrid.Mission()
#
#dfa = message_sending_task(1)
#mission.add_task(dfa)
#scpm = hybrid.SCPM(mission, 1, list(range(2)))
#
#hybrid.test_build(scpm, msg_env)
#
#import numpy as np
#
#A = np.array([
#    [1, -1, 0, -3, 0],
#    [-2, 5, 0, 0, 0],
#    [0, 0, 4, 6, 4],
#    [-4, 0, 2, 7, 0],
#    [0, 8, 0, 0, -5]
#])
#
#x = [3, 2, 5, 4, 1]
#
#print("A.x \n", A @ x)
#
#A = np.array([
#    [10, 20, 0, 0, 0, 0],
#    [0, 30, 0, 40, 0, 0],
#    [0, 0, 50, 60, 70, 0],
#    [0, 0, 0, 0, 0, 80]
#])
#
#x = [1, 2, 3, 4, 5, 6]
#
#print("A.x \n", A @ x)
#
epsilon = 0.0001
#w = [1.0, 0.]
#init_x, init_pi = hybrid.test_initial_policy(scpm, msg_env, w, epsilon)
#
NUM_TASKS = 2
NUM_AGENTS = 2
GPU_BUFFER_SIZE = 50
CPU_COUNT = 1
#print("x:", init_x)
#print("pi:", init_pi)

w = [0.] * NUM_AGENTS + [1./NUM_TASKS] * NUM_TASKS
mission = hybrid.Mission()
for msg in range(NUM_TASKS):
    dfa = message_sending_task(msg + 1)
    mission.add_task(dfa)
scpm = hybrid.SCPM(mission, NUM_AGENTS, list(range(2)))
hybrid.test_threaded_initial_policy(scpm, msg_env, w, epsilon)

hybrid.experiment_gpu_cpu_binary_thread(
    scpm, msg_env, w, epsilon, GPU_BUFFER_SIZE, CPU_COUNT
)
