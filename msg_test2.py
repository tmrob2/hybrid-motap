import hybrid
import numpy as np

msg_env = hybrid.MessageSender()

words = ["", "i", "r", "e", "s", "ini"]

def negvg_message_sending_task(num_msgs):
    task = hybrid.DFA(
        list(range(1 + num_msgs + 3)), 
        0, 
        [num_msgs + 1], # acc
        [num_msgs + 3], # rej
        [num_msgs + 2]  # done
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



NUM_TASKS = 2
NUM_AGENTS = 2
LB = 17.#-630 / NUM_AGENTS
UB = 11.#-690 / NUM_AGENTS
mission = hybrid.Mission()

p_ = [0.01, 0.1]

for msg in range(NUM_TASKS):
    dfa = negvg_message_sending_task(msg + 1)
    mission.add_task(dfa)
scpm = hybrid.SCPM(mission, NUM_AGENTS, list(range(2)))

#p_ = np.random.normal(0.01, 0.01/100, NUM_AGENTS)
env1 = hybrid.MAS([0, 1])
model1_ = [hybrid.Model("MS1", list(range(5)), 0, 0.1) for k in range(NUM_AGENTS)] 
for _ in range(NUM_AGENTS):
    env1.add_environment(model1_.pop())
env1.create_order([0, 1])

#hybrid.test_build_mamdp(scpm, msg_env)
eps1 = 1e-6
eps2 = 0.01
#w = [0.01 / NUM_AGENTS] * NUM_AGENTS + [0.99 / NUM_TASKS] * NUM_TASKS
w = [0.] * NUM_AGENTS + [1.0 / NUM_TASKS] * NUM_TASKS
#hybrid.motap_msg_test_cpu(scpm, msg_env, w, eps1, 1, 1000, 30)
#hybrid.motap_mamdp_test_cpu(scpm, env1, w, eps1, 1, 5000, 2000, True)
#constraint_threshold = [1.] * NUM_AGENTS + [0.6] * NUM_TASKS
#target = np.random.uniform(low=-LB,high=-UB,size=NUM_AGENTS).tolist() + [0.9] * NUM_TASKS
#hybrid.motap_mamdp_synthesis_test(scpm, msg_env, w, target, eps1, eps2, 
#                                  2, 4000, 2000, constraint_threshold)
#hybrid.motap_mamdp_synth(scpm, env1, w, target, eps1, eps2, 
#                         2, 4000, 2000, constraint_threshold)
agent_init_states = env1.get_init_states()
hybrid.motap_scpm_test_cpu(scpm, env1, w, eps1, 1, 1000, 150, agent_init_states)
#hybrid.test_build_scpm(scpm, env1)
#hybrid.test_build_stapu(scpm, env1)
hybrid.motap_stapu_test_cpu(scpm, env1, w, eps1, 1, 1000, 150, agent_init_states)

