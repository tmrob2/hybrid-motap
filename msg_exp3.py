import hybrid
import numpy as np
import random
import argparse

random.seed(a=12345, version=2)
np.random.seed = 12345

# ------------------------------------------------------------------------------
# ARGS: Program args for repeating and controlling experiments
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Test MOTAP Experiment 3.")
parser.add_argument('--agents', dest='num_agents', default=2, help='the number of agents in MAS', 
                    required=True, type=int)
parser.add_argument('--tasks', dest='num_tasks', default=2, help='the number of tasks in MAS', 
                    required=True, type=int)
parser.add_argument('--type', dest='mdp_type', default=1, help='the number of agents in MAS', 
                    required=True, type=int)
parser.add_argument('--iter', dest='max_iter', default=1000, type=int,
                    help='The max number of iterations in VI before loop force terminates')
parser.add_argument('--unstable', dest='max_unstable', default=30, type=int, 
                    help='The number of divegent state-values before infinity is recognised')
parser.add_argument('--clb', dest="clower_bound", default=14., type=float, 
                    help="The lower bound on the cost selection distribution")
parser.add_argument('--cub', dest="cupper_bound", default=16., type=float, 
                    help="The upper bound on the cost selection distribution")
parser.add_argument('--model', dest='model', default='scpm', choices=['scpm', 'mamdp', 'both'], 
                    help='The type of model to run for the experiment')
parser.add_argument('-D', dest='model_debug', default=1, choices=[0, 1, 2, 3], help="debug verbosity", type=int)

args = parser.parse_args()


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

def vg_message_sending_task(num_msgs):
    task = hybrid.DFA(
        list(range(1 + num_msgs + 4)), 
        0, 
        [num_msgs + 1], 
        [num_msgs + 4], 
        [num_msgs + 3]
    )
    task.add_transition(0, "i", 1)
    for w in set(words).difference(set(["i"])):
        task.add_transition(0, w, 0)
    for r in range(num_msgs):
        task.add_transition(1 + r, "s", 1 + r + 1)
        task.add_transition(1 + r, "e", num_msgs + 4)
        for w in set(words).difference(set(["s", "e"])):
            task.add_transition(1 + r, w, 1 + r)
    for w in words:
        task.add_transition(num_msgs + 1, w, num_msgs + 2)
        task.add_transition(num_msgs + 4, w, num_msgs + 4)
    for w in set(words).difference(set(["ini"])):
        task.add_transition(num_msgs + 2, w, num_msgs + 2)
    task.add_transition(num_msgs + 2, "ini", num_msgs + 3)
    for w in words:
        task.add_transition(num_msgs + 3, w, num_msgs + 3)
    return task


NUM_TASKS = args.num_tasks
NUM_AGENTS = args.num_agents
LB = args.clower_bound
UB = args.cupper_bound
UNSTABLE = args.max_unstable
DEBUG = args.model_debug

# TODO better to create a list randomly assigning a probability
p_ = [0.01, 0.1]
#print("model probs: ", p_)
model1_ = [hybrid.Model("MS1", list(range(5)), 0, p_[k]) for k in range(NUM_AGENTS)] 
model2_ = [hybrid.Model("MS2", list(range(6)), 0, p_[k]) for k in range(NUM_AGENTS)] 
model3_ = [hybrid.Model("MS3", list(range(7)), 0, p_[k]) for k in range(NUM_AGENTS)] 

#hybrid.test_build_mamdp(scpm, msg_env)
eps1 = 1e-6
eps2 = 0.01
# No element of w can be exactly 0. (problems with infinity)
w1 = [0.001 / NUM_AGENTS] * NUM_AGENTS + [0.999 / NUM_TASKS] * NUM_TASKS
w = [0.] * NUM_AGENTS + [1.0 / NUM_TASKS] * NUM_TASKS

if args.mdp_type == 1:

    env1 = hybrid.MAS([0, 1])
    for _ in range(NUM_AGENTS):
        env1.add_environment(model1_.pop())
    env1.create_order([0, 1])

    mission = hybrid.Mission()

    for msg in range(NUM_TASKS):
        dfa = vg_message_sending_task(msg + 1)
        mission.add_task(dfa)
    scpm = hybrid.SCPM(mission, NUM_AGENTS, list(range(2)))

    # MDP Type 1
    #hybrid.motap_MS1_test_cpu(scpm, msg_env, w1, eps1, 1, 1000, 30)

    miss_mamdp = hybrid.Mission()

    for msg in range(NUM_TASKS):
        dfa = negvg_message_sending_task(msg + 1)
        miss_mamdp.add_task(dfa)
    scpm2 = hybrid.SCPM(miss_mamdp, NUM_AGENTS, list(range(2)))

    #hybrid.motap_MS1_mamdp_test_cpu(scpm2, msg_env, w, eps1, 1, 5000, 2000)
    constraint_threshold = [1.] * NUM_AGENTS + [0.5] * NUM_TASKS
    target = np.random.uniform(low=-LB,high=-UB,size=NUM_AGENTS).tolist() \
        + [0.7] * NUM_TASKS
    if args.model == "mamdp":
        hybrid.motap_mamdp_synth(scpm2, env1, w, target, eps1, eps2, 
                                    DEBUG, 4000, UNSTABLE, constraint_threshold)
    elif args.model == 'scpm':
        hybrid.motap_synth(scpm, env1, w1, target, eps1, eps2, 
                                DEBUG, 1000, 40, constraint_threshold)
    else: 
        hybrid.motap_mamdp_synth(scpm2, env1, w, target, eps1, eps2, 
                                    DEBUG, 4000, UNSTABLE, constraint_threshold)
        hybrid.motap_synth(scpm, env1, w1, target, eps1, eps2, 
                                DEBUG, 1000, 40, constraint_threshold)
# MDP Type 2
elif args.mdp_type == 2:

    env2 = hybrid.MAS([0, 1])
    for _ in range(NUM_AGENTS):
        env2.add_environment(model2_.pop())
    env2.create_order([0, 1])

    mission = hybrid.Mission()
    
    for msg in range(NUM_TASKS):
        dfa = vg_message_sending_task(msg + 1)
        mission.add_task(dfa)

    miss_mamdp = hybrid.Mission()

    for msg in range(NUM_TASKS):
        dfa = negvg_message_sending_task(msg + 1)
        miss_mamdp.add_task(dfa)
    scpm2_t2 = hybrid.SCPM(miss_mamdp, NUM_AGENTS, list(range(2)))

    scpm_t2 = hybrid.SCPM(mission, NUM_AGENTS, list(range(2)))
    #hybrid.motap_MS2_test_cpu(scpm_t2, msg2_env, w1, eps1, 1, 1000, 30)

    constraint_threshold = [1.] * NUM_AGENTS + [1.] * NUM_TASKS
    target = np.random.uniform(low=-LB,high=-UB,size=NUM_AGENTS).tolist() + \
        [0.7] * NUM_TASKS
    if args.model == 'mamdp':
        hybrid.motap_mamdp_synth(scpm2_t2, env2, w1, target, eps1, eps2, 
                                 2, 4000, 100, constraint_threshold)
    elif args.model == 'scpm':
        hybrid.motap_synth(scpm_t2, env2, w1, target, eps1, eps2, 
                           DEBUG, 1000, 40, constraint_threshold)
    else:
        hybrid.motap_mamdp_synth(scpm2_t2, env2, w1, target, eps1, eps2, 
                                 DEBUG, 4000, 100, constraint_threshold)
        hybrid.motap_synth(scpm_t2, env2, w1, target, eps1, eps2, 
                           DEBUG, 1000, 40, constraint_threshold)

# MDP Type 3
elif args.mdp_type == 3:

    env3 = hybrid.MAS([0, 1])
    for _ in range(NUM_AGENTS):
        env3.add_environment(model3_.pop())
    env3.create_order([0, 1])

    mission = hybrid.Mission()
    
    for msg in range(NUM_TASKS):
        dfa = vg_message_sending_task(msg + 1)
        mission.add_task(dfa)

    miss_mamdp = hybrid.Mission()

    for msg in range(NUM_TASKS):
        dfa = negvg_message_sending_task(msg + 1)
        miss_mamdp.add_task(dfa)
    scpm2_t3 = hybrid.SCPM(miss_mamdp, NUM_AGENTS, list(range(2)))

    scpm_t3 = hybrid.SCPM(mission, NUM_AGENTS, list(range(2)))
    #hybrid.motap_MS3_test_cpu(scpm_t3, msg3_env, w1, eps1, 1, 1000, 50)

    constraint_threshold = [1.] * NUM_AGENTS + [0.7] * NUM_TASKS
    target = np.random.uniform(low=-LB,high=-UB,size=NUM_AGENTS).tolist() + \
        [0.7] * NUM_TASKS
    if args.model == 'mamdp':
        hybrid.motap_mamdp_synth(scpm2_t3, env3, w1, target, eps1, eps2, 
                                2, 5000, UNSTABLE, constraint_threshold)
    elif args.model == 'scpm':
        hybrid.motap_synth(scpm_t3, env3, w1, target, eps1, eps2, 
                            2, 1000, 50, constraint_threshold)
    else:
        hybrid.motap_mamdp_synth(scpm2_t3, env3, w1, target, eps1, eps2, 
                                2, 5000, UNSTABLE, constraint_threshold)
        hybrid.motap_synth(scpm_t3, env3, w1, target, eps1, eps2, 
                            2, 1000, 50, constraint_threshold)

# MDP Type Mixture 12
elif args.mdp_type == 4:
    model_ = [hybrid.Model("MS1", list(range(5)), 0, p_[0])]
    model_.append(hybrid.Model("MS2", list(range(6)), 0, p_[1]))

    env = hybrid.MAS([0, 1])
    for _ in range(NUM_AGENTS):
        env.add_environment(model_.pop())
    env.create_order([0, 1])

    mission = hybrid.Mission()
    
    for msg in range(NUM_TASKS):
        dfa = vg_message_sending_task(msg + 1)
        mission.add_task(dfa)

    miss_mamdp = hybrid.Mission()

    for msg in range(NUM_TASKS):
        dfa = negvg_message_sending_task(msg + 1)
        miss_mamdp.add_task(dfa)
    scpm2_t3 = hybrid.SCPM(miss_mamdp, NUM_AGENTS, list(range(2)))

    scpm_t3 = hybrid.SCPM(mission, NUM_AGENTS, list(range(2)))
    #hybrid.motap_MS3_test_cpu(scpm_t3, msg3_env, w1, eps1, 1, 1000, 50)

    constraint_threshold = [1.] * NUM_AGENTS + [0.7] * NUM_TASKS
    target = np.random.uniform(low=-LB,high=-UB,size=NUM_AGENTS).tolist() + \
        [0.6] * NUM_TASKS
    if args.model == 'mamdp':
        hybrid.motap_mamdp_synth(scpm2_t3, env, w1, target, eps1, eps2, 
                                DEBUG, 5000, UNSTABLE, constraint_threshold)
    elif args.model == 'scpm':
        hybrid.motap_synth(scpm_t3, env, w1, target, eps1, eps2, 
                            DEBUG, 1000, UNSTABLE, constraint_threshold)
    else:
        hybrid.motap_mamdp_synth(scpm2_t3, env, w1, target, eps1, eps2, 
                                DEBUG, 5000, 200, constraint_threshold)
        hybrid.motap_synth(scpm_t3, env, w1, target, eps1, eps2, 
                            DEBUG, 1000, 50, constraint_threshold)

# MDP Type Mixture 13
elif args.mdp_type == 5:
    model_ = [hybrid.Model("MS1", list(range(5)), 0, p_[0])]
    model_.append(hybrid.Model("MS3", list(range(7)), 0, p_[1]))

    env = hybrid.MAS([0, 1])
    for _ in range(NUM_AGENTS):
        env.add_environment(model_.pop())
    env.create_order([0, 1])

    mission = hybrid.Mission()
    
    for msg in range(NUM_TASKS):
        dfa = vg_message_sending_task(msg + 1)
        mission.add_task(dfa)

    miss_mamdp = hybrid.Mission()

    for msg in range(NUM_TASKS):
        dfa = negvg_message_sending_task(msg + 1)
        miss_mamdp.add_task(dfa)
    scpm2 = hybrid.SCPM(miss_mamdp, NUM_AGENTS, list(range(2)))

    scpm = hybrid.SCPM(mission, NUM_AGENTS, list(range(2)))
    #hybrid.motap_MS3_test_cpu(scpm_t3, msg3_env, w1, eps1, 1, 1000, 50)

    constraint_threshold = [1.2] * NUM_AGENTS + [0.7] * NUM_TASKS
    target = np.random.uniform(low=-LB,high=-UB,size=NUM_AGENTS).tolist() + \
        [0.6] * NUM_TASKS
    if args.model == 'mamdp':
        hybrid.motap_mamdp_synth(scpm2, env, w1, target, eps1, eps2, 
                                DEBUG, 5000, UNSTABLE, constraint_threshold)
    elif args.model == 'scpm':
        hybrid.motap_synth(scpm, env, w1, target, eps1, eps2, 
                            DEBUG, 1000, 50, constraint_threshold)
    else:
        hybrid.motap_mamdp_synth(scpm2, env, w1, target, eps1, eps2, 
                                DEBUG, 5000, UNSTABLE, constraint_threshold)
        hybrid.motap_synth(scpm, env, w1, target, eps1, eps2, 
                            DEBUG, 1000, 50, constraint_threshold)

# MDP Type Mixture 23
elif args.mdp_type == 6:
    model_ = [hybrid.Model("MS2", list(range(6)), 0, p_[0])]
    model_.append(hybrid.Model("MS3", list(range(7)), 0, p_[1]))

    env = hybrid.MAS([0, 1])
    for _ in range(NUM_AGENTS):
        env.add_environment(model_.pop())
    env.create_order([0, 1])

    mission = hybrid.Mission()
    
    for msg in range(NUM_TASKS):
        dfa = vg_message_sending_task(msg + 1)
        mission.add_task(dfa)

    miss_mamdp = hybrid.Mission()

    for msg in range(NUM_TASKS):
        dfa = negvg_message_sending_task(msg + 1)
        miss_mamdp.add_task(dfa)
    scpm2 = hybrid.SCPM(miss_mamdp, NUM_AGENTS, list(range(2)))

    scpm = hybrid.SCPM(mission, NUM_AGENTS, list(range(2)))
    #hybrid.motap_MS3_test_cpu(scpm_t3, msg3_env, w1, eps1, 1, 1000, 50)

    constraint_threshold = [1.] * NUM_AGENTS + [0.7] * NUM_TASKS
    target = np.random.uniform(low=-LB,high=-UB,size=NUM_AGENTS).tolist() + \
        [0.5] * NUM_TASKS
    if args.model == 'mamdp':
        hybrid.motap_mamdp_synth(scpm2, env, w1, target, eps1, eps2, 
                                DEBUG, 5000, UNSTABLE, constraint_threshold)
    elif args.model == 'scpm':
        hybrid.motap_synth(scpm, env, w1, target, eps1, eps2, 
                            DEBUG, 1000, 50, constraint_threshold)
    else:
        hybrid.motap_mamdp_synth(scpm2, env, w1, target, eps1, eps2, 
                                DEBUG, 5000, UNSTABLE, constraint_threshold)
        hybrid.motap_synth(scpm, env, w1, target, eps1, eps2, 
                            DEBUG, 1000, 50, constraint_threshold)