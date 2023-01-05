import hybrid
import itertools
import random
import argparse
import numpy as np


random.seed(a=12345, version=2)
np.random.seed = 12345

# ------------------------------------------------------------------------------
# ARGS: Program args for repeating and controlling experiments
# ------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Test MORAP framework with smart-warehouse.")
parser.add_argument('--agents', dest='num_agents', default=2, help='the number of agents in MAS', 
                    required=True, type=int)
parser.add_argument('--size', dest='size', default=6, help='size of the warehouse 6 | 12', 
                    required=True, choices=[6, 8, 10, 12], type=int)
parser.add_argument('--hware', dest='hardware', default='CPU', 
                    choices=['CPU', 'GPU', 'HYBRID'])
parser.add_argument('--cpu', dest='num_cpus', default=2, type=int, 
                    help='The number of cpus to use in hybrid infrastructure')
parser.add_argument('-d', dest='debug', default=0, type=int, choices=[0, 1, 2, 3])
parser.add_argument('-e', dest='vi_eps', default=1e-6, type=float, 
                    help="The threshold for value iteration.")
parser.add_argument('--eps', dest='synth_eps', default=1e-4, type=float, 
                    help="The threshold for modell checking synthesis algorithm")
parser.add_argument('--iter', dest='max_iter', default=1000, type=int,
                    help='The max number of iterations in VI before loop force terminates')
parser.add_argument('--unstable', dest='max_unstable', default=30, type=int, 
                    help='The number of divegent state-values before infinity is recognised')
parser.add_argument('--clb', dest="clower_bound", default=14., type=float, 
                    help="The lower bound on the cost selection distribution")
parser.add_argument('--cub', dest="cupper_bound", default=16., type=float, 
                    help="The upper bound on the cost selection distribution")

args = parser.parse_args()
#
# Params
#
NUM_AGENTS = args.num_agents
NUM_TASKS = NUM_AGENTS
HARDWARE = args.hardware
debug = args.debug
NUM_CPUs = args.num_cpus
EPSILON1 = args.vi_eps
EPSILON2 = args.synth_eps
MAX_ITER = args.max_iter
MAX_UNSTABLE = args.max_unstable
LB = args.clower_bound
UB = args.cupper_bound

# ------------------------------------------------------------------------------
# SETUP: Construct the structures for agent to recognise task progress
# ------------------------------------------------------------------------------

task_progress = {0: "initial", 1: "in_progress", 2: "success", 3: "fail"}

# Set the initial agent locations up front
# We can set the feed points up front as well because they are static

size = args.size
feedpoints = [(size - 1, size // 2)]
# construct all the positions on the outside of the grid which is not the feed position
outer_square  = [(0, i) for i in range(size)] + [(i, 0) for i in range(size)] + \
                [(size - 1, i) for i in range(size) if i not in feedpoints] + \
                [(i, size - 1) for i in range(size)]

#init_agent_positions = random.sample(outer_square, k=NUM_AGENTS)
init_agent_positions = random.choices(outer_square, k=NUM_AGENTS)
print("\n")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("        Warehouse Attributes        ")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("init agent positions", init_agent_positions)
#init_agent_positions = [(0,0)]
#feedpoints = [(2, 2)]
print("Feed points", feedpoints)

# ------------------------------------------------------------------------------
# Env Setup: Construct the warehouse model as an Open AI gym python environment
# ------------------------------------------------------------------------------
actions_to_dir = [[1, 0],[0, 1],[-1, 0],[0, -1]]
warehouse_api = hybrid.Warehouse(
    size, NUM_AGENTS, feedpoints, init_agent_positions, 
    actions_to_dir, 0.995, 1 # <========================== 1 means normal warehouse layout
)


def warehouse_replenishment_task():
    task = hybrid.DFA(list(range(0, 8)), 0, [5], [7], [6])
    # attempt to goto the rack positon without carrying anything
    omega = set(warehouse_api.get_words())

    # The first transition determines if the label is at the rack
    task.add_transition(0, "RS_NC", 1)
    excluded_words = ['_'.join(x) for x in list(itertools.product(["RS", "RE", "NFR", "F"], ["P", "D", "CR", "CNR"]))]
    excluded_words.append("RE_NC")
    for w in excluded_words: 
        task.add_transition(0, f"{w}", 7)
    excluded_words.append("RS_NC")
    for w in omega.difference(set(excluded_words)):
        task.add_transition(0, f"{w}", 0)
    # The second transition determines whether the agent picked up the rack at the 
    # required coord
    task.add_transition(1, "RS_P", 2)
    excluded_words = ['_'.join(x) for x in list(itertools.product(["NFR"], ["P"]))]
    for w in excluded_words:
        task.add_transition(1, f"{w}", 7)
    excluded_words.append("RS_P")
    for w in omega.difference(set(excluded_words)):
        task.add_transition(1, f"{w}", 1)
    # The third transition takes the agent to the feed position while carrying
    task.add_transition(2, "F_CNR", 3)
    excluded_words = ['_'.join(x) for x in list(itertools.product(["F", "RS", "RE", "NFR"], ["NC", "P", "D", "CR"]))]
    for w in excluded_words:
        task.add_transition(2, f"{w}", 7)
    excluded_words.append("F_CNR")
    for w in omega.difference(set(excluded_words)):
        task.add_transition(2, f"{w}", 2)
    # The fourth transition takes the agent from the feed position while carrying 
    # back to the rack position
    task.add_transition(3, "RS_CNR", 4)
    excluded_words = ['_'.join(x) for x in list(itertools.product(["F", "RS", "RE", "NFR"], ["NC", "P", "D", "CR"]))]
    #excluded_words.append("RS_CNR")
    for w in excluded_words:
        task.add_transition(3, f"{w}", 7)
    excluded_words.append("RS_CNR")
    for w in omega.difference(set(excluded_words)):
        task.add_transition(3, f"{w}", 3)
    # The fifth transition tells the agent to drop the rack at the required square
    task.add_transition(4, "RS_D", 5)
    for w in omega.difference(set(["RS_D"])):
        task.add_transition(4, f"{w}", 4)
    for w in omega:
        task.add_transition(5, f"{w}", 6)
    for w in omega:
        task.add_transition(6, f"{w}", 6)
    for w in omega:
        task.add_transition(7, f"{w}", 7)
    
    return task

#rack_samples = random.sample([*warehouse_api.racks], k=NUM_TASKS)
rack_samples = random.choices([*warehouse_api.racks], k=NUM_TASKS)
#rack_samples = [*warehouse_api.racks]
print("rack samples", rack_samples)
#warehouse_api.add_task_rack_end(0, rack_samples[0])
#warehouse_api.add_task_rack_start(0, rack_samples[0])
#warehouse_api.add_task_feed(0, feedpoints[0])
for k in range(NUM_TASKS):
    warehouse_api.add_task_rack_end(k, rack_samples[k])
    warehouse_api.add_task_rack_start(k, rack_samples[k])
    warehouse_api.add_task_feed(k, feedpoints[0])

mission = hybrid.Mission()
dfa = warehouse_replenishment_task()
for task in range(NUM_TASKS):
    mission.add_task(dfa)

scpm = hybrid.SCPM(mission, NUM_AGENTS, list(range(6)))
#w = [0] * NUM_AGENTS + [1. / NUM_TASKS] * NUM_TASKS
#w = [1. / NUM_TASKS + NUM_AGENTS] * (NUM_TASKS + NUM_AGENTS)
w = [0.01 / NUM_AGENTS] * NUM_AGENTS + [0.99 / NUM_TASKS] * NUM_TASKS
print("Check sum w = 1", sum(w))
#hybrid.test_warehouse_gpu_only(scpm, warehouse_api, w, eps, debug) 
#hybrid.test_warehouse_CPU_only(scpm, warehouse_api, w, eps, debug)
#hybrid.test_warehouse_single_CPU(scpm, warehouse_api, w, eps, debug)
#hybrid.test_warehouse_hybrid(scpm, warehouse_api, w, eps, NUM_CPUs, debug)
constraint_threshold = [1.] * NUM_AGENTS + [0.] * NUM_TASKS
target = np.random.uniform(low=-LB,high=-UB,size=NUM_AGENTS).tolist() + [0.7] * NUM_TASKS
print(f"Target: {np.round(target, 2)}")

# Run decentralised model building and synthesis
hybrid.test_warehouse_dec(scpm, warehouse_api, w, target, EPSILON1, EPSILON2, debug, 
                          HARDWARE, NUM_CPUs, MAX_ITER, MAX_UNSTABLE, constraint_threshold)
