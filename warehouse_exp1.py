import hybrid
import itertools
import random

random.seed(a=12345, version=2)

#
# Params
#
NUM_TASKS = 1
NUM_AGENTS = 1


# ------------------------------------------------------------------------------
# SETUP: Construct the structures for agent to recognise task progress
# ------------------------------------------------------------------------------

task_progress = {0: "initial", 1: "in_progress", 2: "success", 3: "fail"}

# Set the initial agent locations up front
# We can set the feed points up front as well because they are static

#init_agent_positions = [(0, 0), (0, 2), (0, 4), (0, 6), (2, 0), 
#                        (2, 0), (4, 0), (6, 0), (8, 0), (9, 0)]
init_agent_positions = [(0,0)]
size = 10
feedpoints = [(size - 1, size // 2)]
#feedpoints = [(2, 2)]
print("Feed points", feedpoints)

# ----------------------------------------
# Env Setup: Construct the warehouse model 
# ----------------------------------------
actions_to_dir = [[1, 0],[0, 1],[-1, 0],[0, -1]]
warehouse_api = hybrid.Warehouse(
    size, NUM_AGENTS, feedpoints, init_agent_positions, 
    actions_to_dir, 0.995, 1 # <============================ 0 means test warehouse layout
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
rack_samples = [*warehouse_api.racks]
print("rack samples", rack_samples)
warehouse_api.add_task_rack_end(0, rack_samples[0])
warehouse_api.add_task_rack_start(0, rack_samples[0])
warehouse_api.add_task_feed(0, feedpoints[0])
#for k in range(NUM_TASKS):
#    warehouse_api.add_task_rack_end(k, rack_samples[k])
#    warehouse_api.add_task_rack_start(k, rack_samples[k])
#    warehouse_api.add_task_feed(k, feedpoints[0])

mission = hybrid.Mission()
dfa = warehouse_replenishment_task()
mission.add_task(dfa)
eps = 1.0e-6
scpm = hybrid.SCPM(mission, NUM_AGENTS, list(range(6)))
#w = [0] * NUM_AGENTS + [1. / NUM_TASKS] * NUM_TASKS
#w = [1. / NUM_TASKS + NUM_AGENTS] * (NUM_TASKS + NUM_AGENTS)
#w = [0.01 / NUM_AGENTS] * NUM_AGENTS + [0.99 / NUM_TASKS] * NUM_TASKS
w = [0., 1]
print("Check sum w = 1", sum(w))

hybrid.warehouse_make_prism_file(scpm, warehouse_api)
hybrid.test_warehouse_policy_optimisation(scpm, warehouse_api, w, eps)