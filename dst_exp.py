import hybrid
import numpy as np
import random
import argparse


words = ["0_7", "8_2", "11_5", "14_0", "15_1", "16_1", 
        "19_6", "20_3", "22_4", "23_7", "neg", ""]

def t1():
    task = hybrid.DFA(
        list(range(8)),
        0,
        [6],
        [5],
        [7]
    )
    acc_w = ["0_7", "8_2", "11_5"]
    for q in range(0, 5):
        for w in acc_w:
            task.add_transition(q, w, 6)

        task.add_transition(q, "neg", q + 1)
        for w in set(words).difference(set(acc_w).union({"neg"})):
            task.add_transition(q, w, q)

    for w in words:
        task.add_transition(5, w, 5)
        task.add_transition(6, w, 7)
        task.add_transition(7, w, 7)

    return task

def t2():
    task = hybrid.DFA(
        list(range(4)),
        0,
        [1],
        [3],
        [2]
    )
    acc_w = ["20_3", "22_4", "23_7"]
    for w in acc_w:
        task.add_transition(0, w, 1)

    task.add_transition(0, "neg", 3)
    for w in set(words).difference(set(acc_w).union({"neg"})):
        task.add_transition(0, w, 0)

    for w in words:
        task.add_transition(1, w, 2)
        task.add_transition(2, w, 2)
        task.add_transition(3, w, 3)

    return task


def t3():
    task = hybrid.DFA(
        list(range(4)),
        0,
        [1],
        [3],
        [2]
    )
    acc_w = ["14_0", "15_1", "16_1", "19_6"]
    for w in acc_w:
        task.add_transition(0, w, 1)

    task.add_transition(0, "neg", 3)
    for w in set(words).difference(set(acc_w).union({"neg"})):
        task.add_transition(0, w, 0)

    for w in words:
        task.add_transition(1, w, 2)
        task.add_transition(2, w, 2)
        task.add_transition(3, w, 3)

    return task

def find_t1():
    task = hybrid.DFA(
        list(range(4)),
        0,
        [1],
        [3],
        [2]
    )
    acc_w = ["0_7"]
    for w in acc_w:
        task.add_transition(0, w, 1)

    task.add_transition(0, "neg", 3)
    for w in set(words).difference(set(acc_w).union({"neg"})):
        task.add_transition(0, w, 0)

    for w in words:
        task.add_transition(1, w, 2)
        task.add_transition(2, w, 2)
        task.add_transition(3, w, 3)

    return task

def find_t2():
    task = hybrid.DFA(
        list(range(4)),
        0,
        [1],
        [3],
        [2]
    )
    acc_w = ["8_2"]
    for w in acc_w:
        task.add_transition(0, w, 1)

    task.add_transition(0, "neg", 3)
    for w in set(words).difference(set(acc_w).union({"neg"})):
        task.add_transition(0, w, 0)

    for w in words:
        task.add_transition(1, w, 2)
        task.add_transition(2, w, 2)
        task.add_transition(3, w, 3)

    return task

def find_t3():
    task = hybrid.DFA(
        list(range(4)),
        0,
        [1],
        [3],
        [2]
    )
    acc_w = ["11_5"]
    for w in acc_w:
        task.add_transition(0, w, 1)

    task.add_transition(0, "neg", 3)
    for w in set(words).difference(set(acc_w).union({"neg"})):
        task.add_transition(0, w, 0)

    for w in words:
        task.add_transition(1, w, 2)
        task.add_transition(2, w, 2)
        task.add_transition(3, w, 3)

    return task

def find_t4():
    task = hybrid.DFA(
        list(range(4)),
        0,
        [1],
        [3],
        [2]
    )
    acc_w = ["14_0"]
    for w in acc_w:
        task.add_transition(0, w, 1)

    task.add_transition(0, "neg", 3)
    for w in set(words).difference(set(acc_w).union({"neg"})):
        task.add_transition(0, w, 0)

    for w in words:
        task.add_transition(1, w, 2)
        task.add_transition(2, w, 2)
        task.add_transition(3, w, 3)

    return task

def find_t5():
    task = hybrid.DFA(
        list(range(4)),
        0,
        [1],
        [3],
        [2]
    )
    acc_w = ["15_1"]
    for w in acc_w:
        task.add_transition(0, w, 1)

    task.add_transition(0, "neg", 3)
    for w in set(words).difference(set(acc_w).union({"neg"})):
        task.add_transition(0, w, 0)

    for w in words:
        task.add_transition(1, w, 2)
        task.add_transition(2, w, 2)
        task.add_transition(3, w, 3)

    return task

def find_t6():
    task = hybrid.DFA(
        list(range(4)),
        0,
        [1],
        [3],
        [2]
    )
    acc_w = ["16_1"]
    for w in acc_w:
        task.add_transition(0, w, 1)

    task.add_transition(0, "neg", 3)
    for w in set(words).difference(set(acc_w).union({"neg"})):
        task.add_transition(0, w, 0)

    for w in words:
        task.add_transition(1, w, 2)
        task.add_transition(2, w, 2)
        task.add_transition(3, w, 3)

    return task

def find_t7():
    task = hybrid.DFA(
        list(range(4)),
        0,
        [1],
        [3],
        [2]
    )
    acc_w = ["19_6"]
    for w in acc_w:
        task.add_transition(0, w, 1)

    task.add_transition(0, "neg", 3)
    for w in set(words).difference(set(acc_w).union({"neg"})):
        task.add_transition(0, w, 0)

    for w in words:
        task.add_transition(1, w, 2)
        task.add_transition(2, w, 2)
        task.add_transition(3, w, 3)

    return task

def find_t8():
    task = hybrid.DFA(
        list(range(4)),
        0,
        [1],
        [3],
        [2]
    )
    acc_w = ["20_3"] #["20_3", "22_4", "23_7"]
    for w in acc_w:
        task.add_transition(0, w, 1)

    task.add_transition(0, "neg", 3)
    for w in set(words).difference(set(acc_w).union({"neg"})):
        task.add_transition(0, w, 0)

    for w in words:
        task.add_transition(1, w, 2)
        task.add_transition(2, w, 2)
        task.add_transition(3, w, 3)

    return task

def find_t9():
    task = hybrid.DFA(
        list(range(4)),
        0,
        [1],
        [3],
        [2]
    )
    acc_w = ["22_4"] #["20_3", "22_4", "23_7"]
    for w in acc_w:
        task.add_transition(0, w, 1)

    task.add_transition(0, "neg", 3)
    for w in set(words).difference(set(acc_w).union({"neg"})):
        task.add_transition(0, w, 0)

    for w in words:
        task.add_transition(1, w, 2)
        task.add_transition(2, w, 2)
        task.add_transition(3, w, 3)

    return task

def find_t10():
    task = hybrid.DFA(
        list(range(4)),
        0,
        [1],
        [3],
        [2]
    )
    acc_w = ["23_7"] #["20_3", "22_4", "23_7"]
    for w in acc_w:
        task.add_transition(0, w, 1)

    task.add_transition(0, "neg", 3)
    for w in set(words).difference(set(acc_w).union({"neg"})):
        task.add_transition(0, w, 0)

    for w in words:
        task.add_transition(1, w, 2)
        task.add_transition(2, w, 2)
        task.add_transition(3, w, 3)

    return task

NUM_TASKS = 6
NUM_AGENTS = 2
UNSTABLE = 200
DEBUG = 1

model1 = hybrid.DSTModel(0, 0)
model2 = hybrid.DSTModel(0, 9)
models = [model1, model2]

eps1 = 1e-6
eps2 = 0.01

w1 = [0.001 / NUM_AGENTS] * NUM_AGENTS + [0.999 / NUM_TASKS] * NUM_TASKS

env = hybrid.DSTMAS([0, 1])
for _ in range(NUM_AGENTS):
    env.add_environment(models.pop())
env.create_order([0, 1])

mission = hybrid.Mission()

#for msg in range(NUM_TASKS):
dfa1 = find_t1()
dfa2 = find_t2()
dfa3 = find_t4()
dfa4 = find_t5()
dfa5 = find_t8()
dfa6 = find_t9()
mission.add_task(dfa1)
mission.add_task(dfa2)
mission.add_task(dfa3)
mission.add_task(dfa4)
mission.add_task(dfa5)
mission.add_task(dfa6)

scpm = hybrid.SCPM(mission, NUM_AGENTS, list(range(4)))

constrain_threshold = [1.0] * NUM_AGENTS + [0.5] * NUM_TASKS
target = [-5, -10, 0.8, 0.8, 0.8, 0.8]
agent_init_states = [11 * 0 + 0, 11 * 0 + 9]
hybrid.dst_stapu_model(scpm, env, w1, eps1, 1, 1000, 150, agent_init_states)
#hybrid.dst_build_test(scpm, env)


