import hybrid

ex = hybrid.Example()

words = ["", "a", "b"]

def never_s1_until_s3():
    task = hybrid.DFA(
        list(range(4)),
        0, 
        [1], # accepting
        [3], #rejecting,
        [2], #done
    )
    task.add_transition(0, "a", 1)
    task.add_transition(0, "b", 3)
    task.add_transition(0, "", 0)
    for w in words:
        task.add_transition(1, w, 2)
        task.add_transition(2, w, 2)
        task.add_transition(3, w, 3)
    return task

mission = hybrid.Mission()
dfa = never_s1_until_s3()
mission.add_task(dfa)
scpm = hybrid.SCPM(mission, 1, list(range(2))) # A single agent, 2 actions, carrying out reachability task

t = [-2.5, 0.7] # achievable
#t = [-1.8, 0.9] # unachievable
eps = 0.0001
w = [1., 0]
hybrid.example_cpu(scpm, ex, w, 1e-6, 1)
hybrid.ex_synthesis(scpm, ex, w, t, 1e-6, 1e-4, 2) # set verbosity to 2 (last param) to print pareto points
