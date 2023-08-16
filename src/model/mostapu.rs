use hashbrown::{HashMap, HashSet};
use sprs::{CsMatBase, TriMatI};
use std::hash::Hash;
use crate::agent::env::Env;
use std::collections::VecDeque;
use super::general::ModelFns;
use super::scpm::SCPM;

#[derive(Clone, Hash, Eq, PartialEq, Debug)]
pub struct StapuState<S> {
    pub s: S,
    pub Q: Vec<i32>,
    pub agentid: i32, 
    pub active_tasks: Option<Vec<i32>>,
    pub remaining: Vec<i32>
}

#[derive(Clone)]
/// This is the centralised SCPM Model
pub struct STAPU<S> {
    pub initial_state: StapuState<S>, // (State, q, taskid, #, current agent)
    pub states: Vec<StapuState<S>>,
    pub actions: Vec<i32>,
    pub P: CsMatBase<f32, i32, Vec<i32>, Vec<i32>, Vec<f32>>,
    pub R: CsMatBase<f32, i32, Vec<i32>, Vec<i32>, Vec<f32>>,
    pub adjusted_state_act_pair: Vec<i32>, 
    pub enabled_actions: Vec<i32>, 
    pub state_map: HashMap<StapuState<S>, usize>,
    pub accepting: HashMap<i32, Vec<usize>>
}

fn add_to_active(active_task_list: &Option<Vec<i32>>, task: i32) -> Option<Vec<i32>>
 {
    if active_task_list.is_none() {
        return Some(vec![task]);
    } else {
        let mut old  = active_task_list.as_ref().unwrap().to_vec();
        if !old.contains(&task) {
            old.push(task);
        }
        return Some(old)
    }
}

impl<S> ModelFns<StapuState<S>> for STAPU<S> {
    fn get_states(&self) -> &[StapuState<S>] {
        &self.states
    }

    fn get_enabled_actions(&self) -> &[i32] {
        &self.enabled_actions
    }
}

impl<S> STAPU<S>
where S: Copy + Eq + Hash, StapuState<S>: Eq + Hash + Clone {
    pub fn new(
        initial_state: StapuState<S>,
        actions: Vec<i32>
    ) -> Self {
        Self {
            initial_state,
            states: Vec::new(),
            actions: actions,
            P: CsMatBase::empty(sprs::CompressedStorage::CSR, 0), 
            R: CsMatBase::empty(sprs::CompressedStorage::CSR, 0),
            adjusted_state_act_pair: Vec::new(),
            enabled_actions: Vec::new(),
            state_map: HashMap::new(),
            accepting: HashMap::new()
        }
    }

    fn insert_state(&mut self, state: StapuState<S>) {
        let state_idx = self.states.len();
        self.states.push(state.clone());
        self.insert_state_mapping(state, state_idx);
    }

    fn insert_state_mapping(&mut self, state: StapuState<S>, state_idx: usize) {
        self.state_map.insert(state, state_idx);
    }
}

pub fn MOSTAPU_bfs<S, E>(
    initial_state: StapuState<S>,
    mdp: &E,
    n_agents: usize,
    n_tasks: usize,
    n_objs: usize, 
    model: &SCPM,
    actions: &[i32],
    initial_agent_states: &[S]
) -> STAPU<S>
where S: Copy + Clone + std::fmt::Debug + Eq + Hash, E: Env<S> {
    let mut stapu = STAPU::new(initial_state.clone(), actions.to_vec());

    let mut visited: HashSet<StapuState<S>> = HashSet::new();
    let mut state_rewards: HashSet<i32> = HashSet::new();
    let mut stack: VecDeque<StapuState<S>> = VecDeque::new();
    let mut adjusted_state_action: HashMap<i32, i32> = HashMap::new();
    let mut enabled_actions: HashMap<i32, i32> = HashMap::new();
    //let mut accepting: HashMap<i32, Vec<usize>> = HashMap::new();

    // construct a new triple matrix fo rthe transitions and the rewards
    // Transition triples
    let mut prows: Vec<i32> = Vec::new();
    let mut pcols: Vec<i32> = Vec::new();
    let mut pvals: Vec<f32> = Vec::new();
    // Rewards triples
    let mut rrows: Vec<i32> = Vec::new();
    let mut rcols: Vec<i32> = Vec::new();
    let mut rvals: Vec<f32> = Vec::new();

    stack.push_back(initial_state.clone());
    visited.insert(initial_state.clone());
    // actions enabled
    stapu.insert_state(initial_state.clone());
    adjusted_state_action
        .insert(*stapu.state_map.get(&initial_state).unwrap() as i32, 0);
    let mut largest_row: i32 = 0;

    //for _ in 0..10 {
    while !stack.is_empty() {
        let state = stack.pop_front().unwrap();
        let sidx = *stapu.state_map.get(&state).unwrap();
        //println!("sidx: {}, state: {:?}", sidx, state);
        let row_idx = *adjusted_state_action.get(&(sidx as i32)).unwrap();

        let mut available_actions: Vec<i32> = actions[1 + n_tasks..].to_vec();
        //let dfa = model.tasks.get_task_copy(state.taskid as usize);

        // We have to work out the available actions here
        // The MOSTAPU model available actions are (b1, a1, ..., ak)
        // If the state is initial in the MDP and each q are in initial or final 
        // but there is some task which is still in the initial state
        // If all of the states have been finished then just the agent actions
        // will be available
        /*let all_fin_or_init = (0..n_tasks)
            .map(|t| 
                model.tasks.get_task(t).fin_or_init(state.Q[t])
            ).all(|x| x);
        */

        /*let all_fin = (0..n_tasks)
            .map(|t| 
                model.tasks.get_task(t).check_fin(state.Q[t])
            ).all(|x| x);
        */
        // First find out which tasks have already been completed
        if //state.s == mdp.get_init_state(state.agentid as usize) && 
            state.remaining.len() > 0 &&
            state.active_tasks.is_none() &&
            state.agentid < n_agents as i32 - 1 {
            if state.remaining.len() > 0 {
                let tasks_: Vec<i32> = state.remaining.iter().map(|x| * x + 1).collect();
                available_actions = [&actions[0..1], &tasks_[..], &actions[1 + n_tasks..]].concat()
            } else {
                available_actions = [&actions[0..1], &actions[1 + n_tasks..]].concat();
            }
        // Next if tasks are already active but there are still tasks remaining
        // then it is also possible for an agent to begin a new task
        } else if //state.active_tasks.is_some() &&
            state.remaining.len() > 0 { // determine which tasks are active already
            let tasks_: Vec<i32> = state.remaining.iter().map(|x| *x + 1).collect();
            available_actions = [&tasks_[..], &actions[1+ n_tasks..]].concat();
        } 

        //println!("Available actions: {:?}", available_actions);

        for action in available_actions {
            match action {
                0 => { 
                    // this is equivalent to passing the remaining set of 
                    // tasks onto the next agent
                    let sprime = StapuState {
                        s: initial_agent_states[state.agentid as usize + 1],
                        Q: state.Q.to_vec(),
                        agentid: state.agentid + 1,
                        active_tasks: None,
                        remaining: state.remaining.to_vec()
                    };
                    match adjusted_state_action.get_mut(&(sidx as i32 + 1)) {
                        Some(adj_sidx) => { *adj_sidx += 1; },
                        None => {
                            adjusted_state_action.insert(sidx as i32 + 1, 
                                adjusted_state_action.get(&(sidx as i32)).unwrap() + 1);
                        },
                    }
                    let current_row;
                    match enabled_actions.get_mut(&(sidx as i32)) {
                        Some(x) => {
                            current_row = *x; 
                            *x += 1;
                        }
                        None => { 
                            current_row = 0;
                            enabled_actions.insert(sidx as i32, 1); 
                        }
                    }
                    if !visited.contains(&sprime) {
                        visited.insert(sprime.clone());
                        stack.push_back(sprime.clone());
                        stapu.insert_state(sprime.clone());
                    }
                    let sprime_idx = *stapu.state_map
                        .get(&sprime)
                        .unwrap();
                    prows.push(row_idx + current_row); 
                    pcols.push(sprime_idx as i32); pvals.push(1.0);
                    largest_row = row_idx + current_row + 1;
                }
                i if 1 <= i && i < n_tasks as i32 + 1 => {
                    // this is equivalent to the same agent starting a new task
                    // reminder here that i is an action
                    let mut new_remaining = state.remaining.to_vec();
                    new_remaining.retain(|x| *x != i - 1);
                    let sprime = StapuState {
                        s: state.s,
                        Q: state.Q.to_vec(),
                        agentid: state.agentid,
                        active_tasks: add_to_active(&state.active_tasks, i - 1),
                        remaining: new_remaining
                    };
                    match adjusted_state_action.get_mut(&(sidx as i32 + 1)) {
                        Some(adj_sidx) => { *adj_sidx += 1; },
                        None => {
                            adjusted_state_action.insert(sidx as i32 + 1, 
                                adjusted_state_action.get(&(sidx as i32)).unwrap() + 1);
                        },
                    }
                    let current_row;
                    match enabled_actions.get_mut(&(sidx as i32)) {
                        Some(x) => {
                            current_row = *x; 
                            *x += 1;
                        }
                        None => { 
                            current_row = 0;
                            enabled_actions.insert(sidx as i32, 1); 
                        }
                    }
                    if !visited.contains(&sprime) {
                        visited.insert(sprime.clone());
                        stack.push_back(sprime.clone());
                        stapu.insert_state(sprime.clone());
                    }
                    let sprime_idx = *stapu.state_map
                        .get(&sprime)
                        .unwrap();
                    prows.push(row_idx + current_row); 
                    pcols.push(sprime_idx as i32); pvals.push(1.0);
                    largest_row = row_idx + current_row + 1;
                } 
                _ => { 
                    // this is equivalent to one of the base agent actions
                    // if there is a task active, continue working on that task
                    // otherwise the MDP continues looping but there is never
                    // any task progress
                    if state.active_tasks.is_some() {
                        match mdp.step_(
                            state.s, 
                            (action - n_tasks as i32 - 1) as u8, 
                            0, 
                            state.agentid
                        ) {
                            Ok(v) => {

                                if !v.is_empty() {
    
                                    let current_row;
                                    match enabled_actions.get_mut(&(sidx as i32)) {
                                        Some(x) => {
                                            current_row = *x; 
                                            *x += 1;
                                        }
                                        None => { 
                                            current_row = 0;
                                            enabled_actions.insert(sidx as i32, 1); 
                                        }
                                    }
                                    // plus one to the state-action pair conversion
                                    match adjusted_state_action.get_mut(&(sidx as i32 + 1)) {
                                        Some(adj_sidx) => { *adj_sidx += 1; },
                                        None => {
                                            adjusted_state_action.insert(sidx as i32 + 1, 
                                                adjusted_state_action.get(&(sidx as i32)).unwrap() + 1);
                                        },
                                    }
            
                                    for (sprime, p, w) in v.iter() {
                                        // In the STAPU model, there could be multiple DFAs which we need to move forward
                                        let tasks = state.active_tasks.as_ref().unwrap().to_vec();
                                        // our goal is to make a new Q state, therefore we will need to transition any q
                                        // with the index of active task
                                        let mut Qnew = state.Q.to_vec();
                                        let mut new_active = state.active_tasks.as_ref().unwrap().to_vec();
                                        // Check if all tasks are done
                                        rrows.push(row_idx + current_row); rcols.push(state.agentid); rvals.push(-1.);
                                        for task in tasks {
                                            let dfa = model.tasks.get_task(task as usize);
                                            if !state_rewards.contains(&(row_idx + current_row as i32)) {
                                                //println!("s: {:?}, @s: {}, q: {}, a: {}", s, row_idx as usize + action, q, action);
                                                if dfa.accepting.contains(&state.Q[task as usize]) {
                                                    rrows.push(row_idx + current_row); rcols.push(task + n_agents as i32); rvals.push(1.);
                                                }
                                                state_rewards.insert(row_idx + current_row);
                                            }
                                            
                                            let qprime: i32 = dfa.get_transition(state.Q[task as usize], w);
                                            Qnew[task as usize] = qprime;
                                            // check if qprime is in done or rejecting which will inactivate the task
                                            if dfa.check_fin(qprime) {
                                                new_active.retain(|x| *x != task);
                                            }
                                            /*if dfa.accepting.contains(&qprime) {
                                                match accepting.get_mut(&state.active_tasks.unwrap()) {
                                                    Some(x) => { x.push(sprime_idx); }
                                                    None => { accepting.insert(state.active_tasks.unwrap(), vec![sprime_idx]); }
                                                }
                                            }*/
                                        }
                                        let new_state = StapuState { 
                                            s: *sprime, 
                                            Q: Qnew, 
                                            agentid: state.agentid, 
                                            active_tasks: if new_active.len() > 0 { Some(new_active) } else { None },
                                            remaining: state.remaining.to_vec()
                                        };
                                        if !visited.contains(&new_state) {
                                            visited.insert(new_state.clone());
                                            stack.push_back(new_state.clone());
                                            stapu.insert_state(new_state.clone());
                                        }
                                        let sprime_idx = *stapu.state_map
                                            .get(&new_state)
                                            .unwrap();
    
                                        // add in the transition to the CxxMatrix
                                        prows.push(row_idx + current_row); pcols.push(sprime_idx as i32); pvals.push(*p);
                                        largest_row = row_idx + current_row + 1;
                                        //println!("s: {:?}, @s: {}, q: {:?}, a: {}, sidx':{}. s': {:?}, q': {:?}, p: {}, w: {}", 
                                        //   state.s, row_idx + current_row as i32, new_state.Q, action, sprime_idx, sprime, state.Q, p, w);
                                    }
                                }
                            }
                            Err(_) => {}
                        }
                    } else {
                        // There are no active tasks
                        match mdp.step_(
                                state.s, 
                                (action - n_tasks as i32 - 1) as u8, 
                                -1 , 
                                state.agentid
                            ) {
                            Ok(v) => {
                                if !v.is_empty() {
                                    
                                    let current_row;
                                    match enabled_actions.get_mut(&(sidx as i32)) {
                                        Some(x) => {
                                            current_row = *x; 
                                            *x += 1;
                                        }
                                        None => { 
                                            current_row = 0;
                                            enabled_actions.insert(sidx as i32, 1); 
                                        }
                                    }
                                    let tasks_done: Vec<bool> = (0..model.tasks.size)
                                        .map(|k| model.tasks.get_task(k).zero_rewards(state.Q[k])).collect();
                                    let all_done = tasks_done.iter().all(|x| *x);
                                    if all_done {
                                        // do nothing
                                    } else {
                                        rrows.push(row_idx + current_row); rcols.push(state.agentid); rvals.push(-1.);
                                    }
    
                                    // plus one to the state-action pair conversion
                                    match adjusted_state_action.get_mut(&(sidx as i32 + 1)) {
                                        Some(adj_sidx) => { *adj_sidx += 1; },
                                        None => {
                                            adjusted_state_action.insert(sidx as i32 + 1, 
                                                adjusted_state_action.get(&(sidx as i32)).unwrap() + 1);
                                        },
                                    }
            
                                    for (sprime, p, _w) in v.iter() {
                                        let new_state = StapuState { 
                                            s: *sprime, 
                                            Q: state.Q.to_vec(), 
                                            agentid: state.agentid, 
                                            active_tasks: state.active_tasks.clone(),
                                            remaining: state.remaining.to_vec()
                                        };
                                        if !visited.contains(&new_state) {
                                            visited.insert(new_state.clone());
                                            stack.push_back(new_state.clone());
                                            stapu.insert_state(new_state.clone());
                                        }
                                        let sprime_idx = *stapu.state_map
                                            .get(&new_state)
                                            .unwrap();
                                        // add in the transition to the CxxMatrix
                                        prows.push(row_idx + current_row); pcols.push(sprime_idx as i32); pvals.push(*p);
                                        largest_row = row_idx + current_row + 1;
                                        //println!("s: {:?}, @s: {}, q: {:?}, a: {}, sidx':{}. s': {:?}, q': {:?}, p: {}, w: {}", 
                                        //   state.s, row_idx + current_row as i32, new_state.Q, action, sprime_idx, sprime, state.Q, p, _w);
                                    }
                                }
                            }
                            Err(_) => {}
                        }
                    }
                    
                }
            }
        }
        
    }
    let pTriMatr = TriMatI::<f32, i32>::from_triplets(
        (largest_row as usize, stapu.states.len()), prows, pcols, pvals
    );
    let rTriMatr = TriMatI::<f32, i32>::from_triplets(
        (largest_row as usize, n_objs), rrows, rcols, rvals
    );    
    //println!("enabled actions \n{:?}", enabled_actions);

    let mut vadj_pairs: Vec<i32> = vec![0; stapu.states.len()];
    let mut venbact: Vec<i32> = vec![0; stapu.states.len()];
    for sidx in 0..stapu.states.len() {
        vadj_pairs[sidx] = adjusted_state_action.remove(&(sidx as i32)).unwrap();
        venbact[sidx] = enabled_actions.remove(&(sidx as i32)).unwrap();
    }
    //println!("adjusted states\n{:?}", vadj_pairs);
    stapu.adjusted_state_act_pair = vadj_pairs;
    stapu.enabled_actions = venbact;
    
    // compress the matrices into CSR format
    stapu.P = pTriMatr.to_csr();
    stapu.R = rTriMatr.to_csr();
    //stapu.accepting = accepting;
    stapu
}   