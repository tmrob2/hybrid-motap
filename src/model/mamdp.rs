use std::{hash::Hash, collections::VecDeque};
use hashbrown::{HashMap, HashSet};
use sprs::{CsMatBase, TriMatI};
use super::{scpm::SCPM, general::ModelFns};
use crate::{agent::env::Env};

#[derive(Hash, Eq, PartialEq, Copy, Clone, Debug)]
pub struct Active {
    pub A: Option<i32>, 
    pub T: Option<i32>
}

/// State for Product MAMDP
#[derive(Hash, PartialEq, Eq, Clone, Debug)]
pub struct MAMDPState<S: Hash + Eq + Clone + Copy> {
    pub S: Vec<S>,
    pub Q: Vec<i32>,
    pub R: Vec<i32>,
    pub Active: Active
}

pub struct MOMAMDP<S: Hash + Eq + Clone + Copy> {
    pub initial_state: MAMDPState<S>,
    pub states: Vec<MAMDPState<S>>,
    pub actions: Vec<i32>,
    pub P: CsMatBase<f32, i32, Vec<i32>, Vec<i32>, Vec<f32>>,
    pub R: CsMatBase<f32, i32, Vec<i32>, Vec<i32>, Vec<f32>>,
    pub adjusted_state_act_pair: Vec<i32>,
    pub enabled_actions: Vec<i32>,
    pub state_map: HashMap<MAMDPState<S>, usize>,
    pub qmap: Vec<i32>,
    pub accepting: HashMap<i32, Vec<usize>>, // <task_id, states in which the task is acc>
    pub obj_rew: HashMap<i32, i32> 
}

impl<S> ModelFns<MAMDPState<S>> for MOMAMDP<S>
where S: Hash + Eq + Copy {
    fn get_states(&self) -> &[MAMDPState<S>] {
        &self.states
    }

    fn get_enabled_actions(&self) -> &[i32] {
        &self.enabled_actions
    }
}

impl<S> MOMAMDP<S> where S: Hash + Eq + Clone + Copy {
    pub fn new(
        initial_state: MAMDPState<S>,
        actions: Vec<i32>,
    ) -> Self {
        MOMAMDP { 
            initial_state, 
            states: Vec::new(), 
            actions, 
            P: CsMatBase::empty(sprs::CompressedStorage::CSR, 0), 
            R: CsMatBase::empty(sprs::CompressedStorage::CSR, 0),
            adjusted_state_act_pair: Vec::new(), 
            enabled_actions: Vec::new(), 
            state_map: HashMap::new(), 
            qmap: Vec::new(),
            accepting: HashMap::new(),
            obj_rew: HashMap::new()
        }
    }

    fn insert_state(&mut self, state: MAMDPState<S>) {
        let state_idx = self.states.len();
        self.states.push(state.clone());
        self.insert_state_mapping(state, state_idx);
    }

    fn insert_state_mapping(&mut self, state: MAMDPState<S>, state_idx: usize) {
        self.state_map.insert(state, state_idx);
    }
}

fn get_actions(remaining: &[i32], n_tasks: usize, n_agents: usize) -> Vec<i32> {
    let mut act_ = Vec::new();
    for r in remaining.iter() {
        for a in 0..n_agents {
            act_.push(a as i32 * n_tasks as i32 + r);
        }
    }
    act_
}

pub fn MOMAMDP_bfs<S, E>(
    initial_state: MAMDPState<S>,
    mdp: &E,
    n_tasks: usize,
    n_agents: usize,
    n_objs: usize,
    model: &SCPM,
    actions: &[i32]
) -> MOMAMDP<S>
where S: Copy + Clone + std::fmt::Debug + Eq + Hash, E: Env<S> {
    let mut mamdp = MOMAMDP::new(initial_state.clone(), actions.to_vec());
    
    let mut visited: HashSet<MAMDPState<S>> = HashSet::new();
    let mut state_rewards: HashSet<i32> = HashSet::new();
    let mut stack: VecDeque<MAMDPState<S>> = VecDeque::new();
    let mut adjusted_state_action: HashMap<i32, i32> = HashMap::new();
    let mut enabled_actions: HashMap<i32,i32> = HashMap::new();
    let mut accepting: HashMap<i32, Vec<usize>> = HashMap::new();
    let mut obj_rew: HashMap<i32, i32> = HashMap::new();

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
    mamdp.insert_state(initial_state.clone());
    adjusted_state_action
        .insert(*mamdp.state_map.get(&initial_state).unwrap() as i32, 0);
    let mut largest_row: i32 = 0;

    while !stack.is_empty() {
        let state = stack.pop_front().unwrap();
        let sidx = *mamdp.state_map.get(&state).unwrap();
        //println!("sidx: {}, state: {:?}", sidx, state);
        let row_idx = *adjusted_state_action.get(&(sidx as i32)).unwrap();

        let mut available_actions: Vec<i32> = actions[n_tasks * n_agents..].to_vec();
        
        if state.Active.A.is_some() && state.Active.T.is_none() {
            // this is the case where a task is currently not active
            // we use the remaining tasks as the action subscripts
            available_actions = [&get_actions(&state.R[..], n_tasks, n_agents)[..], 
                                 &actions[n_tasks * n_agents..]].concat();
        } else if state.Active.A.is_none() && state.Active.T.is_none() {
            available_actions = get_actions(&state.R[..], n_tasks, n_agents);
        }
        //println!("available actions: {:?}", available_actions);

        for action in available_actions {
            match action {
                i if i < (n_tasks * n_agents) as i32 => {
                    // we will use an array decoding trick to convert the action into 
                    // (agent, task)
                    // imagine that there is a 2d array
                    // T      1  2 ... m
                    // A  1   0  0 ... 0
                    //    2   1  0 ... 0
                    //    ... 
                    //    n   0  0 ... 0
                    // which is 1 where the allocated task to agent occurs 
                    // and 0 every where else
                    // we can encode this as i = x + width * y;
                    //                       x = i % width
                    //                       y = i / width
                    // Here width is the number of tasks (clearly)
                    // this is the case where we are picking a new agent
                    // to initiate a task
                    // construct a new state
                    // i is the chosen new task
                    let remaining: Vec<i32> = state.R.iter()
                        .cloned()
                        .filter(|x| *x != i)
                        .collect();
                    let sprime = MAMDPState {
                        S: state.S.to_vec(),
                        Q: state.Q.to_vec(),
                        R: remaining,
                        Active: Active { A: Some(i / (n_tasks as i32)), T: Some(i % (n_tasks as i32)) },
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
                        mamdp.insert_state(sprime.clone());
                    }
                    let sprime_idx = *mamdp.state_map
                        .get(&sprime)
                        .unwrap();
                    prows.push(row_idx + current_row); 
                    pcols.push(sprime_idx as i32); pvals.push(1.0);
                    largest_row = row_idx + current_row + 1;
                }
                _ => { 
                    // this is the case where we are just looking at the enabled 
                    // actions of the agent(s)
                    // Additional information, this is the case where there is a task active
                    // get the DFA relating to the active task
                    obj_rew.insert(sidx as i32, state.Active.A.unwrap());
                    if state.Active.T.is_some() {
                        let dfa = model.tasks.get_task(state.Active.T.unwrap() as usize);
                        match mdp.step_(
                            state.S[state.Active.A.unwrap() as usize], 
                            (action - (n_agents * n_tasks) as i32) as u8, 
                            state.Active.T.unwrap(), 
                            state.Active.A.unwrap()
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
                                    if !state_rewards.contains(&(row_idx + current_row as i32)) {
                                        //println!("s: {:?}, @s: {}, q: {}, a: {}", s, row_idx as usize + action, q, action);
                                        if dfa.accepting.contains(&state.Q[state.Active.T.unwrap() as usize])
                                            || dfa.done.contains(&state.Q[state.Active.T.unwrap() as usize])
                                            || dfa.rejecting.contains(&state.Q[state.Active.T.unwrap() as usize]) {
                                            // do nothing
                                        } else {
                                            rrows.push(row_idx + current_row); rcols.push(state.Active.A.unwrap()); rvals.push(-1.);
                                        }
                                        if dfa.accepting.contains(&state.Q[state.Active.T.unwrap() as usize]) {
                                            rrows.push(row_idx + current_row); rcols.push(state.Active.T.unwrap() + n_agents as i32); rvals.push(1.);
                                        }
                                        state_rewards.insert(row_idx + current_row);
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
                                        let qprime: i32 = dfa.get_transition(state.Q[state.Active.T.unwrap() as usize], w);
                                        let mut newS = state.S.to_vec();
                                        newS[state.Active.A.unwrap() as usize] = sprime.clone();
                                        let mut newQ = state.Q.to_vec();
                                        newQ[state.Active.T.unwrap() as usize] = qprime;

                                        // check if the new Q is a member of finished, or rejecting
                                        let newActive = if dfa.done.contains(&newQ[state.Active.T.unwrap() as usize])
                                            || dfa.rejecting.contains(&state.Q[state.Active.T.unwrap() as usize]) {
                                            Active { A: state.Active.A, T: None }
                                        } else {
                                            state.Active
                                        };
                                        let new_state = MAMDPState {
                                            S: newS,
                                            Q: newQ,
                                            R: state.R.to_vec(),
                                            Active: newActive,
                                        };
                                        if !visited.contains(&new_state) {
                                            visited.insert(new_state.clone());
                                            stack.push_back(new_state.clone());
                                            mamdp.insert_state(new_state.clone());
                                        }
                                        let sprime_idx = *mamdp.state_map
                                            .get(&new_state)
                                            .unwrap();

                                        if dfa.accepting.contains(&qprime) {
                                            match accepting.get_mut(&state.Active.T.unwrap()) {
                                                Some(x) => { x.push(sprime_idx); }
                                                None => { accepting.insert(state.Active.T.unwrap(), vec![sprime_idx]); }
                                            }
                                        }
                                        // add in the transition to the CxxMatrix
                                        prows.push(row_idx + current_row); pcols.push(sprime_idx as i32); pvals.push(*p);
                                        largest_row = row_idx + current_row + 1;
                                        //println!("s: {:?}, @s: {}, q: {}, a: {}, sidx':{}. s': {:?}, q': {}, p: {}, w: {}", 
                                        //   s, row_idx + action as i32, q, action, sprime_idx,sprime, qprime, p, w);
                                    }
                                }
                            }
                            Err(_)  => { }
                        }
                    } else {
                        // this is the case where the task has finished so we have to have the 
                        // option of the agent continuing is path forever in the current Q or
                        // choosing to take a new task
                        match mdp.step_(
                            state.S[state.Active.A.unwrap() as usize], 
                            (action - (n_agents * n_tasks) as i32) as u8, 
                            -1,
                            state.Active.A.unwrap()
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
            
                                    for (sprime, p, _w) in v.iter() {
                                        let mut newS = state.S.to_vec();
                                        newS[state.Active.A.unwrap() as usize] = sprime.clone();

                                        let new_state = MAMDPState {
                                            S: newS,
                                            Q: state.Q.to_vec(),
                                            R: state.R.to_vec(),
                                            Active: state.Active,
                                        };
                                        if !visited.contains(&new_state) {
                                            visited.insert(new_state.clone());
                                            stack.push_back(new_state.clone());
                                            mamdp.insert_state(new_state.clone());
                                        }
                                        let sprime_idx = *mamdp.state_map
                                            .get(&new_state)
                                            .unwrap();

                                        // add in the transition to the CxxMatrix
                                        prows.push(row_idx + current_row); pcols.push(sprime_idx as i32); pvals.push(*p);
                                        largest_row = row_idx + current_row + 1;
                                        //println!("s: {:?}, @s: {}, q: {}, a: {}, sidx':{}. s': {:?}, q': {}, p: {}, w: {}", 
                                        //   s, row_idx + action as i32, q, action, sprime_idx,sprime, qprime, p, w);
                                    }
                                }
                            }
                            Err(_)  => { }
                        }
                    }
                }
            }
        }
    }

    let pTriMatr = TriMatI::<f32, i32>::from_triplets(
        (largest_row as usize, mamdp.states.len()), prows, pcols, pvals
    );
    let rTriMatr = TriMatI::<f32, i32>::from_triplets(
        (largest_row as usize, n_objs), rrows, rcols, rvals
    );    
    //println!("enabled actions \n{:?}", enabled_actions);

    let mut vadj_pairs: Vec<i32> = vec![0; mamdp.states.len()];
    let mut venbact: Vec<i32> = vec![0; mamdp.states.len()];
    for sidx in 0..mamdp.states.len() {
        vadj_pairs[sidx] = adjusted_state_action.remove(&(sidx as i32)).unwrap();
        venbact[sidx] = enabled_actions.remove(&(sidx as i32)).unwrap();
    }
    //println!("adjusted states\n{:?}", vadj_pairs);
    mamdp.adjusted_state_act_pair = vadj_pairs;
    mamdp.enabled_actions = venbact;
    
    // compress the matrices into CSR format
    mamdp.P = pTriMatr.to_csr();
    mamdp.R = rTriMatr.to_csr();
    mamdp.accepting = accepting;
    mamdp.obj_rew = obj_rew;

    /*for state in ctmdp.states.iter() {
        println!("{:?}", state);
    }*/

    mamdp
}