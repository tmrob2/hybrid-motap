use crate::task::dfa::Mission;
use pyo3::prelude::*;
use hashbrown::{HashMap, HashSet};
use sprs::{CsMatBase, TriMatI};
use std::hash::Hash;
use crate::agent::env::Env;
use std::collections::VecDeque;

use super::general::ModelFns;


pub type SCPMState<S> = (S, i32, i32, i32);

#[pyclass]
#[derive(Clone)]
/// This is just a Python interface class which holds 
/// information about the Experiment Models
pub struct SCPM {
    pub num_agents: usize,
    pub tasks: Mission, 
    pub actions: Vec<i32>
}

#[pymethods]
impl SCPM{ 
    #[new]
    fn new(mission: Mission, num_agents: usize, actions: Vec<i32>) -> Self {
        //let num_tasks = mission.size;
        SCPM {
            num_agents,
            tasks: mission,
            actions
        }
    }
}

#[derive(Clone)]
/// This is the centralised SCPM Model
pub struct SCPMModel<S> {
    pub initial_state: SCPMState<S>, // (State, q, taskid, #, current agent)
    pub states: Vec<SCPMState<S>>,
    pub actions: Vec<i32>,
    pub P: CsMatBase<f32, i32, Vec<i32>, Vec<i32>, Vec<f32>>,
    pub R: CsMatBase<f32, i32, Vec<i32>, Vec<i32>, Vec<f32>>,
    pub adjusted_state_act_pair: Vec<i32>, 
    pub enabled_actions: Vec<i32>, 
    pub state_map: HashMap<SCPMState<S>, usize>,
    pub accepting: HashMap<i32, Vec<usize>>
}

impl<S> ModelFns<SCPMState<S>> for SCPMModel<S> {
    fn get_states(&self) -> &[SCPMState<S>] {
        &self.states
    }

    fn get_enabled_actions(&self) -> &[i32] {
        &self.enabled_actions
    }
}

impl<S> SCPMModel<S>
where S: Copy + Eq + Hash, SCPMState<S>: Eq + Hash + Clone {
    pub fn new(
        initial_state: SCPMState<S>,
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

    fn insert_state(&mut self, state: SCPMState<S>) {
        let state_idx = self.states.len();
        self.states.push(state.clone());
        self.insert_state_mapping(state, state_idx);
    }

    fn insert_state_mapping(&mut self, state: SCPMState<S>, state_idx: usize) {
        self.state_map.insert(state, state_idx);
    }
}

pub fn SCPM_bfs<S, E>(
    initial_state: SCPMState<S>,
    mdp: &E,
    n_agents: usize, 
    n_tasks: usize,
    n_objs: usize,
    model: &SCPM, 
    actions: &[i32],
    initial_agent_states: &[S]
) -> SCPMModel<S>
where S: Copy + Clone + std::fmt::Debug + Eq + Hash, E: Env<S> {
    let mut scpm = SCPMModel::new(initial_state.clone(), 
                                                actions.to_vec());
    let mut visited: HashSet<SCPMState<S>> = HashSet::new();
    let mut state_rewards: HashSet<i32> = HashSet::new();
    let mut stack: VecDeque<SCPMState<S>> = VecDeque::new();
    let mut adjusted_state_action: HashMap<i32, i32> = HashMap::new();
    let mut enabled_actions: HashMap<i32, i32> = HashMap::new();
    let mut accepting: HashMap<i32, Vec<usize>> = HashMap::new();

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
    scpm.insert_state(initial_state.clone());
    adjusted_state_action
        .insert(*scpm.state_map.get(&initial_state).unwrap() as i32, 0);
    let mut largest_row: i32 = 0;

    while !stack.is_empty() {
        let (s, q, agentid, taskid) = stack.pop_front().unwrap();
        //println!("tracking: {:?} => contains {}: {}", tracking, agentid, tracking.contains(&agentid));
        let sidx = *scpm.state_map.get(&(s, q, agentid, taskid)).unwrap();
        //println!("sidx: {}, state: ({:?},{},{},{})", sidx, s, q, agentid, taskid);
        let row_idx = *adjusted_state_action.get(&(sidx as i32)).unwrap();
        //println!("Actions: {:?}", actions);
        let mut available_actions: Vec<i32> = actions[2..].to_vec();
        let dfa = model.tasks.get_task_copy(taskid as usize);
        // We have to work our the available actions here
        // The SCPMs available actions are (b1, b2, a1, ..., ak)
        // where (a1, ..., ak) are inherited from the Agent MAS action space
        //
        // If s is initial in the MDP and q is initial in the DFA then b1 exists
        // If s is initial and qj in F or R (fin) then b2 exists
        // otherwise it is just A(s)
        //println!("available_actions before b1, b2 check: {:?}", available_actions);
        if s == mdp.get_init_state(agentid as usize) && 
            dfa.initial_state == q && agentid < (n_agents - 1) as i32{
            // add b1 to the list of possible actions
            available_actions = [&actions[0..1], &actions[2..]].concat();
            //println!("b1 available: {:?}", available_actions);
        } else if s == mdp.get_init_state(agentid as usize) && 
            dfa.check_fin(q) && taskid < (n_tasks - 1) as i32 {
            // add b2 to the list of possible actions
            available_actions = actions[1..].to_vec();
            //println!("b2 available: {:?}", available_actions);
        } // otherwise do nothing because available actions is alread set
        //println!("available actions: {:?}", available_actions);
        for action in available_actions {
            match action {
                0 => { 
                    // this is equivalent to b1
                    // What is the value of (sprime, qprime)?
                    // if b1 is taken then the agent hands over the task to the
                    // next agent
                    let sprime = (
                        initial_agent_states[(agentid + 1) as usize],
                        q,
                        agentid + 1,
                        taskid
                    );
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
                    if !visited.contains(&sprime) {
                        visited.insert(sprime);
                        stack.push_back(sprime);
                        scpm.insert_state(sprime);
                    }

                    let sprime_idx = *scpm.state_map.get(&sprime).unwrap();
                    prows.push(row_idx + current_row); pcols.push(sprime_idx as i32); 
                    pvals.push(1.0);
                    largest_row = row_idx + current_row + 1;
                    //println!("s: {:?}, @s: {}, q: {}, a: {}, sidx':{}. s': {:?}, q': {}, p: {}, w: {}", 
                    //                   s, row_idx + action as i32, q, action, sprime_idx,sprime, q, 1.0, "");
                }
                1 => { 
                    // this is equivalent to b2
                    // when the agent takes b2 this is equivalent to 
                    // getting the next task for the MAS
                    // this means that the taskid increments and the MAS hands
                    // over control to the first agent agentid = 0
                    let dfa_next = &model.tasks.get_task((taskid + 1) as usize);
                    let sprime = (
                        initial_agent_states[0],
                        dfa_next.initial_state,
                        0,
                        taskid + 1
                    );
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
                    if !visited.contains(&sprime) {
                        visited.insert(sprime);
                        stack.push_back(sprime);
                        scpm.insert_state(sprime);
                    }

                    let sprime_idx = *scpm.state_map.get(&sprime).unwrap();
                    prows.push(row_idx + current_row); pcols.push(sprime_idx as i32); 
                    pvals.push(1.0);
                    largest_row = row_idx + current_row + 1;
                }
                _ => {
                    // this is equivalent to a normal action in the Agent MDP 
                    match mdp.step_(s, (action - 2) as u8, taskid, 0) {
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
                                    if dfa.zero_rewards(q) {
                                        // do nothing
                                    } else {
                                        rrows.push(row_idx + current_row); rcols.push(agentid); rvals.push(-1.);
                                    }
                                    if dfa.accepting.contains(&q) {
                                        rrows.push(row_idx + current_row); rcols.push(taskid + n_agents as i32); rvals.push(1.);
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
                                    let qprime: i32 = dfa.get_transition(q, w);
                                    let new_state = (*sprime, qprime, agentid, taskid);
                                    if !visited.contains(&new_state) {
                                        visited.insert(new_state);
                                        stack.push_back(new_state);
                                        scpm.insert_state(new_state);
                                    }
                                    let sprime_idx = *scpm.state_map
                                        .get(&new_state)
                                        .unwrap();

                                    if dfa.accepting.contains(&qprime) {
                                        match accepting.get_mut(&taskid) {
                                            Some(x) => { x.push(sprime_idx); }
                                            None => { accepting.insert(taskid, vec![sprime_idx]); }
                                        }
                                    }
                                    // add in the transition to the CxxMatrix
                                    prows.push(row_idx + current_row); pcols.push(sprime_idx as i32); pvals.push(*p);
                                    largest_row = row_idx + current_row + 1;
                                    //println!("s: {:?}, @s: {}, q: {}, a: {}, sidx':{}. s': {:?}, q': {}, p: {}, w: {}", 
                                    //   s, row_idx + current_row as i32, q, action, sprime_idx,sprime, qprime, p, w);
                                }
                            }
                        }
                        Err(_) => {}
                    }
                }
            }
        }
    }
    let pTriMatr = TriMatI::<f32, i32>::from_triplets(
        (largest_row as usize, scpm.states.len()), prows, pcols, pvals
    );
    let rTriMatr = TriMatI::<f32, i32>::from_triplets(
        (largest_row as usize, n_objs), rrows, rcols, rvals
    );    
    //println!("enabled actions \n{:?}", enabled_actions);

    let mut vadj_pairs: Vec<i32> = vec![0; scpm.states.len()];
    let mut venbact: Vec<i32> = vec![0; scpm.states.len()];
    for sidx in 0..scpm.states.len() {
        vadj_pairs[sidx] = adjusted_state_action.remove(&(sidx as i32)).unwrap();
        venbact[sidx] = enabled_actions.remove(&(sidx as i32)).unwrap();
    }
    //println!("adjusted states\n{:?}", vadj_pairs);
    scpm.adjusted_state_act_pair = vadj_pairs;
    scpm.enabled_actions = venbact;
    
    // compress the matrices into CSR format
    scpm.P = pTriMatr.to_csr();
    scpm.R = rTriMatr.to_csr();
    scpm.accepting = accepting;

    scpm
}