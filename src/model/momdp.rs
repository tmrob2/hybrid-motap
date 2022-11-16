#![allow(non_snake_case)]
use std::collections::VecDeque;
use std::hash::Hash;
use rand::Rng;

use hashbrown::{HashMap, HashSet};
use crate::{CxxMatrixf32, reverse_key_value_pairs};
use crate::agent::env::Env;
use crate::sparse::compress;
use crate::task::dfa::DFA;

pub struct MOProductMDP<S> {
    pub initial_state: (S, i32),
    pub states: Vec<(S, i32)>,
    pub actions: Vec<i32>,
    pub P: CxxMatrixf32,
    pub R: CxxMatrixf32,
    pub agent_id: i32, 
    pub task_id: i32,
    pub adjusted_state_act_pair: Vec<i32>, 
    pub enabled_actions: Vec<i32>, 
    pub state_map: HashMap<(S, i32), usize>,
    reverse_state_map: HashMap<usize, (S, i32)>
}

pub fn choose_random_policy<S>(mdp: &MOProductMDP<S>) -> Vec<i32> {
    let mut pi: Vec<i32> = vec![-1; mdp.states.len()];
    let mut rng = rand::thread_rng();
    for s in 0..mdp.states.len() {
        let rand_act = rng
            .gen_range(0..mdp.enabled_actions[s]);
        pi[s] = rand_act;
    }
    pi
}

fn rewards(
    R: &mut CxxMatrixf32,
    q: i32, 
    sidx: usize,
    task: &DFA,
    num_agents: i32,
    agent_idx: i32,
    task_idx: i32
) {
    if task.accepting.contains(&q)
        || task.done.contains(&q)
        || task.rejecting.contains(&q) {
        // do nothing
    } else {
        R.triple_entry(sidx as i32, agent_idx, -1.);
    }
    if task.accepting.contains(&q) {
        R.triple_entry(sidx as i32, task_idx + num_agents, 1.0);
    }
}

impl<S> MOProductMDP<S>
where S: Copy + Eq + Hash {
    pub fn new(
        initial_state: (S, i32), 
        actions: Vec<i32>, 
        agent_id: i32, 
        task_id: i32
    ) -> Self {
        MOProductMDP { 
            initial_state, 
            states: Vec::new(), 
            actions: actions.to_vec(), 
            P: CxxMatrixf32::new(), 
            R: CxxMatrixf32::new(), 
            agent_id, 
            task_id, 
            adjusted_state_act_pair: Vec::new(),
            enabled_actions: Vec::new(),
            state_map: HashMap::new(),
            reverse_state_map: HashMap::new() 
        }
    }

    fn insert_state(&mut self, state: (S, i32)) {
        let state_idx = self.states.len();
        self.states.push(state);
        self.insert_state_mapping(state, state_idx);
    }

    fn insert_state_mapping(&mut self, state: (S, i32), state_idx: usize) {
        self.state_map.insert(state, state_idx);
    }
}

pub fn product_mdp_bfs<S, E>(
    initial_state: (S, i32),
    mdp: E,
    task: DFA,
    agent_id: i32, 
    task_id: i32,
    n_agents: usize,
    n_objs: usize,
    actions: &[i32]
) -> MOProductMDP<S> 
where S: Copy + std::fmt::Debug + Eq + Hash, E: Env<S> {
    let mut pmdp = 
        MOProductMDP::new(initial_state, actions.to_vec(), agent_id, task_id);

    let mut visited: HashSet<(S, i32)> = HashSet::new();
    let mut state_rewards: HashSet<i32> = HashSet::new();
    let mut stack: VecDeque<(S, i32)> = VecDeque::new();
    let mut adjusted_state_action: HashMap<i32, i32> = HashMap::new();
    let mut enabled_actions: HashMap<i32, i32> = HashMap::new();

    stack.push_back(initial_state);
    visited.insert(initial_state);
    // actions enabled
    pmdp.insert_state(initial_state);
    adjusted_state_action
        .insert(*pmdp.state_map.get(&initial_state).unwrap() as i32, 0);
    let mut largest_row: i32 = 0;

    while !stack.is_empty() {
    //for _ in 0..5 {
        let (s, q) = stack.pop_front().unwrap();
        let sidx = *pmdp.state_map.get(&(s, q)).unwrap();
        let row_idx = *adjusted_state_action.get(&(sidx as i32)).unwrap();
        for action in 0..actions.len() {
            match mdp.step_(s, action as u8) {
                Ok(v) => {
                    if !v.is_empty() {
                        match enabled_actions.get_mut(&(sidx as i32)) {
                            Some(x) => { *x += 1;}
                            None => { enabled_actions.insert(sidx as i32, 1); }
                        }
                        if !state_rewards.contains(&(row_idx + action as i32)) {
                            //println!("s: {:?}, @s: {}, q: {}, a: {}", s, row_idx as usize + action, q, action);
                            rewards(
                                &mut pmdp.R, 
                                q, 
                                row_idx as usize + action, 
                                &task, 
                                n_agents as i32, 
                                agent_id, 
                                task_id
                            );
                            state_rewards.insert(row_idx + action as i32);
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
                            let qprime: i32 = task.get_transition(q, w);
                            if !visited.contains(&(*sprime, qprime)) {
                                visited.insert((*sprime, qprime));
                                stack.push_back((*sprime, qprime));
                                pmdp.insert_state((*sprime, qprime));
                            }
                            let sprime_idx = *pmdp.state_map
                                .get(&(*sprime, qprime))
                                .unwrap();
                            // add in the transition to the CxxMatrix
                            pmdp.P.triple_entry(
                                row_idx + action as i32, 
                                sprime_idx as i32, 
                                *p
                            );
                            largest_row = row_idx + action as i32 + 1;
                            //println!("s: {:?}, @s: {}, q: {}, a: {}, sidx':{}. s': {:?}, q': {}, p: {}, w: {}", 
                            //   s, row_idx + action as i32, q, action, sprime_idx,sprime, qprime, p, w);
                        }
                    }
                }
                Err(_) => {}
            }
        }
    }
    pmdp.P.m = largest_row;
    pmdp.P.n = pmdp.states.len() as i32;
    pmdp.P.nzmax = pmdp.P.nz + 1;
    pmdp.R.m = largest_row;
    pmdp.R.n = n_objs as i32;
    pmdp.R.nzmax = pmdp.R.nz + 1;

    //println!("enabled actions \n{:?}", enabled_actions);

    let mut vadj_pairs: Vec<i32> = vec![0; pmdp.P.n as usize];
    let mut venbact: Vec<i32> = vec![0; pmdp.P.n as usize];
    for sidx in 0..pmdp.P.n as usize {
        vadj_pairs[sidx] = adjusted_state_action.remove(&(sidx as i32)).unwrap();
        venbact[sidx] = enabled_actions.remove(&(sidx as i32)).unwrap();
    }
    //println!("adjusted states\n{:?}", vadj_pairs);
    pmdp.adjusted_state_act_pair = vadj_pairs;
    pmdp.enabled_actions = venbact;
    
    // compress the matrices into CSR format
    pmdp.P = compress::compress(pmdp.P);
    pmdp.R = compress::compress(pmdp.R);
    pmdp
}