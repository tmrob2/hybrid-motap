use hashbrown::{HashSet, HashMap};
use super::{general::ModelFns, scpm::SCPM};
use sprs::{CsMatBase, TriMatI};
use std::hash::Hash;
use crate::agent::env::Env;
use std::collections::VecDeque;

pub type CTState<S> = (S, i32, i32, Vec<i32>, i32);

pub struct CTMDP<S> {
    pub initial_state: CTState<S>, // (State, q, taskid, #, current agent)
    pub states: Vec<CTState<S>>,
    pub actions: Vec<i32>,
    pub P: CsMatBase<f32, i32, Vec<i32>, Vec<i32>, Vec<f32>>,
    pub R: CsMatBase<f32, i32, Vec<i32>, Vec<i32>, Vec<f32>>,
    pub adjusted_state_act_pair: Vec<i32>, 
    pub enabled_actions: Vec<i32>, 
    pub state_map: HashMap<CTState<S>, usize>,
    pub accepting: HashMap<i32, Vec<usize>>
}

impl<S> ModelFns<CTState<S>> for CTMDP<S> {
    fn get_states(&self) -> &[CTState<S>] {
        &self.states
    }

    fn get_enabled_actions(&self) -> &[i32] {
        &self.enabled_actions
    }
}

impl<S> CTMDP<S>
where S: Copy + Eq + Hash, CTState<S>: Eq + Hash + Clone {
    pub fn new(
        initial_state: CTState<S>,
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

    fn insert_state(&mut self, state: CTState<S>) {
        let state_idx = self.states.len();
        self.states.push(state.clone());
        self.insert_state_mapping(state, state_idx);
    }

    fn insert_state_mapping(&mut self, state: CTState<S>, state_idx: usize) {
        self.state_map.insert(state, state_idx);
    }
}

pub fn CTMDP_bfs<S, E>(
    initial_state: CTState<S>,
    mdp: &E,
    n_agents: usize,
    n_objs: usize,
    model: &SCPM,
    actions: &[i32],
    initial_agent_states: &[S]
) -> CTMDP<S>
    where S: Copy + Clone + std::fmt::Debug + Eq + Hash, E: Env<S> {
    let mut ctmdp = CTMDP::new(initial_state.clone(), actions.to_vec());

    let mut visited: HashSet<CTState<S>> = HashSet::new();
    let mut state_rewards: HashSet<i32> = HashSet::new();
    let mut stack: VecDeque<CTState<S>> = VecDeque::new();
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
    ctmdp.insert_state(initial_state.clone());
    adjusted_state_action
        .insert(*ctmdp.state_map.get(&initial_state).unwrap() as i32, 0);
    let mut largest_row: i32 = 0;
    
    //for _ in 0..10 {
    while !stack.is_empty() {
        let (s, q, taskid, tracking, agentid) = stack.pop_front().unwrap();
        //println!("tracking: {:?} => contains {}: {}", tracking, agentid, tracking.contains(&agentid));
        let sidx = *ctmdp.state_map.get(&(s, q, taskid, tracking.to_vec(), agentid)).unwrap();
        //println!("sidx: {}, state: ({:?},{},{},{:?},{})", sidx, s, q, taskid, tracking, agentid);
        let row_idx = *adjusted_state_action.get(&(sidx as i32)).unwrap();
        let mut available_actions: Vec<i32> = actions[3..].to_vec();
        let dfa = model.tasks.get_task_copy(taskid as usize);
        if tracking.contains(&agentid) {
            if dfa.done.contains(&q) || dfa.rejecting.contains(&q) {
                // the action b3 is available
                available_actions = [&actions[2..3], &actions[3..]].concat();
            } else if s == initial_agent_states[agentid as usize] && q == dfa.initial_state {
                available_actions = [&actions[1..2], &actions[3..]].concat();
            }
        } else {
            if !tracking.contains(&agentid){
                available_actions = actions[..2].to_vec();
            }
        }
        //println!("available actions: {:?}", available_actions);

        for action in available_actions {
            match action {
                0 => {
                    // this is equivalent to b1
                    // P(s, q, #, b1, s, q, # U i)
                    // requirement to add a new state
                    let mut new_tracking = tracking.to_vec();
                    new_tracking.push(agentid);
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
                    if !visited.contains(&(s, q, taskid, new_tracking.clone(), agentid)) {
                        visited.insert((s, q, taskid, new_tracking.clone(), agentid));
                        stack.push_back((s, q, taskid, new_tracking.clone(), agentid));
                        ctmdp.insert_state((s, q, taskid, new_tracking.clone(), agentid));
                    }
                    let sprime_idx = *ctmdp.state_map
                        .get(&(s, q, taskid, new_tracking, agentid))
                        .unwrap();
                    // add in the transition to the CxxMatrix
                    //println!("s: {:?}, @s: {}, q: {}, agent: {}, task: {}, a: {}, sidx':{}. s': {:?}, q': {}, agent': {}, task': {}", 
                    //        s, row_idx + action as i32, q, agentid, taskid, action, sprime_idx, s, q, agentid, taskid);
                    prows.push(row_idx + current_row); pcols.push(sprime_idx as i32); pvals.push(1.0);
                    largest_row = row_idx + current_row + 1;
                }
                1 => { 
                    // this is equivalent to b2
                    // P(s, q, #, b2, s_{next i'>i}, q, #)
                    // requirement to add a new state
                    let agents = HashSet::from_iter(agentid + 1..n_agents as i32);
                    let alloc_agents: HashSet<_> = HashSet::from_iter(tracking.to_vec().into_iter());
                    //println!("current agent: {}, agents: {:?}, #: {:?}: I \\ #: {:?}", agentid, agents, alloc_agents, agents.difference(&alloc_agents));
                    let next_agent = agents.difference(&alloc_agents).min();
                    //println!("next agent: {:?}", next_agent);
                    if next_agent.is_some() {
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
                        if !visited.contains(&(s, q, taskid, tracking.clone(), *next_agent.unwrap())) {
                            visited.insert((s, q, taskid, tracking.clone(), *next_agent.unwrap()));
                            stack.push_back((s, q, taskid, tracking.clone(), *next_agent.unwrap()));
                            ctmdp.insert_state((s, q, taskid, tracking.clone(), *next_agent.unwrap()));
                        }
                        //println!("stack:{:?}", stack);
                        let sprime_idx = *ctmdp.state_map
                            .get(&(s, q, taskid, tracking.clone(), *next_agent.unwrap()))
                            .unwrap();
                        // add in the transition to the CxxMatrix
                        //println!("s: {:?}, @s: {}, q: {}, agent: {}, task: {}, a: {}, sidx':{}. s': {:?}, q': {}, agent': {}, task': {}", 
                        //    s, row_idx + action as i32, q, agentid, taskid, action, sprime_idx, s, q, *next_agent.unwrap(), taskid);
                        prows.push(row_idx + current_row); pcols.push(sprime_idx as i32); pvals.push(1.0);
                        largest_row = row_idx + current_row + 1;
                    }
                }
                2 => {
                    // this is equivalent to b3
                    // P(s, q, #, b3, s_{i not in #}, q_j + 1, #)
                    // possible requirement to add a new state
                    if (taskid as usize) < n_objs - n_agents - 1 {
                        let agents = HashSet::from_iter(0..n_agents as i32);
                        let alloc_agents: HashSet<_> = HashSet::from_iter(tracking.to_vec().into_iter());
                        let next_agent = agents.difference(&alloc_agents).min();
                        //println!("next agent: {:?}", next_agent);
                        if next_agent.is_some() {
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
                            let snew_0 = initial_agent_states[*next_agent.unwrap() as usize];
                            let next_dfa = model.tasks.get_task_copy((taskid + 1) as usize);
                            let qnew_0 = next_dfa.initial_state;
                            if !visited.contains(&(snew_0, qnew_0, taskid + 1, tracking.clone(), *next_agent.unwrap())) {
                                visited.insert((snew_0, qnew_0, taskid + 1, tracking.clone(), *next_agent.unwrap()));
                                stack.push_back((snew_0, qnew_0, taskid + 1, tracking.clone(), *next_agent.unwrap()));
                                ctmdp.insert_state((snew_0, qnew_0, taskid + 1, tracking.clone(), *next_agent.unwrap()));
                            }
                            let sprime_idx = *ctmdp.state_map
                                .get(&(snew_0, qnew_0, taskid + 1, tracking.clone(), *next_agent.unwrap())
                                )
                                .unwrap();
                            // add in the transition to the CxxMatrix
                            //println!("s: {:?}, @s: {}, q: {}, agent: {}, task: {}, a: {}, sidx':{}. s': {:?}, q': {}, agent': {}, task': {}", 
                            //           s, row_idx + action as i32, q, agentid, taskid, action, sprime_idx,snew_0, qnew_0, *next_agent.unwrap(), taskid + 1);
                            prows.push(row_idx + current_row); pcols.push(sprime_idx as i32); pvals.push(1.0);
                            largest_row = row_idx + current_row + 1;
                        }
                    }
                }
                _ => {
                    // these actions are equivalent to the warehouse actions
                    // i.e. implement env.step()
                    // have to subtract 3 from the base action as we used the first three spots 
                    // for the actions b1, b2, b3
                    match mdp.step_(s, (action - 3) as u8, taskid, 0) {
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
                                    if dfa.accepting.contains(&q)
                                        || dfa.done.contains(&q)
                                        || dfa.rejecting.contains(&q) {
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
                                    if !visited.contains(&(*sprime, qprime, taskid, tracking.to_vec(), agentid)) {
                                        visited.insert((*sprime, qprime, taskid, tracking.to_vec(), agentid));
                                        stack.push_back((*sprime, qprime, taskid, tracking.to_vec(), agentid));
                                        ctmdp.insert_state((*sprime, qprime, taskid, tracking.to_vec(), agentid));
                                    }
                                    let sprime_idx = *ctmdp.state_map
                                        .get(&(*sprime, qprime, taskid, tracking.to_vec(), agentid))
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
                                    //   s, row_idx + action as i32, q, action, sprime_idx,sprime, qprime, p, w);
                                }
                            }
                        }
                        Err(_) => {}
                    }
                }
            }
        }
    }

    //println!("{:?}", accepting);

    let pTriMatr = TriMatI::<f32, i32>::from_triplets(
        (largest_row as usize, ctmdp.states.len()), prows, pcols, pvals
    );
    let rTriMatr = TriMatI::<f32, i32>::from_triplets(
        (largest_row as usize, n_objs), rrows, rcols, rvals
    );    
    //println!("enabled actions \n{:?}", enabled_actions);

    let mut vadj_pairs: Vec<i32> = vec![0; ctmdp.states.len()];
    let mut venbact: Vec<i32> = vec![0; ctmdp.states.len()];
    for sidx in 0..ctmdp.states.len() {
        vadj_pairs[sidx] = adjusted_state_action.remove(&(sidx as i32)).unwrap();
        venbact[sidx] = enabled_actions.remove(&(sidx as i32)).unwrap();
    }
    //println!("adjusted states\n{:?}", vadj_pairs);
    ctmdp.adjusted_state_act_pair = vadj_pairs;
    ctmdp.enabled_actions = venbact;
    
    // compress the matrices into CSR format
    ctmdp.P = pTriMatr.to_csr();
    ctmdp.R = rTriMatr.to_csr();
    ctmdp.accepting = accepting;

    /*for state in ctmdp.states.iter() {
        println!("{:?}", state);
    }*/

    ctmdp
}