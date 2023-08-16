use pyo3::prelude::*;
use crate::agent::env::Env;
use crate::model::mostapu::MOSTAPU_bfs;
use crate::solvers::motap::{motap_cpu_only_solver, mamdp_cpu_solver, scpm_solver, stapu_solver};
use std::time::Instant;
use rayon::prelude::*;
use crate::algorithms::synth::{mamdp_scheduler_synthesis, motap_scheduler_synthesis};
use crate::model::mamdp::{MOMAMDP_bfs, MAMDPState};
use crate::model::mostapu::StapuState;
use crate::{product, debug_level, motap_prism_file_generator, generic_motap_mostapu_cpu};
use crate::model::momdp::product_mdp_bfs;
use crate::model::scpm::{SCPM, SCPMState, SCPM_bfs};
use crate::model::momdp::MOProductMDP;
use crate::Debug;

type State = i32;

#[pyclass]
#[derive(Clone)]
pub struct Model {
    pub states: Vec<i32>,
    pub initial_state: i32, 
    pub type_name: EnvType,
    pub p1: f32
}

#[pymethods]
impl Model {
    #[new]
    pub fn new(
        mtype: String,  
        states: Vec<i32>, 
        initial_state: i32, 
        p: f32
    ) -> Self {
        let mut model = Model {
            states,
            initial_state,
            type_name: EnvType::MS1,
            p1: p
        };
        match &*mtype {
            "MS1" => { model.type_name = EnvType::MS1; }
            "MS2" => { model.type_name = EnvType::MS2; }
            "MS3" => { model.type_name = EnvType::MS3; }
            _ => { }
        }
        model
    }
}

#[pyclass]
pub struct MAS {
    pub agents: Vec<Model>,
    pub size: usize,
    pub ordering: Vec<usize>,
    pub action_space: Vec<i32>
}

#[derive(Clone)]
pub enum EnvType {
    MS1,
    MS2,
    MS3
}

#[pymethods]
impl MAS {
    #[new]
    pub fn new(action_space: Vec<i32>) -> Self {
        MAS {
            agents: Vec::new(),
            size: 0,
            ordering: Vec::new(),
            action_space
        }
    }

    pub fn create_order(&mut self, order: Vec<usize>) {
        self.ordering = order;
    }

    pub fn add_environment(&mut self, env: Model) {
        self.agents.push(env);
    }

    pub fn get_init_states(&self) -> Vec<i32> {
        let mut init_states: Vec<i32> = Vec::new();
        for agent in self.agents.iter() {
            init_states.push(agent.initial_state);
        }
        init_states
    }
}

/*
-------------------------------------------------------------------
|                     MDP TYPE DEFINITIONS                        |
-------------------------------------------------------------------
*/

impl Env<State> for MAS {
    fn step_(&self, s: State, action: u8, _task_id: i32, agent_id: i32) -> Result<Vec<(State, f32, String)>, String> {
        let pfail = self.agents[self.ordering[agent_id as usize]].p1;
        let transition: Result<Vec<(State, f32, String)>, String> = match &self.agents[self.ordering[agent_id as usize]].type_name {
            EnvType::MS1 => { 
                match s {
                    0 => {
                        // return the transition for state 0
                        match action {
                            0 => {Ok(vec![(0, pfail, "ini".to_string()), (1, 1.-pfail, "i".to_string())])}
                            _ => {Err("Action not-implemented".to_string())}
                        }
                    }
                    1 => {
                        // return the transition for state 1
                        match action {
                            0 => {Ok(vec![(2, 1.0, "r".to_string())])}
                            _ => {Err("Action not-implemented".to_string())}
                        }
                    }
                    2 => { 
                        match action {
                            0 => {Ok(vec![(3, 1.-pfail, "s".to_string()), (4, pfail, "e".to_string())])}
                            1 => {Ok(vec![(4, 1.0, "e".to_string())])}
                            _ => {Err("Action not-implemented".to_string())}
                        }
                    }
                    3 => {
                        match action {
                            0 => {Ok(vec![(2, 1.0, "r".to_string())])}
                            _ => {Err("Action not-implemented".to_string())}
                        }
                    }
                    4 => {
                        match action {
                            0 => {Ok(vec![(0, 1.0, "ini".to_string())])}
                            _ => {Err("Action not-implemented".to_string())}
                        }
                    }
                    _ => {
                        // Not implemented error
                        Err("Not-implemented".to_string())
                    }
                }
            }
            EnvType::MS2 => { 
                match s {
                    0 => {
                        // return the transition for state 0
                        match action {
                            0 => {Ok(vec![(0, 0.01, "ini".to_string()), (1, 0.99, "i".to_string())])}
                            _ => {Err("Action not-implemented".to_string())}
                        }
                    }
                    1 => {
                        // return the transition for state 1
                        match action {
                            0 => {Ok(vec![(2, 1.0, "r".to_string())])}
                            _ => {Err("Action not-implemented".to_string())}
                        }
                    }
                    2 => { 
                        match action {
                            0 => {Ok(vec![(3, (1. - self.agents[self.ordering[agent_id as usize]].p1) / 2.0, "s".to_string()), 
                                          (5, self.agents[self.ordering[agent_id as usize]].p1, "e".to_string()), 
                                          (4, (1.0 - self.agents[self.ordering[agent_id as usize]].p1) / 2.0, "s".to_string())])}
                            1 => {Ok(vec![(5, 1.0, "e".to_string())])}
                            _ => {Err("Action not-implemented".to_string())}
                        }
                    }
                    3 => {
                        match action {
                            0 => {Ok(vec![(2, 1.0, "r".to_string())])}
                            _ => {Err("Action not-implemented".to_string())}
                        }
                    }
                    4 => {
                        match action {
                            0 => {Ok(vec![(5, 0.5, "e".to_string()), 
                                          (4, 0.1, "s".to_string()), 
                                          (2, 1.0 - 0.6, "r".to_string())])}
                            _ => {Err("Action not-implemented".to_string())}
                        }
                    }
                    5 => {
                        match action {
                            0 => { Ok(vec![(0, 1.0, "ini".to_string())])}
                            _ => {Err("Action not-implemented".to_string())}
                        }
                    }
                    _ => {
                        // Not implemented error
                        Err("Not-implemented".to_string())
                    }
                }
            }
            EnvType::MS3 => {
                match s {
                    0 => {
                        // return the transition for state 0
                        match action {
                            0 => {Ok(vec![(0, 0.01, "ini".to_string()), (1, 0.99, "i".to_string())])}
                            _ => {Err("Action not-implemented".to_string())}
                        }
                    }
                    1 => {
                        // return the transition for state 1
                        match action {
                            0 => {Ok(vec![(2, 1.0, "r".to_string())])}
                            _ => {Err("Action not-implemented".to_string())}
                        }
                    }
                    2 => { 
                        match action {
                            0 => {Ok(vec![(3, (1. - self.agents[self.ordering[agent_id as usize]].p1 ) / 2.0, "s".to_string()), 
                                          (5, self.agents[self.ordering[agent_id as usize]].p1 / 2., "e".to_string()),  
                                          (6, self.agents[self.ordering[agent_id as usize]].p1 / 2., "e".to_string()), 
                                          (4, (1.0 - self.agents[self.ordering[agent_id as usize]].p1) / 2.0, "s".to_string())])}
                            1 => {Ok(vec![(5, 1.0, "e".to_string())])}
                            _ => {Err("Action not-implemented".to_string())}
                        }
                    }
                    3 => {
                        match action {
                            0 => {Ok(vec![(2, 1.0, "r".to_string())])}
                            _ => {Err("Action not-implemented".to_string())}
                        }
                    }
                    4 => {
                        match action {
                            0 => {Ok(vec![(5, 0.3, "e".to_string()),
                                          (6, 0.3, "e".to_string()), 
                                          (4, 0.1, "s".to_string()), 
                                          (2, 0.3, "r".to_string())])}
                            _ => {Err("Action not-implemented".to_string())}
                        }
                    }
                    5 => {
                        match action {
                            0 => { Ok(vec![(0, 1.0, "ini".to_string())])}
                            _ => {Err("Action not-implemented".to_string())}
                        }
                    }
                    6 => {
                        match action {
                            0 => { Ok(vec![(0, 1.0, "ini".to_string())])}
                            _ => {Err("Action not-implemented".to_string())}
                        }
                    }
                    _ => {
                        // Not implemented error
                        Err("Not-implemented".to_string())
                    }
        
                }
            }
        };
        transition
    }

    fn get_init_state(&self, _agent: usize) -> State {
        0
    }

    fn set_task(&mut self, _task_id: usize) {
    }

    fn get_action_space(&self) -> Vec<i32> {
        self.action_space.to_vec()
    }

    fn get_states_len(&self, agent_id: i32) -> usize {
        self.agents[self.ordering[agent_id as usize]].states.len()
    }
}


/*
-------------------------------------------------------------------
|                     PYTHON IMPLEMENTATIONS                      |
-------------------------------------------------------------------
*/

#[pyfunction]
pub fn motap_mamdp_synth(
    model: &SCPM,
    envs: &MAS,
    w: Vec<f32>,
    target: Vec<f32>,
    epsilon1: f32,
    epsilon2: f32,
    debug: i32,
    max_iter: usize,
    max_unstable: i32,
    constraint_threshold: Vec<f32>
) where MAS: Env<State> { 
    generic_motap_synth(model, envs, w, target, epsilon1, epsilon2, 
        debug, max_iter, max_unstable, constraint_threshold);
}

#[pyfunction]
pub fn motap_synth(
    model: &SCPM,
    env: &MAS,
    w: Vec<f32>,
    target: Vec<f32>,
    epsilon1: f32,
    epsilon2: f32,
    debug: i32,
    max_iter: usize,
    max_unstable: i32,
    constraint_threshold: Vec<f32>
) { 
    generic_motap_synthesis(model, env, w, target, epsilon1, epsilon2, 
        debug, max_iter, max_unstable, constraint_threshold);
}

#[pyfunction]
pub fn motap_test_cpu(
    model: &SCPM,
    env: &MAS,
    w: Vec<f32>,
    epsilon1: f32,
    debug: i32,
    max_iter: usize,
    max_unstable: i32,
) { 
    generic_motap_msg_test_cpu(model, env, w, epsilon1, debug, max_iter, max_unstable);
}

#[pyfunction]
pub fn motap_mamdp_test_cpu(
    model: &SCPM,
    env: &MAS,
    w: Vec<f32>,
    epsilon1: f32,
    debug: i32,
    max_iter: usize,
    max_unstable: i32,
    get_prism: bool
) { 
    generic_motap_mamdp_cpu(model, env, w, epsilon1, debug, max_iter, max_unstable, get_prism);
}

#[pyfunction]
pub fn motap_scpm_test_cpu(
    model: &SCPM,
    env: &MAS,
    w: Vec<f32>,
    epsilon1: f32,
    debug: i32,
    max_iter: usize,
    max_unstable: i32,
    initial_agent_states: Vec<State>
) { 
    generic_motap_scpm_cpu(model, env, w, epsilon1, debug, max_iter, max_unstable, &initial_agent_states[..]);
}

#[pyfunction]
pub fn motap_stapu_test_cpu(
    model: &SCPM,
    env: &MAS,
    w: Vec<f32>,
    epsilon1: f32,
    debug: i32,
    max_iter: usize,
    max_unstable: i32,
    initial_agent_states: Vec<State>
) { 
    generic_motap_mostapu_cpu(model, env, w, epsilon1, debug, max_iter, max_unstable, &initial_agent_states[..]);
}
/*
-------------------------------------------------------------------
|                     GENERIC IMPLEMENTATIONS                     |
-------------------------------------------------------------------
*/

pub fn generic_motap_synth<E>(
    model: &SCPM,
    envs: &E,
    w: Vec<f32>,
    target: Vec<f32>,
    epsilon1: f32,
    epsilon2: f32,
    debug: i32,
    max_iter: usize,
    max_unstable: i32,
    constraint_threshold: Vec<f32>
) where E: Env<State> {
    println!("--------------------------");
    println!("MAMDP MOTAP SYNTH A:{} T:{}", model.num_agents, model.tasks.size);
    println!("--------------------------");
    let t1 = Instant::now();
    let dbug = debug_level(debug);
    //model.construct_products(&mut mdp);

    let initial_state: MAMDPState<State> = MAMDPState {
        S: (0..model.num_agents).map(|a| envs.get_init_state(a)).collect(),
        Q: vec![0; model.tasks.size],
        R: (0..model.tasks.size as i32).collect(),
        Active: crate::model::mamdp::Active { A: None, T: None }
    };
    let alloc_acts: Vec<i32> = (0..(model.tasks.size*model.num_agents) as i32).collect();
    let agent_acts: Vec<i32> = model.actions.iter()
        .map(|a| a + (model.num_agents * model.tasks.size) as i32)
        .collect();
    let actions = [&alloc_acts[..], &agent_acts[..]].concat();

    let mamdp = MOMAMDP_bfs(
        initial_state, 
        envs, 
        model.tasks.size, 
        model.num_agents, 
        model.tasks.size + model.num_agents, 
        model, 
        &actions
    );
    match dbug {
        Debug::None => { }
        _ => { 
            println!("Time to create mamdp model: {}\n|S|: {},\n|P|: {}", 
                t1.elapsed().as_secs_f32(),
                mamdp.states.len(),
                mamdp.P.nnz()
            );
        }
    }

    let t2 = Instant::now();

    let res = mamdp_scheduler_synthesis(
        &mamdp, model.num_agents, model.tasks.size, w, &target, 
        epsilon1, epsilon2, dbug, max_iter, max_unstable,
        &constraint_threshold
    );

    match dbug {
        Debug::None => { }
        _ => { 
            match res {
                Ok((X, rt, l, v)) => {
                    let avg_rt = rt.iter().fold(0., |acc, x| acc + x)/ rt.len() as f32;
                    if v.is_some() {
                        /*if dbug == Debug::Verbose1 {
                            println!("v: {:?}", v.as_ref().unwrap());
                        }*/
                        let v_ = v.unwrap();
                        // compute the total expected cost for the MAS
                        let mut cost: f32 = 0.;
                        for k in 0..v_.len() {
                            let r = X.get(&k).unwrap();
                            cost += r[..model.num_agents].iter()
                                .map(|x| v_[k] * *x)
                                .fold(0.0, |acc, y| acc + y);
                        }
                        println!("MAS cost: {}", cost);
                    } else {
                        println!("No randomised allocation function meets target threshold!");
                    }
                    println!("Average loop run time: {}", avg_rt);
                    println!("Number of iterations: {}", l); 
                    println!("Total model checking time: {}", avg_rt * l as f32);
                    println!("Synthesis total run time: {}", t2.elapsed().as_secs_f32());
                } 
                Err(e) => {
                    println!("Error: {:?}", e);
                }
            }
            
        }
    }
}

pub fn generic_motap_mamdp_cpu<E>(
    model: &SCPM,
    env: &E,
    w: Vec<f32>,
    eps: f32,
    debug: i32,
    max_iter: usize,
    max_unstable: i32,
    prism_file: bool
) where E: Env<State> {
println!("--------------------------");
println!("   MOTAP MAMDP CPU TEST   ");
println!("--------------------------");

    let t1 = Instant::now();
    let dbug = debug_level(debug);
    let initial_state: MAMDPState<State> = MAMDPState {
        S: (0..model.num_agents).map(|a| env.get_init_state(a)).collect(),
        Q: vec![0; model.tasks.size],
        R: (0..model.tasks.size as i32).collect(),
        Active: crate::model::mamdp::Active { A: None, T: None }
    };
    let alloc_acts: Vec<i32> = (0..(model.tasks.size*model.num_agents) as i32).collect();
    let agent_acts: Vec<i32> = model.actions.iter()
        .map(|a| a + (model.num_agents * model.tasks.size) as i32)
        .collect();
    let actions = [&alloc_acts[..], &agent_acts[..]].concat();
    let mamdp = MOMAMDP_bfs(
        initial_state, 
        env, 
        model.tasks.size, 
        model.num_agents, 
        model.tasks.size + model.num_agents, 
        model, 
        &actions
    );
    let mut acc: Vec<usize> = Vec::new();
    for state in mamdp.states.iter() {
        match state.Active.T {
            Some(t) => {
                let dfa = model.tasks.get_task(t as usize);
                if dfa.accepting.contains(&state.Q[t as usize]) {
                    acc.push(*mamdp.state_map.get(state).unwrap());
                }
            }
            None => { }
        }
    }
    if prism_file {
        motap_prism_file_generator(
            mamdp.P.view(), 
            mamdp.states.len(), 
            &mamdp.adjusted_state_act_pair, 
            &mamdp.enabled_actions, 
            *mamdp.state_map.get(&mamdp.initial_state).unwrap(),
            &acc[..], 
            &mamdp.obj_rew,
            model.num_agents as i32,
            model.tasks.size as i32 
        ).unwrap();
    }
    match dbug {
        Debug::None => { }
        _ => { 
            println!("Time to create mamdp model: {}\n|S|: {},\n|P|: {}", 
                t1.elapsed().as_secs_f32(),
                mamdp.states.len(),
                mamdp.P.nnz()
            );
        }
    }

    let result = mamdp_cpu_solver(&mamdp, model.num_agents, 
        model.tasks.size, &w, eps, dbug, max_iter, max_unstable);
    match dbug {
        Debug::None => { }
        _ => { 
            println!("Total runtime {}", t1.elapsed().as_secs_f32());
            println!("result: {:?}", result);
        }
    }
}

pub fn generic_motap_scpm_cpu<E>(
    model: &SCPM,
    env: &E,
    w: Vec<f32>,
    eps: f32,
    debug: i32,
    max_iter: usize,
    max_unstable: i32,
    initial_agent_states: &[State]
) where E: Env<State> {
println!("--------------------------");
println!("   MOTAP SCPM CPU TEST   ");
println!("--------------------------");

    let t1 = Instant::now();
    let dbug = debug_level(debug);
    let initial_state: SCPMState<State> = (
        env.get_init_state(0),
        model.tasks.get_task(0).initial_state,
        0,
        0
    );
    let alloc_acts: Vec<i32> = (0..2).collect();
    let agent_acts: Vec<i32> = model.actions.iter()
        .map(|a| a + 2)
        .collect();
    let actions = [&alloc_acts[..], &agent_acts[..]].concat();
    let scpm = SCPM_bfs(
        initial_state, 
        env, 
        model.num_agents, 
        model.tasks.size, 
        model.tasks.size + model.num_agents, 
        model, 
        &actions,
        &initial_agent_states
    );
    let mut acc: Vec<usize> = Vec::new();
    for state in scpm.states.iter() {
        let dfa = model.tasks.get_task(state.3 as usize);
        if dfa.accepting.contains(&state.1) {
            acc.push(*scpm.state_map.get(state).unwrap());
        }
    }
    //
    match dbug {
        Debug::None => { }
        _ => { 
            println!("Time to create mamdp model: {}\n|S|: {},\n|P|: {}", 
                t1.elapsed().as_secs_f32(),
                scpm.states.len(),
                scpm.P.nnz()
            );
        }
    }

    let result = scpm_solver(&scpm, model.num_agents, 
        model.tasks.size, &w, eps, dbug, max_iter, max_unstable);
    match dbug {
        Debug::None => { }
        _ => { 
            println!("Total runtime {}", t1.elapsed().as_secs_f32());
            println!("result: {:?}", result);
        }
    }
}


pub fn generic_motap_synthesis<E>(
    model: &SCPM,
    env: &E,
    w: Vec<f32>,
    target: Vec<f32>,
    epsilon1: f32,
    epsilon2: f32,
    debug: i32,
    max_iter: usize,
    max_unstable: i32,
    constraint_threshold: Vec<f32>
) where E: Env<State> + Sync{
    println!("--------------------------");
    println!("   MOTAP SYNTH A:{} T:{}", model.num_agents, model.tasks.size);
    println!("--------------------------");
    let t1 = Instant::now();
    let dbug = debug_level(debug);
    //model.construct_products(&mut mdp);
    let pairs = 
        product(0..model.num_agents, 0..model.tasks.size);
    let models_ls: Vec<MOProductMDP<State>> = pairs.into_par_iter().map(|(a, t)| {
        let pmdp = product_mdp_bfs(
            (env.get_init_state(a), 0), 
            env,
            &model.tasks.get_task(t), 
            a as i32, 
            t as i32, 
            model.num_agents, 
            model.num_agents + model.tasks.size,
            &model.actions
        );
        pmdp
    }).collect(); 
    match dbug {
        Debug::None => { }
        _ => { 
            println!("Time to create {} models: {:?}\n|S|: {},\n|P|: {}", 
                models_ls.len(), t1.elapsed().as_secs_f32(),
                models_ls.iter().fold(0, |acc, m| acc + m.states.len()),
                models_ls.iter().fold(0, |acc, m| acc + m.P.nnz())
            );
        }
    }

    let t2 = Instant::now();

    let res = motap_scheduler_synthesis(
        models_ls, model.num_agents, model.tasks.size, w, &target, 
        epsilon1, epsilon2, dbug, max_iter, max_unstable,
        &constraint_threshold
    );

    match dbug {
        Debug::None => { }
        _ => { 
            match res {
                Ok((X, rt, l, v)) => {
                    let avg_rt = rt.iter().fold(0., |acc, x| acc + x)/ rt.len() as f32;
                    if v.is_some() {
                        /*if dbug == Debug::Verbose1 {
                            println!("v: {:?}", v.as_ref().unwrap());
                        }*/
                        let v_ = v.unwrap();
                        // compute the total expected cost for the MAS
                        let mut cost: f32 = 0.;
                        for k in 0..v_.len() {
                            let r = X.get(&k).unwrap();
                            cost += r[..model.num_agents].iter()
                                .map(|x| v_[k] * *x)
                                .fold(0.0, |acc, y| acc + y);
                        }
                        println!("MAS cost: {}", cost);
                    } else {
                        println!("No randomised allocation function meets target threshold!");
                    }                
                    println!("Average loop run time: {}", avg_rt);
                    println!("Number of iterations: {}", l); 
                    println!("Total model checking time: {}", avg_rt * l as f32);
                    println!("Synthesis total run time: {}", t2.elapsed().as_secs_f32());
                } 
                Err(e) => {
                    println!("Error: {:?}", e);
                }
            }
            
        }
    }
}

pub fn generic_motap_msg_test_cpu<E> (
    model: &SCPM,
    env: &E,
    w: Vec<f32>,
    eps: f32,
    debug: i32,
    max_iter: usize,
    max_unstable: i32
) where E: Env<State> + Sync {
println!("--------------------------");
println!("     MOTAP CPU TEST       ");
println!("--------------------------");

    let t1 = Instant::now();
    let dbug = debug_level(debug);
    //model.construct_products(&mut mdp);
    let pairs = 
        product(0..model.num_agents, 0..model.tasks.size);
    let models_ls: Vec<MOProductMDP<State>> = pairs.into_par_iter().map(|(a, t)| {
        let pmdp = product_mdp_bfs(
            (env.get_init_state(a), 0), 
            env,
            &model.tasks.get_task(t), 
            a as i32, 
            t as i32, 
            model.num_agents, 
            model.num_agents + model.tasks.size,
            &model.actions
        );
        pmdp
    }).collect(); 
    match dbug {
        Debug::None => { }
        _ => { 
            println!("Time to create {} models: {:?}\n|S|: {},\n|P|: {}", 
                models_ls.len(), t1.elapsed().as_secs_f32(),
                models_ls.iter().fold(0, |acc, m| acc + m.states.len()),
                models_ls.iter().fold(0, |acc, m| acc + m.P.nnz())
            );
        }
    }

    let result = motap_cpu_only_solver(&models_ls, model.num_agents, 
                                            model.tasks.size, &w, eps, dbug, max_iter, max_unstable);
    match dbug {
        Debug::None => { }
        _ => { 
            println!("Total runtime {}", t1.elapsed().as_secs_f32());
            println!("result: {:?}", result);
        }
    }
}