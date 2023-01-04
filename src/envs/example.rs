use pyo3::prelude::*;
use std::time::Instant;
use crate::{agent::env::Env, debug_level, 
    model::{momdp::product_mdp_bfs, scpm::SCPM}, cpu_only_solver, Debug, 
    algorithms::synth::{scheduler_synthesis, HardwareChoice}};


type State = i32;


#[pyclass]
#[derive(Clone)]
// Example Agent
pub struct Example {
    pub states: Vec<i32>,
    pub initial_state: i32,
    pub action_space: Vec<i32>
}

#[pymethods]
impl Example {
    #[new]
    pub fn new() -> Self {
        Example {
            states: (0..4).collect(),
            initial_state: 0,
            action_space: (0..2).collect()
        }
    }
}

impl Env<State> for Example {
    fn step_(&self, s: State, action: u8, _task_id: i32) -> Result<Vec<(State, f32, String)>, String> {
        let transition: Result<Vec<(State, f32, String)>, String> = match s {
            0 => { 
                match action {
                    0 => { Ok(vec![(0, 0.3, "".to_string()), (1, 0.2, "b".to_string()), (2, 0.5, "".to_string())])}
                    1 => { Ok(vec![(1, 0.9, "b".to_string()), (2, 0.1, "".to_string())]) }
                    _ => { Err("Action not implemented".to_string()) }
                }
            }
            1 => { 
                match action { 
                    0 => { Ok(vec![(1, 1.0, "b".to_string())]) }
                    _ => { Err("Action not implemented".to_string()) }
                }
            }
            2 => { 
                match action { 
                    //_0 => { Ok(vec![(0, 1.0, "".to_string())]) }
                    0 => { Ok(vec![(3, 1.0, "a".to_string())]) }
                    _ => { Err("Action not implemented".to_string()) }
                }
            }
            3 => { 
                match action {
                    0 => { Ok(vec![(3, 1.0, "a".to_string())])}
                    _ => { Err("Action not implemented".to_string()) }
                }
            }
            _ => { Err("Not implemented".to_string()) }
        };
        transition
    }

    fn get_init_state(&self, _agent: usize) -> State {
        0
    }

    fn set_task(&mut self, _task_id: usize) {
        todo!()
    }

    fn get_action_space(&self) -> Vec<i32> {
        self.action_space.to_vec()
    }

    fn get_states_len(&self) -> usize {
        self.states.len()
    }
}

#[pyfunction]
pub fn example_cpu(
    model: &SCPM,
    env: &Example,
    w: Vec<f32>,
    eps: f32,
    debug: i32,
    max_iter: usize,
    max_unstable: i32
) {
    let dbug = debug_level(debug);

    let pmdp = product_mdp_bfs(
        (env.get_init_state(0), 0), 
        env, 
        &model.tasks.get_task(0), 
        0, 
        0, 
        1, 
        2, 
        &model.actions
    );

    let result = cpu_only_solver(&[pmdp], model.num_agents, 
        model.tasks.size, &w, eps, dbug, max_iter, max_unstable);
    match dbug {
        Debug::None => { }
            _ => { 
            println!("result: {:?}", result);
        }
    }
}

#[pyfunction]
pub fn ex_synthesis(
    model: &SCPM,
    env: &Example,
    w: Vec<f32>,
    target: Vec<f32>,
    epsilon1: f32,
    epsilon2: f32,
    debug: i32,
    max_iter: usize,
    max_unstable: i32
) {
println!(
"--------------------------\n
        SYNTH TEST         \n
--------------------------"
);
    let t1 = Instant::now();
    let dbug = debug_level(debug);
    //model.construct_products(&mut mdp);
    let pmdp = product_mdp_bfs(
        (env.get_init_state(0), 0), 
        env,
        &model.tasks.get_task(0), 
        0, 
        0, 
        model.num_agents, 
        model.num_agents + model.tasks.size,
        &model.actions
    );
    let models_ls = vec![pmdp];
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

    let res = scheduler_synthesis(
        models_ls, model.num_agents, model.tasks.size, w, &target, 
        epsilon1, epsilon2, HardwareChoice::CPU, dbug, max_iter, max_unstable
    );

    match dbug {
        Debug::None => { }
        _ => { 
            match res {
                Ok((_, rt, l)) => {
                    let avg_rt = rt.iter().fold(0., |acc, x| acc + x)/ rt.len() as f32;
                    println!("Average loop run time: {}", avg_rt);
                    println!("Number of iterations: {}", l); 
                    println!("Synthesis total run time: {}", t2.elapsed().as_secs_f32());
                } 
                Err(e) => {
                    println!("Error: {:?}", e);
                }
            }
            
        }
    }
}