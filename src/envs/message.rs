use std::borrow::Borrow;

//use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use crate::sparse::argmax::argmaxM;
use crate::{CxxMatrixf32, cpu_intial_policy, adjust_value_vector, cpu_policy_optimisation};
use crate::agent::env::Env;
use crate::model::momdp::{product_mdp_bfs, choose_random_policy};
use crate::model::scpm::SCPM;
use crate::threading::qthread::start_threads_compute_model;

type State = i32;

#[pyclass]
#[derive(Clone)]
// A message sender is a single agent
pub struct MessageSender {
    pub states: Vec<i32>,
    pub initial_state: i32,
    pub action_space: Vec<i32>
}

#[pymethods]
impl MessageSender {
    #[new]
    pub fn new() -> Self {
        MessageSender {
            states: (0..5).collect(),
            initial_state: 0,
            action_space: (0..2).collect()
        }
    }
}

impl Env<State> for MessageSender {
    fn step_(&self, s: State, action: u8) -> Result<Vec<(State, f32, String)>, String> {
        let transition: Result<Vec<(State, f32, String)>, String> = match s {
            0 => {
                // return the transition for state 0
                match action {
                    0 => {Ok(vec![(0, 0.01, "".to_string()), (1, 0.99, "i".to_string())])}
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
                // return the transition for state 2
                match action {
                    0 => {Ok(vec![(3, 0.99, "s".to_string()), (4, 0.01, "e".to_string())])}
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
                    0 => {Ok(vec![(0, 1.0, "".to_string())])}
                    _ => {Err("Action not-implemented".to_string())}
                }
            }
            _ => {
                // Not implemented error
                Err("Not-implemented".to_string())
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
}

#[pyfunction]
pub fn test_build(
    model: &SCPM,
    env: &MessageSender
) -> (CxxMatrixf32, CxxMatrixf32)
where MessageSender: Env<State> {
    // just want to run an implementation to check a model build outcome
    let pmdp = product_mdp_bfs(
        (env.get_init_state(0),0), 
        env.clone(), 
        model.tasks.get_task(0), 
        0, 
        0, 
        model.num_agents, 
        model.num_agents + model.tasks.size, 
        &model.actions
    );
    (pmdp.P, pmdp.R)
}

#[pyfunction]
pub fn thread_test(
    model: &SCPM,
    env: &MessageSender
) where MessageSender: Env<State> {
   start_threads_compute_model(model, env.clone());
}

#[pyfunction]
pub fn test_initial_policy(
    model: &SCPM,
    env: &MessageSender,
    w: Vec<f32>, 
    epsilon: f32
) -> (CxxMatrixf32, CxxMatrixf32, Vec<f32>, Vec<i32>)
where MessageSender: Env<State> {
    // First step of the test is to construct an initial random policy
    // which can be chosen directly from the action vector which contains
    // the number of enabled actions in each of the states. 

    let pmdp = product_mdp_bfs(
        (env.get_init_state(0), 0), 
        env.clone(), 
        model.tasks.get_task(0), 
        0, 
        0, 
        model.num_agents, 
        model.num_agents + model.tasks.size, 
        &model.actions
    );
    // compute the initial random policy based on the enabled actions
    let pi = choose_random_policy(&pmdp);
    println!("Pi: {:?}", pi);
    println!("Rm: {:?}", pmdp.R.m);
    println!("Rn: {:?}", pmdp.R.n);
    println!("Pm: {:?}", pmdp.P.m);
    println!("Pn: {:?}", pmdp.P.n);

    /*for (k, (s, q)) in pmdp.states.iter().enumerate() {
        println!("sidx: {}, (s:{}, q:{})", k, s, q);
    }*/

    // construct the matrices under the initial random policy.
    let rowblock = pmdp.states.len() as i32;
    let pcolblock = rowblock as i32;
    let rcolblock = (model.num_agents + model.tasks.size) as i32;
    let initP = argmaxM(&pmdp.P, &pi, rowblock, pcolblock, &pmdp.adjusted_state_act_pair);
    let initR = argmaxM(&pmdp.R, &pi, rowblock, rcolblock, &pmdp.adjusted_state_act_pair);


    // using the random policy determine the value of the intial policy
    let mut r_v: Vec<f32> = vec![0.; initR.m as usize];
    let mut x: Vec<f32> = vec![0.; initP.n as usize];
    let mut y: Vec<f32> = vec![0.; initP.m as usize];
    cpu_intial_policy(&initP, &initR, &mut r_v, &w, &mut x, &mut y, epsilon);

    /*let init_value_vector_adj = adjust_value_vector(
        &x, &pmdp.adjusted_state_act_pair, &pmdp.enabled_actions, pmdp.P.m as usize
    );*/

    (initP, initR, x.to_owned(), pi)

}

#[pyfunction]
pub fn test_policy_optimisation(
    model: &SCPM,
    env: &MessageSender,
    w: Vec<f32>, 
    epsilon: f32,
    mut x: Vec<f32>,
    mut pi: Vec<i32>
) -> ()
where MessageSender: Env<State> {
    // First step of the test is to construct an initial random policy
    // which can be chosen directly from the action vector which contains
    // the number of enabled actions in each of the states. 
    println!("Policy\n{:?}", pi);
    println!("Init value vec:\n{:?}", x);

    let pmdp = product_mdp_bfs(
        (env.get_init_state(0), 0), 
        env.clone(), 
        model.tasks.get_task(0), 
        0, 
        0, 
        model.num_agents, 
        model.num_agents + model.tasks.size, 
        &model.actions
    );

    // using the random policy determine the value of the intial policy
    let mut r_v: Vec<f32> = vec![0.; pmdp.R.m as usize];
    let mut y: Vec<f32> = vec![0.; pmdp.P.m as usize];
    cpu_policy_optimisation(&pmdp.P, &pmdp.R, &mut r_v, &w, 
        &mut x, &mut y, &mut pi, &pmdp.enabled_actions, 
        &pmdp.adjusted_state_act_pair, epsilon);

    println!("x: {:?}", x);
}
