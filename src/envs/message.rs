use std::time::Instant;
//use futures::executor::block_on;
//use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use crate::algorithms::dp::{initial_policy, optimal_policy};
use crate::sparse::argmax::argmaxM;
use crate::{product, cuda_initial_policy_value, cuda_policy_optimisation, 
    cuda_warm_up_gpu, gpu_only_solver, cpu_only_solver, hybrid_solver, debug_level};
use crate::agent::env::Env;
use crate::model::momdp::{product_mdp_bfs, choose_random_policy};
use crate::model::scpm::SCPM;
use crate::model::momdp::MOProductMDP;
use crate::Debug;

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
    fn step_(&self, s: State, action: u8, _task_id: i32) -> Result<Vec<(State, f32, String)>, String> {
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

    fn get_states_len(&self) -> usize {
        self.states.len()
    }
}

#[pyfunction]
pub fn test_build(
    model: &SCPM,
    env: &MessageSender
) -> ()
where MessageSender: Env<State> {
    // just want to run an implementation to check a model build outcome
    let _pmdp = product_mdp_bfs(
        (env.get_init_state(0),0), 
        env, 
        &model.tasks.get_task(0), 
        0, 
        0, 
        model.num_agents, 
        model.num_agents + model.tasks.size, 
        &model.actions
    );
}

#[pyfunction]
pub fn test_initial_policy(
    model: &SCPM,
    env: &MessageSender,
    w: Vec<f32>, 
    epsilon: f32
) -> (Vec<f32>, Vec<i32>)
where MessageSender: Env<State> {
    // First step of the test is to construct an initial random policy
    // which can be chosen directly from the action vector which contains
    // the number of enabled actions in each of the states. 

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
    // compute the initial random policy based on the enabled actions
    let pi = choose_random_policy(&pmdp);
    println!("Pi: {:?}", pi);
    println!("Rm: {:?}", pmdp.R.shape().0);
    println!("Rn: {:?}", pmdp.R.shape().1);
    println!("Pm: {:?}", pmdp.P.shape().0);
    println!("Pn: {:?}", pmdp.P.shape().1);

    /*for (k, (s, q)) in pmdp.states.iter().enumerate() {
        println!("sidx: {}, (s:{}, q:{})", k, s, q);
    }*/

    // construct the matrices under the initial random policy.
    let rowblock = pmdp.states.len() as i32;
    let pcolblock = rowblock as i32;
    let rcolblock = (model.num_agents + model.tasks.size) as i32;
    let initP = argmaxM(pmdp.P.view(), &pi, rowblock, pcolblock, &pmdp.adjusted_state_act_pair);
    let initR = argmaxM(pmdp.R.view(), &pi, rowblock, rcolblock, &pmdp.adjusted_state_act_pair);

    println!("P: {:?}", pmdp.P.to_dense());
    println!("R: {:?}", pmdp.R.to_dense());

    // using the random policy determine the value of the intial policy
    let mut r_v: Vec<f32> = vec![0.; initR.shape().0 as usize];
    let mut x: Vec<f32> = vec![0.; initP.shape().1 as usize];
    let mut y: Vec<f32> = vec![0.; initP.shape().0 as usize];
    
    initial_policy(initP.view(), initR.view(), &w, epsilon, &mut r_v, &mut x, &mut y);

    (x.to_owned(), pi)

}

#[pyfunction]
pub fn test_threaded_initial_policy(
    model: &SCPM,
    env: &MessageSender,
    w: Vec<f32>, 
    epsilon: f32
) -> ()
where MessageSender: Env<State> {
    // First step of the test is to construct an initial random policy
    // which can be chosen directly from the action vector which contains
    // the number of enabled actions in each of the states. 
    let t1 = Instant::now();
    // construct the product of all of the model, env pairs
    //let mut _v: Vec<MOProductMDP<State>> = Vec::with_capacity(model.num_agents * model.tasks.size);
    let mut w_init = vec![0.; model.num_agents + model.tasks.size];
    for k in 0..model.num_agents {
        w_init[k] = 1.;
    }
    // the question is can we optimise this?
    let _output: Vec<(usize, usize, f32)> = product(0..model.num_agents, 0..model.tasks.size).into_par_iter().map(|(a, t)| {
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

        let mut pi = choose_random_policy(&pmdp);

        let rowblock = pmdp.states.len() as i32;
        let pcolblock = rowblock as i32;
        let rcolblock = (model.num_agents + model.tasks.size) as i32;
        let initP = 
            argmaxM(pmdp.P.view(), &pi, rowblock, pcolblock, 
                    &pmdp.adjusted_state_act_pair);
        let initR = 
            argmaxM(pmdp.R.view(), &pi, rowblock, rcolblock, 
                    &pmdp.adjusted_state_act_pair);

        let mut r_v: Vec<f32> = vec![0.; initR.shape().0];
        let mut x: Vec<f32> = vec![0.; initP.shape().1];
        let mut y: Vec<f32> = vec![0.; initP.shape().0];
        
        initial_policy(initP.view(), initR.view(), &w, epsilon, 
                       &mut r_v, &mut x, &mut y);

        // taking the initial policy and the value vector for the initial policy
        // what is the optimal policy
        let mut r_v: Vec<f32> = vec![0.; pmdp.R.shape().0];
        let mut y: Vec<f32> = vec![0.; pmdp.P.shape().0];
        let r = optimal_policy(pmdp.P.view(), pmdp.R.view(), &w, epsilon, 
                       &mut r_v, &mut x, &mut y, &mut pi, 
                       &pmdp.enabled_actions, &pmdp.adjusted_state_act_pair,
                       *pmdp.state_map.get(&pmdp.initial_state).unwrap()
                    );
        (a, t, r)
    }).collect();

    // The allocation function can be worked out at this point. 
    println!("Elapsed time: {:?}", t1.elapsed().as_secs_f32());
    //println!("Solution: {:?}", output);

}

#[pyfunction]
pub fn test_cuda_initial_policy(
    model: &SCPM,
    env: &MessageSender,
    w: Vec<f32>, 
    epsilon: f32
) -> ()
where MessageSender: Env<State> {
    // First step of the test is to construct an initial random policy
    // which can be chosen directly from the action vector which contains
    // the number of enabled actions in each of the states. 
    let t1 = Instant::now();
    // construct the product of all of the model, env pairs
    //let mut _v: Vec<MOProductMDP<State>> = Vec::with_capacity(model.num_agents * model.tasks.size);
    let mut w_init = vec![0.; model.num_agents + model.tasks.size];
    for k in 0..model.num_agents {
        w_init[k] = 1.;
    }
    // the question is can we optimise this?
    let pmdp = product_mdp_bfs(
        (env.get_init_state(0), 0), 
        env, 
        &model.tasks.get_task(0), 
        0 as i32, 
        0 as i32, 
        model.num_agents, 
        model.num_agents + model.tasks.size, 
        &model.actions
    );

    let pi = choose_random_policy(&pmdp);

    let rowblock = pmdp.states.len() as i32;
    let pcolblock = rowblock as i32;
    let rcolblock = (model.num_agents + model.tasks.size) as i32;
    let initP = 
        argmaxM(pmdp.P.view(), &pi, rowblock, pcolblock, 
                &pmdp.adjusted_state_act_pair);
    let initR = 
        argmaxM(pmdp.R.view(), &pi, rowblock, rcolblock, 
                &pmdp.adjusted_state_act_pair);

    let mut r_v: Vec<f32> = vec![0.; initR.shape().0];
    let mut x: Vec<f32> = vec![0.; initP.shape().1];
    let mut y: Vec<f32> = vec![0.; initP.shape().0];
    let mut unstable: Vec<i32> = vec![0; initP.shape().0];
    
    cuda_initial_policy_value(initP.view(), initR.view(), &w, epsilon, 
                              &mut r_v, &mut x, &mut y, &mut unstable);
    println!("x: {:?}", x);

    // The allocation function can be worked out at this point. 
    println!("Elapsed time: {:?}", t1.elapsed().as_secs_f32());
    //println!("Solution: {:?}", output);

}

#[pyfunction]
pub fn test_cudacpu_opt_pol(
    model: &SCPM,
    env: &MessageSender,
    w: Vec<f32>, 
    epsilon: f32
) -> ()
where MessageSender: Env<State> {
    // First step of the test is to construct an initial random policy
    // which can be chosen directly from the action vector which contains
    // the number of enabled actions in each of the states. 

    let mut w_init = vec![0.; model.num_agents + model.tasks.size];
    for k in 0..model.num_agents {
        w_init[k] = 1.;
    }

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
    // compute the initial random policy based on the enabled actions
    let mut pi = choose_random_policy(&pmdp);

    // construct the matrices under the initial random policy.
    let rowblock = pmdp.states.len() as i32;
    let pcolblock = rowblock as i32;
    let rcolblock = (model.num_agents + model.tasks.size) as i32;
    let initP = 
        argmaxM(pmdp.P.view(), &pi, rowblock, pcolblock, &pmdp.adjusted_state_act_pair);
    let initR = 
        argmaxM(pmdp.R.view(), &pi, rowblock, rcolblock, &pmdp.adjusted_state_act_pair);

    // using the random policy determine the value of the intial policy
    let mut r_v: Vec<f32> = vec![0.; initR.shape().0 as usize];
    let mut x: Vec<f32> = vec![0.; initP.shape().1 as usize];
    let mut y: Vec<f32> = vec![0.; initP.shape().0 as usize];
    
    initial_policy(initP.view(), initR.view(), &w_init, epsilon, &mut r_v, &mut x, &mut y);

    let mut r_v: Vec<f32> = vec![0.; pmdp.R.shape().0];
    let mut y: Vec<f32> = vec![0.; pmdp.P.shape().0];
    let mut xcpu = x.to_vec();
    let t1 = Instant::now();
    let rcpu = optimal_policy(pmdp.P.view(), pmdp.R.view(), &w, epsilon, 
        &mut r_v, &mut xcpu, &mut y, &mut pi, &pmdp.enabled_actions, 
        &pmdp.adjusted_state_act_pair, *pmdp.state_map.get(&pmdp.initial_state).unwrap()
    );
    println!("CPU policy optimisation: {:?} (ns)", t1.elapsed().as_nanos());
    
    let mut xgpu = x.to_vec();
    let mut r_v: Vec<f32> = vec![0.; pmdp.R.shape().0];
    let mut y: Vec<f32> = vec![0.; pmdp.P.shape().0];
    let t1 = Instant::now();
    let mut stable: Vec<f32> = vec![0.; xgpu.len()];
    let rgpu = cuda_policy_optimisation(pmdp.P.view(), pmdp.R.view(), &w, 
        epsilon, &mut pi, &mut xgpu, &mut y, &mut r_v, &pmdp.enabled_actions, 
        &pmdp.adjusted_state_act_pair, *pmdp.state_map.get(&pmdp.initial_state).unwrap(),
        &mut stable
    );
    println!("GPU policy optimisation: {:?} (ns)", t1.elapsed().as_nanos());

    println!("CPU: {}, GPU: {}", rcpu, rgpu);
}

#[pyfunction]
pub fn experiment_gpu_cpu_binary_thread(
    model: &SCPM,
    env: &MessageSender,
    w: Vec<f32>, 
    epsilon: f32,
    CPU_COUNT: usize,
    debug: i32
) -> ()
where MessageSender: Env<State> {
println!(
"--------------------------\n
        HYBRID TEST        \n
--------------------------"
);
    // first construct the models
    cuda_warm_up_gpu();
    let t1 = Instant::now();
    let dbug = debug_level(debug);
    let pairs = 
    product(0..model.num_agents, 0..model.tasks.size);
    let output: Vec<MOProductMDP<State>> = pairs.into_par_iter().map(|(a, t)| {
        //env.set_task(*t);
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
            println!("Time to create {} models: {:?}", output.len(), t1.elapsed().as_secs_f32()); 
        }
    }
    let results = hybrid_solver(
        output, model.num_agents, model.tasks.size, &w, epsilon, CPU_COUNT, dbug
    );
    match dbug {
        Debug::None => { }
        _ => {
            println!("result: {:?}", results);
            println!("Total runtime {}", t1.elapsed().as_secs_f32());
        }
    }
    
    // construct the allocation function

}

#[pyfunction]
pub fn msg_test_gpu_stream(
    model: &SCPM,
    env: &MessageSender,
    w: Vec<f32>,
    eps: f32,
    debug: i32
) {
println!(
"--------------------------\n
        GPU TEST           \n
--------------------------"
);
    cuda_warm_up_gpu();

    let t1 = Instant::now();
    let dbug = debug_level(debug);
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
                models_ls.iter().fold(0, |acc, m| acc + m.P.shape().1),
                models_ls.iter().fold(0, |acc, m| acc + m.P.nnz())
            );
        }
    }
    let result = gpu_only_solver(models_ls, model.num_agents, 
                                            model.tasks.size, &w, eps, dbug);
    
    match dbug {
        Debug::None => { }
        _ => {
            println!("Total runtime {}", t1.elapsed().as_secs_f32());
            println!("result: {:?}", result);
        }
    }
}

#[pyfunction]
pub fn msg_test_cpu(
    model: &SCPM,
    env: &MessageSender,
    w: Vec<f32>,
    eps: f32,
    debug: i32
) {
println!(
"--------------------------\n
        CPU TEST           \n
--------------------------"
);
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
                models_ls.iter().fold(0, |acc, m| acc + m.P.shape().1),
                models_ls.iter().fold(0, |acc, m| acc + m.P.nnz())
            );
        }
    }

    let result = cpu_only_solver(models_ls, model.num_agents, 
                                            model.tasks.size, &w, eps, dbug);
    match dbug {
        Debug::None => { }
        _ => { 
            println!("Total runtime {}", t1.elapsed().as_secs_f32());
            println!("result: {:?}", result);
        }
    }
}
