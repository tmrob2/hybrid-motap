use std::time::Instant;
//use futures::executor::block_on;
//use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use crate::algorithms::dp::{initial_policy, optimal_policy};
use crate::sparse::argmax::argmaxM;
use crate::{product, cuda_initial_policy_value, cuda_policy_optimisation, 
    cuda_warm_up_gpu, cuda_initial_policy_value_pinned_graph, allocation_fn,
    cuda_multi_obj_solution};
use crate::agent::env::Env;
use crate::model::momdp::{product_mdp_bfs, choose_random_policy};
use crate::model::scpm::SCPM;
use crate::algorithms::hybrid::hybrid_stage1;
use crate::model::momdp::MOProductMDP;
use hashbrown::HashMap;
use sprs::CsMatBase;

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
    CPU_COUNT: usize
) -> ()
where MessageSender: Env<State> {

    // first construct the models
    let t1 = Instant::now();
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
    println!("Time to create {} models: {:?}", output.len(), t1.elapsed().as_secs_f32()); 
    hybrid_stage1(
        output, model.num_agents, model.tasks.size, w, epsilon, CPU_COUNT
    );
}

#[pyfunction]
pub fn msg_test_gpu_stream(
    model: &SCPM,
    env: &mut MessageSender,
    w: Vec<f32>,
    eps: f32
) {
    cuda_warm_up_gpu();
    let t1 = Instant::now();
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

    let mut w_init = vec![0.; model.num_agents + model.tasks.size];
    for k in 0..model.num_agents {
        w_init[k] = 1.;
    }

    println!("Time to create {} models: {:?}\n|S|: {},\n|P|: {}", 
        models_ls.len(), t1.elapsed().as_secs_f32(),
        models_ls.iter().fold(0, |acc, m| acc + m.P.shape().1),
        models_ls.iter().fold(0, |acc, m| acc + m.P.nnz())
    );
    let t2 = Instant::now();
    let mut results: HashMap<i32, Vec<Option<(i32, Vec<i32>, f32)>>> = HashMap::new();
    for pmdp in models_ls.iter() {
        let mut pi = choose_random_policy(pmdp);
        let rowblock = pmdp.states.len() as i32;
        let pcolblock = rowblock as i32;
        let rcolblock = (model.num_agents + model.tasks.size) as i32;
        let initP = 
            argmaxM(pmdp.P.view(), &pi, rowblock, pcolblock, &pmdp.adjusted_state_act_pair);
        let initR = 
            argmaxM(pmdp.R.view(), &pi, rowblock, rcolblock, &pmdp.adjusted_state_act_pair);

        let mut r_v_init: Vec<f32> = vec![0.; initR.shape().0 as usize];
        let mut x_init: Vec<f32> = vec![0.; initP.shape().1 as usize];
        let mut y_init: Vec<f32> = vec![0.; initP.shape().0 as usize];
        let mut unstable: Vec<i32> = vec![0; initP.shape().0 as usize];
        let mut stable: Vec<f32> = vec![0.; x_init.len()];
        let mut y: Vec<f32> = vec![0.; pmdp.P.shape().0];
        let mut rmv: Vec<f32> = vec![0.; pmdp.P.shape().0];

        cuda_initial_policy_value_pinned_graph(
            initP.view(), 
            initR.view(), 
            pmdp.P.view(), 
            pmdp.R.view(), 
            &w_init,
            &w, 
            eps, 
            &mut x_init, 
            &mut y_init, 
            &mut r_v_init, 
            &mut y, 
            &mut rmv, 
            &mut unstable, 
            &mut pi, 
            &pmdp.enabled_actions, 
            &pmdp.adjusted_state_act_pair,
            &mut stable,
        );

        match results.get_mut(&pmdp.task_id) {
            Some(v) => { 
                v[pmdp.agent_id as usize] = Some((
                    pmdp.agent_id, 
                    pi.to_owned(), 
                    x_init[*pmdp.state_map.get(&pmdp.initial_state).unwrap()]
                ));
            }
            None => {
                results.insert(
                    pmdp.task_id,
                    (0..model.num_agents).map(|i| if i as i32 == pmdp.agent_id{
                        // insert the current tuple
                        Some((i as i32, 
                        pi.to_owned(),
                        x_init[*pmdp.state_map.get(&pmdp.initial_state).unwrap()]))
                    } else {
                        None
                    }).collect::<Vec<Option<(i32, Vec<i32>, f32)>>>()
                );
            }
        }   
    }
    println!("Time to do stage 1 {}", t2.elapsed().as_secs_f32());
    let allocation = allocation_fn(
        &results, model.tasks.size, model.num_agents
    );

    // Then for each allocation we need to make the argmax P, R matrices
    let allocatedArgMax: Vec<(CsMatBase<f32, i32, Vec<i32>, Vec<i32>, Vec<f32>>,
                              CsMatBase<f32, i32, Vec<i32>, Vec<i32>, Vec<f32>>)> 
        = allocation.into_par_iter().map(|(t, a, pi)| {
        let pmdp: &MOProductMDP<State> = models_ls.iter()
            .filter(|m| m.agent_id == a && m.task_id == t)
            .collect::<Vec<&MOProductMDP<State>>>()[0];
            let rowblock = pmdp.states.len() as i32;
            let pcolblock = rowblock as i32;
            let rcolblock = (model.num_agents + model.tasks.size) as i32;
            let argmaxP = 
                argmaxM(pmdp.P.view(), &pi, rowblock, pcolblock, &pmdp.adjusted_state_act_pair);
            let argmaxR = 
                argmaxM(pmdp.R.view(), &pi, rowblock, rcolblock, &pmdp.adjusted_state_act_pair);
            (argmaxP, argmaxR)
    }).collect();
    let nobjs = model.num_agents + model.tasks.size;
    let (P, R) = allocatedArgMax[0].to_owned();
    let mut storage = vec![0.; P.shape().0 * nobjs];
    cuda_multi_obj_solution(P.view(), R.view(), &mut storage, eps, nobjs as i32);
    println!("Total runtime {}", t2.elapsed().as_secs_f32());
    println!("Test result: {:?}", storage);
}
