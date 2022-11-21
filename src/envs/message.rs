use std::time::{Instant, Duration};
//use futures::executor::block_on;
//use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::IntoParallelIterator;
use sprs::visu::print_nnz_pattern;
use crate::algorithms::dp::{initial_policy, optimal_policy};
use crate::sparse::argmax::argmaxM;
use crate::{product, select_task_agent_pairs};
use crate::agent::env::Env;
use crate::model::momdp::{product_mdp_bfs, choose_random_policy};
use crate::model::scpm::SCPM;
use rayon::prelude::*;
use std::thread;
use crossbeam_channel::{unbounded, Receiver, Sender};

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
        model.tasks.get_task(0), 
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
            model.tasks.get_task(t), 
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

enum ControlMessage {
    Quit,
    Data(Vec<(usize, usize)>),
}

#[pyfunction]
pub fn experiment_gpu_cpu_binary_thread(
    model: &SCPM,
    env: &MessageSender,
    w: Vec<f32>, 
    epsilon: f32,
    GPU_BUFFER_SIZE: usize,
    CPU_COUNT: usize
) -> ()
where MessageSender: Env<State> {
    // TODO make this function generic

    // First step of the test is to construct an initial random policy
    // which can be chosen directly from the action vector which contains
    // the number of enabled actions in each of the states. 
    //let t1 = Instant::now();
    // construct the product of all of the model, env pairs
    //let mut _v: Vec<MOProductMDP<State>> = Vec::with_capacity(model.num_agents * model.tasks.size);
    let mut w_init = vec![0.; model.num_agents + model.tasks.size];
    let mut gpu_pairs;

    for k in 0..model.num_agents {
        w_init[k] = 1.;
    }
    
    let mut pairs = 
        product(0..model.num_agents, 0..model.tasks.size);
    // The idea of this test is to take estimates for products sizes
    // if the product size sum is larger than the number of cuda cores then 
    // send this partition to a thread for processing
    
    let (s1, r1) : (Sender<ControlMessage>, Receiver<ControlMessage>) = unbounded();
    let (s2, r2) : (Sender<i32>, Receiver<i32>) = unbounded();
    let e = env.clone();
    let m = model.clone();
    let thr = thread::spawn(move || {
        loop {
            match r1.try_recv() {
                Ok(data) => { 
                    match data {
                        ControlMessage::Quit => { break; }
                        // send the list items to oblivion
                        ControlMessage::Data(x) => { 
                            println!("Received data: {:?}", x);
                            for (a, t) in x.into_iter() {
                                // test building some models with the GPU
                                let pmdp = product_mdp_bfs(
                                    (e.get_init_state(a), 0), 
                                    &e, 
                                    m.tasks.get_task(t), 
                                    a as i32, 
                                    t as i32, 
                                    m.num_agents, 
                                    m.num_agents + m.tasks.size, 
                                    &m.actions
                                );
                                // The product MDPs will need to be combined into a 
                                // large matrix model and that data is then sent to the GPU
                                
                                // So we have a P, and R, but the rows and cols need editing
                                println!("CSR details");
                                println!("i: {:?}", pmdp.P.indptr().as_slice());
                                println!("j: {:?}", pmdp.P.indices());
                                println!("x: {:?}", pmdp.P.data());
                            }
                            thread::sleep(Duration::from_secs(5));
                            s2.send(99).unwrap();
                        }
                    }
                }
                Err(_) => { }
            }
            //Ok(data) => { println!("Received some data: {:?}", data); }
            //Err(e) => { println!("Received error: {:?}", e);}
        }
        let msg = "thread closed successfully";
        msg
    });

    (pairs, gpu_pairs) = 
        select_task_agent_pairs(model, env, pairs, GPU_BUFFER_SIZE);
    println!("remaining pairs: {:?}", pairs);
    println!("gpu pairs: {:?}", gpu_pairs);
    // fill the initial GPU buffer
    match s1.send(ControlMessage::Data(gpu_pairs)) {
        Ok(_) => { }
        Err(e) => { println!("err: {:?}", e); }
    };
    while pairs.len() > 0 {
        match r2.try_recv() {
            Ok(data) => { 
                println!("Received some work product from the GPU: {:?}", data);
                (pairs, gpu_pairs) = 
                    select_task_agent_pairs(model, env, pairs, GPU_BUFFER_SIZE);
                println!("Sending some more work to the GPU: [{:?}]", gpu_pairs);
                s1.send(ControlMessage::Data(gpu_pairs)).unwrap();
                // TODO call GPU work here
                // The GPU is a bit complicated, and we will need to send a copy of the
                // model and the environment to the thread for constructing model. 
            }
            Err(_) => { 
                // carve out a block of products to process on the CPU
                let cpu_pairs: Vec<(usize, usize)> = 
                    pairs.drain(0..std::cmp::min(CPU_COUNT, pairs.len())).collect();
                println!("GPU still busy, doing some work on the CPU [{:?}]", cpu_pairs);
                thread::sleep(Duration::from_secs(3));
                // TODO call CPU work here
                // The CPU is not too complicated because we will get access 
                // to access to the global memory
            }
        }
    }

    s1.send(ControlMessage::Quit).unwrap();

    println!("{:?}", thr.join().unwrap());
    

}
