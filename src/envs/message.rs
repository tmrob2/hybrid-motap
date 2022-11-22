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
    Data((usize, usize)),
    CPUData(Vec<(usize, usize)>)
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
    // TODO make this function generic

    // First step of the test is to construct an initial random policy
    // which can be chosen directly from the action vector which contains
    // the number of enabled actions in each of the states. 
    //let t1 = Instant::now();
    // construct the product of all of the model, env pairs
    //let mut _v: Vec<MOProductMDP<State>> = Vec::with_capacity(model.num_agents * model.tasks.size);
    let mut w_init = vec![0.; model.num_agents + model.tasks.size];

    for k in 0..model.num_agents {
        w_init[k] = 1.;
    }
    
    let mut pairs = 
        product(0..model.num_agents, 0..model.tasks.size);
    // The idea of this test is to take estimates for products sizes
    // if the product size sum is larger than the number of cuda cores then 
    // send this partition to a thread for processing
    
    let (gpu_s1, gpu_r1) : (Sender<ControlMessage>, Receiver<ControlMessage>) = unbounded();
    let (gpu_s2, gpu_r2) : (Sender<i32>, Receiver<i32>) = unbounded();

    let (cpu_s1, cpu_r1) : (Sender<ControlMessage>, Receiver<ControlMessage>) = unbounded();
    let (cpu_s2, cpu_r2) : (Sender<i32>, Receiver<i32>) = unbounded();

    let e = env.clone();
    let m = model.clone();
    let gpu_thread = thread::spawn(move || {
        loop {
            match gpu_r1.try_recv() {
                Ok(data) => { 
                    match data {
                        ControlMessage::Quit => { break; }
                        // send the list items to oblivion
                        ControlMessage::Data((a, t)) => { 

                            // TODO we need to store the current sum of modified stat space size

                            println!("Received data: ({},{})", a, t);

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
                            // send the data to CUDA to be processed and then returned to this thread
                            // so that we can send the processed data back to the main thread.                            
                            thread::sleep(Duration::from_secs(1));
                            gpu_s2.send(99).unwrap();
                        }
                        _ => {}
                    }
                }
                Err(_) => { }
            }
            //Ok(data) => { println!("Received some data: {:?}", data); }
            //Err(e) => { println!("Received error: {:?}", e);}
        }
        let msg = "GPU controller thread closed successfully";
        msg
    });

    // Create another thread for the CPUs to do work on
    let ecpu = env.clone();
    let mcpu = model.clone();
    let cpu_thread = thread::spawn(move || {
        // continuously loop and wait for data to be sent to the CPU for allocation
        loop {
            match cpu_r1.try_recv() {
                Ok(data) => { 
                    match data {
                        ControlMessage::Quit => { break; }
                        // send the list items to oblivion
                        ControlMessage::CPUData(x) => { 

                            // do the Rayon allocation of threads. 
                            println!("CPUs Received data: {:?}", x);

                            // TODO we need to store the current sum of modified stat space size
                            let _output: Vec<(usize, usize, f32)> = x.into_par_iter().map(|(a, t)| {
                                let pmdp = product_mdp_bfs(
                                    (ecpu.get_init_state(a), 0), 
                                    &ecpu, 
                                    mcpu.tasks.get_task(t), 
                                    a as i32, 
                                    t as i32, 
                                    mcpu.num_agents, 
                                    mcpu.num_agents + mcpu.tasks.size, 
                                    &mcpu.actions
                                );
                                (a, t, 1.0)
                            }).collect();

                            // send the data to CUDA to be processed and then returned to this thread
                            // so that we can send the processed data back to the main thread.                            
                            thread::sleep(Duration::from_secs(2));
                            cpu_s2.send(100).unwrap();
                        }
                        _ => { }
                    }
                }
                Err(_) => { }
            }
        }
        let msg = "CPU controller thread closed successfully";
        msg
    });

    // fill the initial GPU buffer and the CPU buffer with models
    gpu_s1.send(ControlMessage::Data(pairs.pop().unwrap())).unwrap();
    cpu_s1.send(ControlMessage::CPUData(
        pairs.drain(..std::cmp::min(CPU_COUNT, pairs.len())).collect())
    ).unwrap();
    while pairs.len() > 0 {
        // First try and allocate the data to the GPU, if the GPU is free
        // Otherwise try and allocate the data to the CPU if they are free
        // If no device is free, then continue looping until one of the devices
        // becomes free.
        match gpu_r2.try_recv() {
            Ok(data) => { 
                println!("Received some work product from the GPU: {:?}", data);
                let gpu_new_data = pairs.pop().unwrap();
                println!("Sending some more work to the GPU: [{:?}]", gpu_new_data);
                gpu_s1.send(ControlMessage::Data(gpu_new_data)).unwrap();
                // TODO call GPU work here
                // The GPU is a bit complicated, and we will need to send a copy of the
                // model and the environment to the thread for constructing model. 
            }
            Err(_) => { 
                // On receive error don't do anything, the GPU is not ready to take
                // on new messages
            }
        }

        match cpu_r2.try_recv() {
            Ok(data) => {
                println!("Received some work product from the CPUs: {:?}", data);
                let cpu_new_data: Vec<(usize, usize)> = pairs.drain(..CPU_COUNT).collect();
                println!("Sending some more work to the CPU: {:?}", cpu_new_data);
                cpu_s1.send(ControlMessage::CPUData(cpu_new_data)).unwrap();
            }
            Err(_) => { }
        }
    }

    gpu_s1.send(ControlMessage::Quit).unwrap();
    cpu_s1.send(ControlMessage::Quit).unwrap();

    println!("{:?}", cpu_thread.join().unwrap());
    println!("{:?}", gpu_thread.join().unwrap());
    

}
