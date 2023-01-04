//use crate::model::scpm::SCPM;
//use crate::agent::env::Env;
use crate::{cuda_initial_policy_value_pinned_graph, cuda_multi_obj_solution, Debug, choose_random_policy};
use crate::sparse::argmax::argmaxM;
use crate::model::momdp::MOProductMDP;
use crate::model::general::ModelFns;
use crossbeam_channel::{unbounded, Receiver, Sender};
use std::thread;
use std::time::Instant;
use rayon::prelude::*;
use super::dp::{initial_policy, optimal_policy, optimal_values};
use hashbrown::HashMap;
use sprs::CsMatBase;

pub enum CtrlMsg<S> {
    Quit,
    Data(MOProductMDP<S>),
    CPUData(Vec<MOProductMDP<S>>),
}

pub enum MOCtrlMsg {
    Quit,
    GPUMOData((
        i32, i32,
        CsMatBase<f32, i32, Vec<i32>, Vec<i32>, Vec<f32>>, 
        CsMatBase<f32, i32, Vec<i32>, Vec<i32>, Vec<f32>>,
        usize,
    )),
    CPUMOData(Vec<(
        i32, i32,
        CsMatBase<f32, i32, Vec<i32>, Vec<i32>, Vec<f32>>, 
        CsMatBase<f32, i32, Vec<i32>, Vec<i32>, Vec<f32>>,
        usize
    )>)
}

// TODO We need a smarter way of dealing with models without making a clone of 
// all of the models
pub fn hybrid_stage1<S>(
    mut models: Vec<MOProductMDP<S>>,
    num_agents: usize,
    num_tasks: usize,
    w: Vec<f32>, 
    epsilon: f32,
    CPU_COUNT: usize,
    debug: crate::Debug,
    max_iter: usize,
    max_unstable: i32
) -> (Vec<MOProductMDP<S>>, Vec<i32>, HashMap<(i32, i32), Vec<i32>>)
where S: Copy + Clone + std::fmt::Debug + Eq + std::hash::Hash + 'static, 
      MOProductMDP<S>: Send + Clone {

    // TODO probably need to do something with the output here to format it into 
    // ksomething that the allocation function expects.
    let t1 = Instant::now();
    let mut models_return: Vec<MOProductMDP<S>> = Vec::new();
    let mut M: Vec<i32> = vec![0; num_agents * num_tasks];
    let mut Pi: HashMap<(i32, i32), Vec<i32>> = HashMap::new();
    // First step of the test is to construct an initial random policy
    // which can be chosen directly from the action vector which contains
    // the number of enabled actions in each of the states. 
    //let t1 = Instant::now();
    // construct the product of all of the model, env pairs
    //let mut _v: Vec<MOProductMDP<State>> = Vec::with_capacity(model.num_agents * model.tasks.size);
    
    let (gpu_s1, gpu_r1) : (Sender<CtrlMsg<S>>, Receiver<CtrlMsg<S>>) = unbounded();
    let (gpu_s2, gpu_r2) : (Sender<(MOProductMDP<S>, Vec<i32>, f32)>, Receiver<(MOProductMDP<S>, Vec<i32>, f32)>) = unbounded();
    
    let (cpu_s1, cpu_r1) : (Sender<CtrlMsg<S>>, Receiver<CtrlMsg<S>>) = unbounded();
    let (cpu_s2, cpu_r2) : (Sender<Vec<(MOProductMDP<S>, Vec<i32>, f32)>>, Receiver<Vec<(MOProductMDP<S>, Vec<i32>, f32)>>) = unbounded();
    
    let mut w_init = vec![0.; num_agents + num_tasks];
    for k in 0..num_agents {
        w_init[k] = 1.;
    }
    let w_gpu = w.to_vec();
    let gpu_thread = thread::spawn(move || {
        loop {
            match gpu_r1.try_recv() {
                Ok(data) => { 
                    match data {
                        CtrlMsg::Quit => { break; }
                        // send the list items to oblivion
                        CtrlMsg::Data(pmdp) => { 

                            // TODO we need to store the current sum of modified stat space size
                            let state_size = pmdp.get_states().len();
                            let enabled_actions = pmdp.get_enabled_actions();
                            let mut pi = choose_random_policy(state_size, enabled_actions);

                            let rowblock = pmdp.states.len() as i32;
                            let pcolblock = rowblock as i32;
                            let rcolblock = (num_agents + num_tasks) as i32;
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
                            
                            match debug {
                                Debug::Verbose1 => { 
                                    println!("Received GPU model: ({},{})", pmdp.agent_id, pmdp.task_id);
                                }
                                _ => { }
                            }
                            // send the data to CUDA to be processed and then returned to this thread
                            // so that we can send the processed data back to the main thread.                            
                            cuda_initial_policy_value_pinned_graph(
                                initP.view(), 
                                initR.view(), 
                                pmdp.P.view(), 
                                pmdp.R.view(), 
                                &w_init,
                                &w_gpu, 
                                epsilon, 
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
                                max_iter as i32,
                                max_unstable
                            );
                            let r = x_init[*pmdp.state_map.get(&pmdp.initial_state).unwrap()];
                            match debug {
                                Debug::Verbose1 => { 
                                    println!("GPU sending data");                                }
                                _ => { }
                            }
                            
                            gpu_s2.send((pmdp, pi, r)).unwrap();
                        }
                        _ => {}
                    }
                }
                Err(_) => { }
            }
            //Ok(data) => { println!("Received some data: {:?}", data); }
            //Err(e) => { println!("Received error: {:?}", e);}
        }
        match debug {
            Debug::Verbose1 | Debug::Verbose2 => { 
                let msg = "GPU controller thread closed successfully";
                println!("{:?}", msg);
            }
            _ => { }
        }
    });

    // Create another thread for the CPUs to do work on
    let cpu_thread = thread::spawn(move || {
        // continuously loop and wait for data to be sent to the CPU for allocation
        loop {
            match cpu_r1.try_recv() {
                Ok(data) => { 
                    match data {
                        CtrlMsg::Quit => { break; }
                        // send the list items to oblivion
                        CtrlMsg::CPUData(model_ls) => { 

                            // do the Rayon allocation of threads. 
                            match debug {
                                Debug::Verbose1 => { 
                                    println!("CPUs Received data: {:?} models", 
                                    model_ls.iter().map(|m| (m.agent_id, m.task_id)).collect::<Vec<(i32, i32)>>());                              }
                                _ => { }
                            }

                            let output: Vec<(MOProductMDP<S>, Vec<i32>, f32)> = model_ls.into_par_iter().map(|pmdp| {

                                let state_size = pmdp.get_states().len();
                                let enabled_actions = pmdp.get_enabled_actions();
                                let mut pi = choose_random_policy(state_size, enabled_actions);

                                let rowblock = pmdp.states.len() as i32;
                                let pcolblock = rowblock as i32;
                                let rcolblock = (num_agents + num_tasks) as i32;
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
                                            &mut r_v, &mut x, &mut y, max_iter, max_unstable);

                                // taking the initial policy and the value vector for the initial policy
                                // what is the optimal policy
                                let mut r_v: Vec<f32> = vec![0.; pmdp.R.shape().0];
                                let mut y: Vec<f32> = vec![0.; pmdp.P.shape().0];
                                let r = optimal_policy(pmdp.P.view(), pmdp.R.view(), &w, epsilon, 
                                            &mut r_v, &mut x, &mut y, &mut pi, 
                                            &pmdp.enabled_actions, &pmdp.adjusted_state_act_pair,
                                            *pmdp.state_map.get(&pmdp.initial_state).unwrap(),
                                            max_iter
                                            );
                                (pmdp, pi, r)
                            }).collect();
                            cpu_s2.send(output).unwrap();
                        }
                        _ => { }
                    }
                }
                Err(_) => { }
            }
        }
        match debug {
            Debug::Verbose1 | Debug::Verbose2 => { 
                let msg = "CPU controller thread closed successfully";
                println!("{:?}", msg)
            }
            _ => { }
        }
    });
    
    let mut received_count = 0;
    let models_to_send = models.len();
    // fill the initial GPU buffer and the CPU buffer with models
    gpu_s1.send(CtrlMsg::Data(models.pop().unwrap())).unwrap();
    cpu_s1.send(CtrlMsg::CPUData(
        models.drain(..std::cmp::min(CPU_COUNT, models.len())).collect())
    ).unwrap();
    // need to gather the results of the first stage specifically for input
    // into the allocation function 
    while received_count < models_to_send {
        // First try and allocate the data to the GPU, if the GPU is free
        // Otherwise try and allocate the data to the CPU if they are free
        // If no device is free, then continue looping until one of the devices
        // becomes free.
        match gpu_r2.try_recv() {
            Ok((gpu_model, pi, data)) => { 
                match debug {
                    Debug::Verbose1 => { 
                        println!("Received some work product from the GPU: [{}, {}]",
                            gpu_model.agent_id, gpu_model.task_id);
                    }
                    _ => { }
                }
                //let v = results.get_mut(&gpu_model.task_id).unwrap();
                //v[gpu_model.agent_id as usize] = Some((gpu_model.agent_id,pi,data));
                M[gpu_model.task_id as usize * num_agents + gpu_model.agent_id as usize] = 
                    (data * 1_000_000.0) as i32;
                Pi.insert((gpu_model.agent_id, gpu_model.task_id), pi);
                models_return.push(gpu_model);
                received_count += 1;
                if !models.is_empty() {
                    let gpu_new_data = models.pop().unwrap();
                    match debug {
                        Debug::Verbose1 => { 
                            println!("Sending some more work to the GPU: id [{},{}]", 
                                gpu_new_data.agent_id, gpu_new_data.task_id);
                        }
                        _ => { }
                    }
                    gpu_s1.send(
                        CtrlMsg::Data(gpu_new_data)
                    ).unwrap();
                    // TODO call GPU work here. 
                }
            }
            Err(_) => { 
                // On receive error don't do anything, the GPU is not ready to take
                // on new messages
            }
        }

        match cpu_r2.try_recv() {
            Ok(mut output) => {
                // capture the CPU models in a larger list and then we can rteturn
                // this from out function
                // use drain to make sure we don't double up on memory
                if !output.is_empty() {
                    match debug {
                        Debug::Verbose1 => { 
                            println!("Received some work product from the CPUs: {:?}", 
                                output.iter().map(|(m, _, _)| (m.agent_id, m.task_id)).collect::<Vec<(i32, i32)>>());
                        }
                        _ => { }
                    }
                    received_count += output.len();
                    output.drain(..).for_each(|(m, pi, r)| {
                        //let v = results.get_mut(&m.task_id).unwrap();
                        //v[m.agent_id as usize] = Some((m.agent_id,pi, r));
                        M[m.task_id as usize * num_agents + m.agent_id as usize] = 
                            (r * 1_000_000.0) as i32;
                        Pi.insert((m.agent_id, m.task_id), pi);
                        models_return.push(m);
                    });
                    let cpu_new_data: Vec<MOProductMDP<S>> = 
                        models.drain(..std::cmp::min(CPU_COUNT, models.len())).collect();
                    
                    match debug {
                        Debug::Verbose1 => { 
                            println!("Sending some more work to the CPU: {:?}", 
                                cpu_new_data.iter().map(|m| (m.agent_id, m.task_id)).collect::<Vec<(i32, i32)>>());
                        }
                        _ => { }
                    }
                        
                    if !cpu_new_data.is_empty() {
                        cpu_s1.send(
                            CtrlMsg::CPUData(cpu_new_data)
                        ).unwrap();
                    }
                        
                }
            }
            Err(_) => { }
        }
    }

    gpu_s1.send(CtrlMsg::Quit).unwrap();
    cpu_s1.send(CtrlMsg::Quit).unwrap();
    
    gpu_thread.join().unwrap();
    cpu_thread.join().unwrap();

    match debug {
        Debug::Verbose1 | Debug::Verbose2  => { 
            println!("Time to do stage 1: {:?}", t1.elapsed().as_secs_f32())
        }
        _ => { }
    }

    (models_return, M, Pi)
}

pub fn hybrid_stage2(
    mut allocation: Vec<(i32, i32, CsMatBase<f32, i32, Vec<i32>, Vec<i32>, Vec<f32>>,
        CsMatBase<f32, i32, Vec<i32>, Vec<i32>, Vec<f32>>, usize)>,
    epsilon: f32,
    nobjs: usize,
    num_agents: usize,
    num_tasks: usize,
    CPU_COUNT: usize,
    debug: crate::Debug,
    max_iter: usize,
    max_unstable: i32
) -> Vec<f32> {
    // First step we need the channels for passing messages between the threads
    let (gpu_s1, gpu_r1): (Sender<MOCtrlMsg>, Receiver<MOCtrlMsg>) = unbounded();
    let (gpu_s2, gpu_r2): (Sender<(i32, i32, usize, Vec<f32>, usize)>, 
                           Receiver<(i32, i32, usize, Vec<f32>, usize)>) = unbounded();

    let (cpu_s1, cpu_r1): (Sender<MOCtrlMsg>, Receiver<MOCtrlMsg>) = unbounded();
    let (cpu_s2, cpu_r2): (Sender<Vec<(i32, i32, usize, Vec<f32>, usize)>>, 
                           Receiver<Vec<(i32, i32, usize, Vec<f32>, usize)>>) = unbounded();
    
    let t1 = Instant::now();

    let gpu_thread = thread::spawn(move || {
        loop {
            match gpu_r1.try_recv() {
                Ok(data) => {
                    match data {
                        MOCtrlMsg::Quit => { break; },
                        MOCtrlMsg::GPUMOData((a, t, P, R, init)) => {
                            match debug {
                                Debug::Verbose1 => { 
                                    println!("GPU received MO data: [{},{}]", a, t);
                                }
                                _ => { }
                            }
                            let r = cuda_multi_obj_solution(
                                P.view(), 
                                R.view(), 
                                epsilon, 
                                nobjs as i32,
                                max_iter as i32,
                                max_unstable
                            );
                            // return the data back to the main thread and wait for
                            // more data
                            // channel will block until the message is sent
                            let s = P.shape().0;
                            gpu_s2.send((a, t, s, r, init)).unwrap();
                        }
                        _ => {}
                    }
                }
                Err(_) => { }
            }
        }
        match debug {
            Debug::Verbose1 | Debug::Verbose2 => {
                let msg = "GPU controller thread closed successfully";
                println!("{:?}", msg);
            }
            _ => { }
        }
    });

    let cpu_thread  = thread::spawn(move || {
        loop {
            match cpu_r1.try_recv() {
                Ok(data) => {
                    match data {
                        MOCtrlMsg::Quit => { break; }
                        MOCtrlMsg::CPUMOData(v) => {
                            match debug {
                                Debug::Verbose1 | Debug::Verbose2 => { 
                                    println!("CPU received MO data: {:?}", 
                                        v.iter().map(|(a, t, _, _, _)| (*a, *t)).collect::<Vec<(i32, i32)>>())
                                }
                                _ => { }
                            }
                            let output: Vec<(i32, i32, usize, Vec<f32>, usize)> = v.into_par_iter()
                                .map(|(a, t, P, R, init)| {
                                let r = optimal_values(P.view(), R.view(), epsilon, nobjs, max_iter, max_unstable);   
                                let s = P.shape().0;
                                (a, t, s, r, init)
                            }).collect();
                            cpu_s2.send(output).unwrap();
                        }
                        _ => { }
                    }
                }
                Err(_) => { }
            }
        }
        match debug {
            Debug::Verbose1 | Debug::Verbose2 => { 
                let msg = "CPU controller thread closed successfully";
                println!("{:?}", msg);
            }
            _ => { }
        }
    });

    let mut received_count = 0;

    gpu_s1.send(MOCtrlMsg::GPUMOData(allocation.pop().unwrap())).unwrap();
    cpu_s1.send(MOCtrlMsg::CPUMOData(
        allocation.drain(..std::cmp::min(CPU_COUNT, allocation.len())).collect()
    )).unwrap();

    let mut r_: Vec<f32> = vec![0.; nobjs];

    while received_count < num_tasks {
        match gpu_r2.try_recv() {
            Ok((a, t, s, r, init)) => {
                match debug {
                    Debug::Verbose1 => { 
                        println!("Received some MO work product from the GPU");
                    }
                    _ => { }
                }
                received_count += 1;
                r_[a as usize] += r[(a as usize) * s + init];
                let kt = num_agents + t as usize;
                r_[num_agents + t as usize] += r[kt * s + init];
                if !allocation.is_empty() {
                    gpu_s1.send(
                        MOCtrlMsg::GPUMOData(allocation.pop().unwrap())
                    ).unwrap();
                }
            }
            Err(_) => { 

            }
        }

        match cpu_r2.try_recv() {
            Ok(mut v) => { 
                if !v.is_empty() {
                    match debug {
                        Debug::Verbose1 => { 
                            println!("Received some work product from the CPUs");
                        }
                        _ => { }
                    }
                    received_count += v.len();
                    v.drain(..).for_each(|(a, t, s, r, init)| {
                        r_[a as usize] += r[(a as usize) * s + init];
                        let kt = num_agents + t as usize;
                        r_[num_agents + t as usize] += r[kt * s + init];
                    });
                    let cpu_new_data: Vec<(i32, i32,
                        CsMatBase<f32, i32, Vec<i32>, Vec<i32>, Vec<f32>>, 
                        CsMatBase<f32, i32, Vec<i32>, Vec<i32>, Vec<f32>>,
                        usize)> = allocation
                            .drain(..std::cmp::min(CPU_COUNT, allocation.len()))
                            .collect();
                    match debug {
                        Debug::Verbose1 => { 
                            println!("Sending some more work to the CPU"); 
                        }
                        _ => { }
                    }
                    
                    if !cpu_new_data.is_empty() {
                        cpu_s1.send(
                            MOCtrlMsg::CPUMOData(cpu_new_data)
                        ).unwrap();
                    }
                }
            }
            Err(_) => { 

            }
        }
    }

    gpu_s1.send(MOCtrlMsg::Quit).unwrap();
    cpu_s1.send(MOCtrlMsg::Quit).unwrap();
    
    gpu_thread.join().unwrap();
    cpu_thread.join().unwrap();

    match debug {
        Debug::Verbose1 | Debug::Verbose2 => { 
            println!("Time to do stage 2: {:?}", t1.elapsed().as_secs_f32())
        }
        _ => { }
    }
    r_

}