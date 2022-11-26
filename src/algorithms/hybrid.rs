use crate::model::scpm::SCPM;
use crate::agent::env::Env;
use crate::product;
use crate::sparse::argmax::argmaxM;
use crate::model::momdp::{MOProductMDP, choose_random_policy};
use crossbeam_channel::{unbounded, Receiver, Sender};
use std::thread;
use std::time::Duration;
use rayon::prelude::*;
use super::dp::{initial_policy, optimal_policy};

pub enum ControlMessage<S> {
    Quit,
    Data(MOProductMDP<S>),
    CPUData(Vec<MOProductMDP<S>>)
}

// TODO We need a smarter way of dealing with models without making a clone of 
// all of the models
pub fn hybrid_stage1<S>(
    mut models: Vec<MOProductMDP<S>>,
    num_agents: usize,
    num_tasks: usize,
    w: Vec<f32>, 
    epsilon: f32,
    CPU_COUNT: usize
) -> Vec<(MOProductMDP<S>, f32)>
where S: Copy + Clone + std::fmt::Debug + Eq + std::hash::Hash + 'static, 
      MOProductMDP<S>: Send + Clone {

    let mut model_return: Vec<(MOProductMDP<S>, f32)> = Vec::new();
    // First step of the test is to construct an initial random policy
    // which can be chosen directly from the action vector which contains
    // the number of enabled actions in each of the states. 
    //let t1 = Instant::now();
    // construct the product of all of the model, env pairs
    //let mut _v: Vec<MOProductMDP<State>> = Vec::with_capacity(model.num_agents * model.tasks.size);
    let mut w_init = vec![0.; num_agents + num_tasks];
    
    let (gpu_s1, gpu_r1) : (Sender<ControlMessage<S>>, Receiver<ControlMessage<S>>) = unbounded();
    let (gpu_s2, gpu_r2) : (Sender<(MOProductMDP<S>, f32)>, Receiver<(MOProductMDP<S>, f32)>) = unbounded();

    let (cpu_s1, cpu_r1) : (Sender<ControlMessage<S>>, Receiver<ControlMessage<S>>) = unbounded();
    let (cpu_s2, cpu_r2) : (Sender<Vec<(MOProductMDP<S>, f32)>>, Receiver<Vec<(MOProductMDP<S>, f32)>>) = unbounded();

    let gpu_thread = thread::spawn(move || {
        loop {
            match gpu_r1.try_recv() {
                Ok(data) => { 
                    match data {
                        ControlMessage::Quit => { break; }
                        // send the list items to oblivion
                        ControlMessage::Data(m) => { 

                            // TODO we need to store the current sum of modified stat space size

                            println!("Received data: ({},{})", m.agent_id, m.task_id);
                            // send the data to CUDA to be processed and then returned to this thread
                            // so that we can send the processed data back to the main thread.                            
                            thread::sleep(Duration::from_secs(1));
                            gpu_s2.send((m, 1.0)).unwrap();
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
    let cpu_thread = thread::spawn(move || {
        // continuously loop and wait for data to be sent to the CPU for allocation
        loop {
            match cpu_r1.try_recv() {
                Ok(data) => { 
                    match data {
                        ControlMessage::Quit => { break; }
                        // send the list items to oblivion
                        ControlMessage::CPUData(model_ls) => { 

                            // do the Rayon allocation of threads. 
                            println!("CPUs Received data: {} models", model_ls.len());

                            // TODO we need to store the current sum of modified stat space size
                            let output: Vec<(MOProductMDP<S>, f32)> = model_ls.into_par_iter().map(|pmdp| {

                                let mut pi = choose_random_policy(&pmdp);

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
                                (pmdp, r)
                            }).collect();

                            // send the data to CUDA to be processed and then returned to this thread
                            // so that we can send the processed data back to the main thread.                            
                            thread::sleep(Duration::from_secs(2));
                            cpu_s2.send(output).unwrap();
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
    gpu_s1.send(ControlMessage::Data(models.pop().unwrap())).unwrap();
    cpu_s1.send(ControlMessage::CPUData(
        models.drain(..std::cmp::min(CPU_COUNT, models.len())).collect())
    ).unwrap();
    while models.len() > 0 {
        // First try and allocate the data to the GPU, if the GPU is free
        // Otherwise try and allocate the data to the CPU if they are free
        // If no device is free, then continue looping until one of the devices
        // becomes free.
        match gpu_r2.try_recv() {
            Ok((gpu_model, data)) => { 
                println!("Received some work product from the GPU: {:?}", data);
                let gpu_new_data = models.pop().unwrap();
                println!("Sending some more work to the GPU: id [{},{}]", gpu_new_data.agent_id, gpu_new_data.task_id);
                gpu_s1.send(ControlMessage::Data(gpu_new_data)).unwrap();
                // TODO call GPU work here. 
                model_return.push((gpu_model, data));
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
                output.drain(..).for_each(|m| model_return.push(m));
                println!("Received some work product from the CPUs");
                let cpu_new_data: Vec<MOProductMDP<S>> = models.drain(..CPU_COUNT).collect();
                println!("Sending some more work to the CPU: {:?}", 
                    cpu_new_data.iter().map(|m| (m.agent_id, m.task_id)).collect::<Vec<(i32, i32)>>());
                cpu_s1.send(ControlMessage::CPUData(cpu_new_data)).unwrap();
            }
            Err(_) => { }
        }
    }

    gpu_s1.send(ControlMessage::Quit).unwrap();
    cpu_s1.send(ControlMessage::Quit).unwrap();

    println!("{:?}", cpu_thread.join().unwrap());
    println!("{:?}", gpu_thread.join().unwrap());
    
    model_return

}