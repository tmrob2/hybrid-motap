use crate::model::centralised::CTMDP;
use crate::model::general::ModelFns;
//use crate::envs::example::Example;
use crate::model::momdp::MOProductMDP;
use crate::sparse::argmax::argmaxM;
use crate::algorithms::dp::{initial_policy, optimal_policy, optimal_values};
use crate::algorithms::hybrid::{hybrid_stage2, hybrid_stage1};
use hungarian::minimize;
use std::time::Instant;
use std::hash::Hash;
use crate::Debug;
use rayon::prelude::*;
use sprs::CsMatBase;
use hashbrown::HashMap;
use crate::{choose_random_policy, 
    cuda_initial_policy_value_pinned_graph,
    cuda_multi_obj_solution
};
/*
-------------------------------------------------------------------
|                              SOLVERS                            |
|                                                                 |
-------------------------------------------------------------------
*/

pub fn hybrid_solver<S>(
    output: Vec<MOProductMDP<S>>,
    num_agents: usize,
    num_tasks: usize,
    w: &[f32],
    eps: f32,
    CPU_COUNT: usize, 
    debug: Debug,
    max_iter: usize,
    max_unstable: i32
) -> (Vec<f32>, Vec<MOProductMDP<S>>, f32)
where S: Copy + Clone + std::fmt::Debug + Eq + Hash + Send + Sync + 'static {
    let t1 = Instant::now();
    let (models, M, Pi) = hybrid_stage1(
        output, num_agents, num_tasks, w.to_vec(), eps, CPU_COUNT, debug, 
        max_iter, max_unstable
    );
    let assignment = 
        minimize(&M, num_tasks, num_agents);

    match debug {
        Debug::Verbose2 => {
            println!("Allocation matrix");
            for task in 0..num_tasks {
                for agent in 0..num_agents {
                    if agent == num_agents - 1 {
                        print!("{}\n", -M[task * num_tasks + agent]);
                    } else {
                        print!("{}, ", -M[task * num_tasks + agent]);
                    }
                }
            }
            println!("End allocation matrix");
            println!("assignment:\n{:?}", assignment);
        }
        _ => { }   
    }
    // The assignment needs to be transformed into something that we can process
    // i.e. for each allocated task construct a vector which is (a, t, pi)
    let allocation: Vec<(i32, i32, Vec<i32>)> = assignment.iter()
        .enumerate()
        .filter(|(_i, x)| x.is_some())
        .map(|(i, x)| 
            (x.unwrap() as i32, i as i32, Pi.get(&(x.unwrap() as i32, i as i32)).unwrap().to_owned())
        ).collect();

    // Then for each allocation we need to make the argmax P, R matrices
    let allocatedArgMax: Vec<(i32, i32, CsMatBase<f32, i32, Vec<i32>, Vec<i32>, Vec<f32>>,
                              CsMatBase<f32, i32, Vec<i32>, Vec<i32>, Vec<f32>>, usize)> 
        = allocation.into_par_iter().map(|(a, t, pi)| {
        let pmdp: &MOProductMDP<S> = models.iter()
            .filter(|m| m.agent_id == a && m.task_id == t)
            .collect::<Vec<&MOProductMDP<S>>>()[0];
        let rowblock = pmdp.states.len() as i32;
        let pcolblock = rowblock as i32;
        let rcolblock = (num_agents + num_tasks) as i32;
        let argmaxP = 
            argmaxM(pmdp.P.view(), &pi, rowblock, pcolblock, &pmdp.adjusted_state_act_pair);
        let argmaxR = 
            argmaxM(pmdp.R.view(), &pi, rowblock, rcolblock, &pmdp.adjusted_state_act_pair);
        let init = *pmdp.state_map.get(&pmdp.initial_state).unwrap();
        (a, t, argmaxP, argmaxR, init)
    }).collect();
    let nobjs = num_agents + num_tasks;
    let returns = hybrid_stage2(
        allocatedArgMax, eps, nobjs, 
        num_agents, num_tasks, CPU_COUNT, debug, max_iter, max_unstable);
    
    //println!("result: {:?}", returns);
    (returns, models, t1.elapsed().as_secs_f32())
}

pub fn gpu_only_solver<S>(
    models_ls: &[MOProductMDP<S>],
    num_agents: usize,
    num_tasks: usize,
    w: &[f32],
    eps: f32,
    debug: Debug,
    max_iter: i32, 
    max_unstable: i32
) -> (Vec<f32>, f32)
where S: Copy + std::fmt::Debug + Eq + Hash + Send + Sync + 'static {
    
    
    let mut w_init = vec![0.; num_agents + num_tasks];
    for k in 0..num_agents {
        w_init[k] = 1.;
    }
    
    let t2 = Instant::now();
    //let mut results: HashMap<i32, Vec<Option<(i32, Vec<i32>, f32)>>> = HashMap::new();
    let mut M: Vec<i32> = vec![0; num_agents * num_tasks];
    let mut Pi: HashMap<(i32, i32), Vec<i32>> = HashMap::new();
    for pmdp in models_ls.iter() {
        let enabled_actions = pmdp.get_enabled_actions();
        let state_size = pmdp.get_states().len();
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
            max_iter, 
            max_unstable
        );
        
        M[pmdp.task_id as usize * num_agents + pmdp.agent_id as usize] = 
            (x_init[*pmdp.state_map.get(&pmdp.initial_state).unwrap()] * 1_000_000.0) as i32;
        Pi.insert((pmdp.agent_id, pmdp.task_id), pi.to_owned());
    }

    let assignment = 
        minimize(&M, num_tasks, num_agents);

    match debug {
        Debug::Verbose2 => {
            println!("Allocation matrix");
            for task in 0..num_tasks {
                for agent in 0..num_agents {
                    if agent == num_agents - 1 {
                        print!("{}\n", -M[task * num_tasks + agent]);
                    } else {
                        print!("{}, ", -M[task * num_tasks + agent]);
                    }
                }
            }
            println!("End allocation matrix");
            println!("assignment:\n{:?}", assignment);
        }
        _ => { }   
    }

    match debug {
        Debug::Verbose1 | Debug::Verbose2 => { 
            println!("Time to do stage 1 {}", t2.elapsed().as_secs_f32());
        }
        _ => { }
    }

    let allocation: Vec<(i32, i32, Vec<i32>)> = assignment.iter()
        .enumerate()
        .filter(|(_i, x)| x.is_some())
        .map(|(i, x)| 
            (x.unwrap() as i32, i as i32, Pi.get(&(x.unwrap() as i32, i as i32)).unwrap().to_owned())
        ).collect();

    // Then for each allocation we need to make the argmax P, R matrices
    let allocatedArgMax: Vec<(i32, i32, CsMatBase<f32, i32, Vec<i32>, Vec<i32>, Vec<f32>>,
                              CsMatBase<f32, i32, Vec<i32>, Vec<i32>, Vec<f32>>, usize)> 
        = allocation.into_par_iter().map(|(a, t, pi)| {
        let pmdp: &MOProductMDP<S> = models_ls.iter()
            .filter(|m| m.agent_id == a && m.task_id == t)
            .collect::<Vec<&MOProductMDP<S>>>()[0];
        let rowblock = pmdp.states.len() as i32;
        let pcolblock = rowblock as i32;
        let rcolblock = (num_agents + num_tasks) as i32;
        let argmaxP = 
            argmaxM(pmdp.P.view(), &pi, rowblock, pcolblock, &pmdp.adjusted_state_act_pair);
        let argmaxR = 
            argmaxM(pmdp.R.view(), &pi, rowblock, rcolblock, &pmdp.adjusted_state_act_pair);
        let init = *pmdp.state_map.get(&pmdp.initial_state).unwrap();
        (a, t, argmaxP, argmaxR, init)
    }).collect();
    let nobjs = num_agents + num_tasks;
    let mut returns: Vec<f32> = vec![0.; nobjs];
    for (a, t, P, R, init) in allocatedArgMax.into_iter() {
        let r = cuda_multi_obj_solution(P.view(), R.view(), eps, nobjs as i32, max_iter, max_unstable);
        returns[a as usize] += r[(a as usize) * P.shape().0 + init];
        let kt = num_agents + t as usize;
        returns[num_agents + t as usize] += r[kt * P.shape().0 + init];
    }
    (returns, t2.elapsed().as_secs_f32())
}

pub fn cpu_only_solver<S>(
    models_ls: &[MOProductMDP<S>],
    num_agents: usize,
    num_tasks: usize,
    w: &[f32],
    eps: f32,
    debug: Debug,
    max_iter: usize,
    max_unstable: i32
) -> (Vec<f32>, f32)
where S: Copy + std::fmt::Debug + Eq + Hash + Send + Sync + 'static {
    let t2 = Instant::now();
    // Input all of the models into the rayon framework
    //
    let mut output: Vec<(i32, i32, Vec<i32>, f32)> = models_ls.par_iter().map(|pmdp| {
        
        let enabled_actions = pmdp.get_enabled_actions();
        let state_size = pmdp.get_states().len();
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
        
        initial_policy(initP.view(), initR.view(), &w, eps, 
                    &mut r_v, &mut x, &mut y, max_iter, max_unstable);

        // taking the initial policy and the value vector for the initial policy
        // what is the optimal policy
        let mut r_v: Vec<f32> = vec![0.; pmdp.R.shape().0];
        let mut y: Vec<f32> = vec![0.; pmdp.P.shape().0];
        let r = optimal_policy(pmdp.P.view(), pmdp.R.view(), &w, eps, 
                    &mut r_v, &mut x, &mut y, &mut pi, 
                    &pmdp.enabled_actions, &pmdp.adjusted_state_act_pair,
                    *pmdp.state_map.get(&pmdp.initial_state).unwrap(),
                    max_iter);
        (pmdp.agent_id, pmdp.task_id, pi, r)
    }).collect();
    let mut M: Vec<i32> = vec![0; num_agents * num_tasks];
    let mut Pi: HashMap<(i32, i32), Vec<i32>> = HashMap::new();
    for (i,j,pi,r) in output.drain(..) {
        M[j as usize * num_agents + i as usize] = (r * 1_000_000.0) as i32;
        Pi.insert((i,j), pi);
    }
    
    let assignment = 
        minimize(&M, num_tasks, num_agents);

    match debug {
        Debug::Verbose2 => {
            println!("Allocation matrix");
            for task in 0..num_tasks {
                for agent in 0..num_agents {
                    if agent == num_agents - 1 {
                        print!("{}\n", -M[task * num_tasks + agent]);
                    } else {
                        print!("{}, ", -M[task * num_tasks + agent]);
                    }
                }
            }
            println!("End allocation matrix");
            println!("assignment:\n{:?}", assignment);
        }
        _ => { }   
    }
    // The assignment needs to be transformed into something that we can process
    // i.e. for each allocated task construct a vector which is (a, t, pi)
    let allocation: Vec<(i32, i32, Vec<i32>)> = assignment.iter()
        .enumerate()
        .filter(|(_i, x)| x.is_some())
        .map(|(i, x)| 
            (x.unwrap() as i32, i as i32, Pi.get(&(x.unwrap() as i32, i as i32)).unwrap().to_owned())
        ).collect();
    
    match debug {
        Debug::Verbose1 | Debug::Verbose2 => {
            println!("Time to do stage 1 {}", t2.elapsed().as_secs_f32());
        }
        _ => { }
    }
    
    let allocatedArgMax: Vec<(i32, i32, CsMatBase<f32, i32, Vec<i32>, Vec<i32>, Vec<f32>>,
                              CsMatBase<f32, i32, Vec<i32>, Vec<i32>, Vec<f32>>, usize)> 
        = allocation.into_par_iter().map(|(a, t, pi)| {
        let pmdp: &MOProductMDP<S> = models_ls.iter()
            .filter(|m| m.agent_id == a && m.task_id == t)
            .collect::<Vec<&MOProductMDP<S>>>()[0];
        let rowblock = pmdp.states.len() as i32;
        let pcolblock = rowblock as i32;
        let rcolblock = (num_agents + num_tasks) as i32;
        let argmaxP = 
            argmaxM(pmdp.P.view(), &pi, rowblock, pcolblock, &pmdp.adjusted_state_act_pair);
        let argmaxR = 
            argmaxM(pmdp.R.view(), &pi, rowblock, rcolblock, &pmdp.adjusted_state_act_pair);
        let init = *pmdp.state_map.get(&pmdp.initial_state).unwrap();
        (a, t, argmaxP, argmaxR, init)
    }).collect();
    let nobjs = num_agents + num_tasks;
    let mut returns: Vec<f32> = vec![0.; nobjs];
    let ret_ :Vec<(i32, i32, f32, f32)> = allocatedArgMax.into_par_iter().map(|(a, t, P, R, init)| {
        let r = optimal_values(P.view(), R.view(), eps, nobjs, max_iter, max_unstable);
        let kt = num_agents + t as usize;
        (a, t, r[(a as usize) * P.shape().0 + init], r[kt * P.shape().0 + init])
    }).collect();

    for (a, t, a_cost, t_prob) in ret_.into_iter() {
        returns[a as usize] += a_cost;
        returns[num_agents + t as usize] += t_prob;
    }
    
    (returns, t2.elapsed().as_secs_f32())
}

pub fn single_cpu_solver<S>(
    models_ls: Vec<MOProductMDP<S>>,
    num_agents: usize,
    num_tasks: usize,
    w: &[f32],
    eps: f32,
    debug: Debug,
    max_iter: usize,
    max_unstable: i32
) -> Vec<f32>
where S: Copy + std::fmt::Debug + Eq + Hash + Send + Sync + 'static {
    let t2 = Instant::now();
    // Input all of the models into the rayon framework
    //
    let mut output: Vec<(i32, i32, Vec<i32>, f32)> = models_ls.iter().map(|pmdp| {
        let enabled_actions = pmdp.get_enabled_actions();
        let state_size = pmdp.get_states().len();
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
        
        initial_policy(initP.view(), initR.view(), &w, eps, 
                    &mut r_v, &mut x, &mut y, max_iter, max_unstable);

        // taking the initial policy and the value vector for the initial policy
        // what is the optimal policy
        let mut r_v: Vec<f32> = vec![0.; pmdp.R.shape().0];
        let mut y: Vec<f32> = vec![0.; pmdp.P.shape().0];
        let r = optimal_policy(pmdp.P.view(), pmdp.R.view(), &w, eps, 
                    &mut r_v, &mut x, &mut y, &mut pi, 
                    &pmdp.enabled_actions, &pmdp.adjusted_state_act_pair,
                    *pmdp.state_map.get(&pmdp.initial_state).unwrap(),
                    max_iter);
        (pmdp.agent_id, pmdp.task_id, pi, r)
    }).collect();
    let mut M: Vec<i32> = vec![0; num_agents * num_tasks];
    let mut Pi: HashMap<(i32, i32), Vec<i32>> = HashMap::new();
    for (i,j,pi,r) in output.drain(..) {
        M[j as usize * num_agents + i as usize] = (r * 1_000_000.0) as i32;
        Pi.insert((i,j), pi);
    }
    
    let assignment = 
        minimize(&M, num_tasks, num_agents);

    match debug {
        Debug::Verbose2 => {
            println!("Allocation matrix");
            for task in 0..num_tasks {
                for agent in 0..num_agents {
                    if agent == num_agents - 1 {
                        print!("{}\n", -M[task * num_tasks + agent]);
                    } else {
                        print!("{}, ", -M[task * num_tasks + agent]);
                    }
                }
            }
            println!("End allocation matrix");
            println!("assignment:\n{:?}", assignment);
        }
        _ => { }   
    }
    // The assignment needs to be transformed into something that we can process
    // i.e. for each allocated task construct a vector which is (a, t, pi)
    let allocation: Vec<(i32, i32, Vec<i32>)> = assignment.iter()
        .enumerate()
        .filter(|(_i, x)| x.is_some())
        .map(|(i, x)| 
            (x.unwrap() as i32, i as i32, Pi.get(&(x.unwrap() as i32, i as i32)).unwrap().to_owned())
        ).collect();
    
    match debug {
        Debug::Verbose1 | Debug::Verbose2 => {
            println!("Time to do stage 1 {}", t2.elapsed().as_secs_f32());
        }
        _ => { }
    }
    
    let allocatedArgMax: Vec<(i32, i32, CsMatBase<f32, i32, Vec<i32>, Vec<i32>, Vec<f32>>,
                              CsMatBase<f32, i32, Vec<i32>, Vec<i32>, Vec<f32>>, usize)> 
        = allocation.into_iter().map(|(a, t, pi)| {
        let pmdp: &MOProductMDP<S> = models_ls.iter()
            .filter(|m| m.agent_id == a && m.task_id == t)
            .collect::<Vec<&MOProductMDP<S>>>()[0];
        let rowblock = pmdp.states.len() as i32;
        let pcolblock = rowblock as i32;
        let rcolblock = (num_agents + num_tasks) as i32;
        let argmaxP = 
            argmaxM(pmdp.P.view(), &pi, rowblock, pcolblock, &pmdp.adjusted_state_act_pair);
        let argmaxR = 
            argmaxM(pmdp.R.view(), &pi, rowblock, rcolblock, &pmdp.adjusted_state_act_pair);
        let init = *pmdp.state_map.get(&pmdp.initial_state).unwrap();
        (a, t, argmaxP, argmaxR, init)
    }).collect();
    let nobjs = num_agents + num_tasks;
    let mut returns: Vec<f32> = vec![0.; nobjs];
    for (a, t, P, R, init) in allocatedArgMax.into_iter() {
        let r = optimal_values(P.view(), R.view(), eps, nobjs, max_iter, max_unstable);
        returns[a as usize] += r[(a as usize) * P.shape().0 + init];
        let kt = num_agents + t as usize;
        returns[num_agents + t as usize] += r[kt * P.shape().0 + init];
    }
    
    returns
}

pub fn ctmdp_cpu_solver<S>(
    ctmdp: &CTMDP<S>,
    num_agents: usize,
    num_tasks: usize,
    w: &[f32],
    eps: f32,
    debug: Debug,
    max_iter: usize,
    max_unstable: i32
) -> (Vec<f32>, f32)
where S: Copy + std::fmt::Debug + Eq + Hash + Send + Sync + 'static {
    let elapsed_time: f32;
    let t2 = Instant::now();
    // Input all of the models into the rayon framework
    //
    let enabled_actions = ctmdp.get_enabled_actions();
    let state_size = ctmdp.get_states().len();
    let mut pi = choose_random_policy(state_size, enabled_actions);

    let rowblock = ctmdp.states.len() as i32;
    let pcolblock = rowblock as i32;
    let rcolblock = (num_agents + num_tasks) as i32;
    let initP = 
        argmaxM(ctmdp.P.view(), &pi, rowblock, pcolblock, 
                &ctmdp.adjusted_state_act_pair);
    let initR = 
        argmaxM(ctmdp.R.view(), &pi, rowblock, rcolblock, 
                &ctmdp.adjusted_state_act_pair);

    //println!("init P shape: {:?}", initP.shape());

    let mut r_v: Vec<f32> = vec![0.; initR.shape().0];
    let mut x: Vec<f32> = vec![0.; initP.shape().1];
    let mut y: Vec<f32> = vec![0.; initP.shape().0];

    let wagent = vec![1.; num_agents]; 
    let wtask = vec![0.; num_tasks];
    let winit = [&wagent[..], &wtask[..]].concat();
    
    initial_policy(initP.view(), initR.view(), &winit, eps, 
                   &mut r_v, &mut x, &mut y, max_iter, max_unstable);

    //println!("init x: {:?}", x);

    // taking the initial policy and the value vector for the initial policy
    // what is the optimal policy
    let mut r_v: Vec<f32> = vec![0.; ctmdp.R.shape().0];
    let mut y: Vec<f32> = vec![0.; ctmdp.P.shape().0];
    optimal_policy(ctmdp.P.view(), ctmdp.R.view(), &w, eps, 
        &mut r_v, &mut x, &mut y, &mut pi, 
        &ctmdp.enabled_actions, &ctmdp.adjusted_state_act_pair,
        *ctmdp.state_map.get(&ctmdp.initial_state).unwrap(),
        max_iter
    );
    
    match debug {
        Debug::Verbose1 | Debug::Verbose2 => {
            println!("Time to do stage 1 {}", t2.elapsed().as_secs_f32());
        }
        _ => { }
    }

    let rowblock = ctmdp.states.len() as i32;
    let pcolblock = rowblock as i32;
    let rcolblock = (num_agents + num_tasks) as i32;
    let argmaxP = 
        argmaxM(ctmdp.P.view(), &pi, rowblock, pcolblock, &ctmdp.adjusted_state_act_pair);
    let argmaxR = 
        argmaxM(ctmdp.R.view(), &pi, rowblock, rcolblock, &ctmdp.adjusted_state_act_pair);
    let init = *ctmdp.state_map.get(&ctmdp.initial_state).unwrap();
    //println!("R: \n{:?}", argmaxR.to_dense());
    let nobjs = num_agents + num_tasks;
    let r = optimal_values(argmaxP.view(), argmaxR.view(), eps, nobjs, max_iter, max_unstable);
    //println!("r: {:?}", r);
    let mut returns = vec![0.; nobjs];
    for k in 0..nobjs {
        returns[k] = r[k * argmaxP.shape().0 + init];
        //println!("Objective: {}\n{:.2?}", k, &r[k * argmaxP.shape().0..(k + 1) * argmaxP.shape().0])
    }
    elapsed_time = t2.elapsed().as_secs_f32();
    match debug {
        Debug::Verbose1 | Debug::Verbose2 => {
            println!("Time to do stage 1 + 2 {}", elapsed_time);
        }
        _ => { }
    }
    (returns, elapsed_time)
}

pub fn ctmdp_gpu_solver<S>(
    ctmdp: &CTMDP<S>,
    num_agents: usize,
    num_tasks: usize,
    w: &[f32],
    eps: f32,
    debug: Debug,
    max_iter: i32,
    max_unstable: i32
) -> (Vec<f32>, f32)
where S: Copy + std::fmt::Debug + Eq + Hash + Send + Sync + 'static {
    let elapsed_time: f32;
    let t2 = Instant::now();
    // Input all of the models into the rayon framework
    //
    let enabled_actions = ctmdp.get_enabled_actions();
    let state_size = ctmdp.get_states().len();
    let mut pi = choose_random_policy(state_size, enabled_actions);

    let rowblock = ctmdp.states.len() as i32;
    let pcolblock = rowblock as i32;
    let rcolblock = (num_agents + num_tasks) as i32;
    let initP = 
        argmaxM(ctmdp.P.view(), &pi, rowblock, pcolblock, 
                &ctmdp.adjusted_state_act_pair);
    let initR = 
        argmaxM(ctmdp.R.view(), &pi, rowblock, rcolblock, 
                &ctmdp.adjusted_state_act_pair);

    let wagent = vec![1.; num_agents]; 
    let wtask = vec![0.; num_tasks];
    let winit = [&wagent[..], &wtask[..]].concat();

    //println!("init P shape: {:?}", initP.shape());

    let mut r_v_init: Vec<f32> = vec![0.; initR.shape().0 as usize];
    let mut x_init: Vec<f32> = vec![0.; initP.shape().1 as usize];
    let mut y_init: Vec<f32> = vec![0.; initP.shape().0 as usize];
    let mut unstable: Vec<i32> = vec![0; initP.shape().0 as usize];
    let mut stable: Vec<f32> = vec![0.; x_init.len()];
    let mut y: Vec<f32> = vec![0.; ctmdp.P.shape().0];
    let mut rmv: Vec<f32> = vec![0.; ctmdp.P.shape().0];

    cuda_initial_policy_value_pinned_graph(
        initP.view(), 
        initR.view(), 
        ctmdp.P.view(), 
        ctmdp.R.view(), 
        &winit,
        &w, 
        eps, 
        &mut x_init, 
        &mut y_init, 
        &mut r_v_init, 
        &mut y, 
        &mut rmv, 
        &mut unstable, 
        &mut pi, 
        &ctmdp.enabled_actions, 
        &ctmdp.adjusted_state_act_pair,
        &mut stable,
        max_iter, 
        max_unstable
    );

    let rowblock = ctmdp.states.len() as i32;
    let pcolblock = rowblock as i32;
    let rcolblock = (num_agents + num_tasks) as i32;
    let argmaxP = 
        argmaxM(ctmdp.P.view(), &pi, rowblock, pcolblock, &ctmdp.adjusted_state_act_pair);
    let argmaxR = 
        argmaxM(ctmdp.R.view(), &pi, rowblock, rcolblock, &ctmdp.adjusted_state_act_pair);
    let init = *ctmdp.state_map.get(&ctmdp.initial_state).unwrap();
    //println!("R: \n{:?}", argmaxR.to_dense());
    let nobjs = num_agents + num_tasks;
    let r = cuda_multi_obj_solution(argmaxP.view(), argmaxR.view(), eps, nobjs as i32, max_iter, max_unstable);
    //println!("r: {:?}", r);
    let mut returns = vec![0.; nobjs];
    for k in 0..nobjs {
        returns[k] = r[k * argmaxP.shape().0 + init];
        //println!("Objective: {}\n{:.2?}", k, &r[k * argmaxP.shape().0..(k + 1) * argmaxP.shape().0])
    }
    elapsed_time = t2.elapsed().as_secs_f32();
    match debug {
        Debug::Verbose1 | Debug::Verbose2 => {
            println!("Time to do stage 1 + 2 {}", elapsed_time);
        }
        _ => { }
    }
    (returns, elapsed_time)
}