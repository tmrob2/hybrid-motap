use crate::model::mamdp::MOMAMDP;
use crate::model::momdp::MOProductMDP;
use crate::Debug;
use crate::model::mostapu::STAPU;
use crate::model::scpm::SCPMModel;
use std::time::Instant;
use std::hash::Hash;
use rayon::prelude::*;
use crate::model::general::ModelFns;
use crate::choose_random_policy;
use crate::sparse::argmax::argmaxM;
use crate::algorithms::dp::{initial_policy, optimal_policy, optimal_values};
use hashbrown::HashMap;
use crate::algorithms::allocation::allocation_fn;
use sprs::CsMatBase;

/*
-------------------------------------------------------------------
|                              SOLVERS                            |
|                                                                 |
-------------------------------------------------------------------
*/

pub fn motap_cpu_only_solver<S>(
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
    let mut M: Vec<f32> = vec![0.; num_agents * num_tasks];
    let mut Pi: HashMap<(i32, i32), Vec<i32>> = HashMap::new();
    // The indexing of the cost matrix is important
    for (i,j,pi,r) in output.drain(..) {
        M[j as usize * num_agents + i as usize] = r;
        Pi.insert((i,j), pi);
    }

    //println!("M: \n{:?}", M);

    let t2 = Instant::now();
    let tot_exp_cost = compute_max_tot_exp_cost(&M[..], num_agents, num_tasks);
    //println!("tot exp cost: {}", tot_exp_cost);
    let alloc: Vec<i32>;
    match allocation_fn(&M[..], num_agents, num_tasks, 
        10., tot_exp_cost) {
        Ok(result) => { alloc = result; }
        Err(e) => {  panic!("Error: {:?}", e) }
    };
    let mip_t = t2.elapsed().as_secs_f32();
    //println!("Allocation: {:?}", alloc);
    
    let allocation: Vec<(i32, i32, Vec<i32>)> = alloc.iter()
    .enumerate()
    .map(|(i, x)| 
        (*x, i as i32, Pi.get(&(*x, i as i32)).unwrap().to_owned())
    ).collect();

    match debug {
        Debug::Verbose1 | Debug::Verbose2 => {
            println!("MIP time: {:?}", mip_t);
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

pub fn mamdp_cpu_solver<S>(
    mamdp: &MOMAMDP<S>,
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
    let enabled_actions = mamdp.get_enabled_actions();
    let state_size = mamdp.get_states().len();
    let mut pi = choose_random_policy(state_size, enabled_actions);

    let rowblock = mamdp.states.len() as i32;
    let pcolblock = rowblock;
    let rcolblock = (num_agents + num_tasks) as i32;

    let initP = 
        argmaxM(mamdp.P.view(), &pi, rowblock, pcolblock, 
    &mamdp.adjusted_state_act_pair);
    let initR = 
        argmaxM(mamdp.R.view(), &pi, rowblock, rcolblock, 
        &mamdp.adjusted_state_act_pair);
    
    let mut r_v: Vec<f32> = vec![0.; initR.shape().0];
    let mut x: Vec<f32> = vec![0.; initP.shape().1];
    let mut y: Vec<f32> = vec![0.; initP.shape().0];

    let wagent = vec![1.; num_agents]; 
    let wtask = vec![0.; num_tasks];
    let winit = [&wagent[..], &wtask[..]].concat();
    
    initial_policy(initP.view(), initR.view(), &winit, eps, 
                    &mut r_v, &mut x, &mut y, max_iter, max_unstable);

    let mut r_v: Vec<f32> = vec![0.; mamdp.R.shape().0];
    let mut y: Vec<f32> = vec![0.; mamdp.P.shape().0];
    optimal_policy(mamdp.P.view(), mamdp.R.view(), &w, eps, 
        &mut r_v, &mut x, &mut y, &mut pi, 
        &mamdp.enabled_actions, &mamdp.adjusted_state_act_pair,
        *mamdp.state_map.get(&mamdp.initial_state).unwrap(),
        max_iter
    );
    
    match debug {
        Debug::Verbose1 | Debug::Verbose2 => {
            println!("Time to do stage 1 {}", t2.elapsed().as_secs_f32());
        }
        _ => { }
    }

    let rowblock = mamdp.states.len() as i32;
    let pcolblock = rowblock as i32;
    let rcolblock = (num_agents + num_tasks) as i32;
    let argmaxP = 
        argmaxM(mamdp.P.view(), &pi, rowblock, pcolblock, &mamdp.adjusted_state_act_pair);
    let argmaxR = 
        argmaxM(mamdp.R.view(), &pi, rowblock, rcolblock, &mamdp.adjusted_state_act_pair);
    let init = *mamdp.state_map.get(&mamdp.initial_state).unwrap();
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

pub fn scpm_solver<S>(
    scpm: &SCPMModel<S>,
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
    let enabled_actions = scpm.get_enabled_actions();
    let state_size = scpm.get_states().len();
    let mut pi = choose_random_policy(state_size, enabled_actions);

    let rowblock = scpm.states.len() as i32;
    let pcolblock = rowblock;
    let rcolblock = (num_agents + num_tasks) as i32;

    let initP = 
        argmaxM(scpm.P.view(), &pi, rowblock, pcolblock, 
    &scpm.adjusted_state_act_pair);
    let initR = 
        argmaxM(scpm.R.view(), &pi, rowblock, rcolblock, 
        &scpm.adjusted_state_act_pair);
    
    let mut r_v: Vec<f32> = vec![0.; initR.shape().0];
    let mut x: Vec<f32> = vec![0.; initP.shape().1];
    let mut y: Vec<f32> = vec![0.; initP.shape().0];

    let wagent = vec![1.; num_agents]; 
    let wtask = vec![0.; num_tasks];
    let winit = [&wagent[..], &wtask[..]].concat();
    
    initial_policy(initP.view(), initR.view(), &winit, eps, 
                    &mut r_v, &mut x, &mut y, max_iter, max_unstable);

    let mut r_v: Vec<f32> = vec![0.; scpm.R.shape().0];
    let mut y: Vec<f32> = vec![0.; scpm.P.shape().0];
    optimal_policy(scpm.P.view(), scpm.R.view(), &w, eps, 
        &mut r_v, &mut x, &mut y, &mut pi, 
        &scpm.enabled_actions, &scpm.adjusted_state_act_pair,
        *scpm.state_map.get(&scpm.initial_state).unwrap(),
        max_iter
    );
    
    match debug {
        Debug::Verbose1 | Debug::Verbose2 => {
            println!("Time to do stage 1 {}", t2.elapsed().as_secs_f32());
        }
        _ => { }
    }

    let rowblock = scpm.states.len() as i32;
    let pcolblock = rowblock as i32;
    let rcolblock = (num_agents + num_tasks) as i32;
    let argmaxP = 
        argmaxM(scpm.P.view(), &pi, rowblock, pcolblock, 
        &scpm.adjusted_state_act_pair);
    let argmaxR = 
        argmaxM(scpm.R.view(), &pi, rowblock, rcolblock, 
            &scpm.adjusted_state_act_pair);
    let init = *scpm.state_map.get(&scpm.initial_state).unwrap();
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

pub fn stapu_solver<S>(
    stapu: &STAPU<S>,
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
    let enabled_actions = stapu.get_enabled_actions();
    let state_size = stapu.get_states().len();
    let mut pi = choose_random_policy(state_size, enabled_actions);

    let rowblock = stapu.states.len() as i32;
    let pcolblock = rowblock;
    let rcolblock = (num_agents + num_tasks) as i32;

    let initP = 
        argmaxM(stapu.P.view(), &pi, rowblock, pcolblock, 
    &stapu.adjusted_state_act_pair);
    let initR = 
        argmaxM(stapu.R.view(), &pi, rowblock, rcolblock, 
        &stapu.adjusted_state_act_pair);
    
    let mut r_v: Vec<f32> = vec![0.; initR.shape().0];
    let mut x: Vec<f32> = vec![0.; initP.shape().1];
    let mut y: Vec<f32> = vec![0.; initP.shape().0];

    let wagent = vec![1.; num_agents]; 
    let wtask = vec![0.; num_tasks];
    let winit = [&wagent[..], &wtask[..]].concat();
    
    initial_policy(initP.view(), initR.view(), &winit, eps, 
                    &mut r_v, &mut x, &mut y, max_iter, max_unstable);

    let mut r_v: Vec<f32> = vec![0.; stapu.R.shape().0];
    let mut y: Vec<f32> = vec![0.; stapu.P.shape().0];
    optimal_policy(stapu.P.view(), stapu.R.view(), &w, eps, 
        &mut r_v, &mut x, &mut y, &mut pi, 
        &stapu.enabled_actions, &stapu.adjusted_state_act_pair,
        *stapu.state_map.get(&stapu.initial_state).unwrap(),
        max_iter
    );
    
    match debug {
        Debug::Verbose1 | Debug::Verbose2 => {
            println!("Time to do stage 1 {}", t2.elapsed().as_secs_f32());
        }
        _ => { }
    }

    let rowblock = stapu.states.len() as i32;
    let pcolblock = rowblock as i32;
    let rcolblock = (num_agents + num_tasks) as i32;
    let argmaxP = 
        argmaxM(stapu.P.view(), &pi, rowblock, pcolblock, 
        &stapu.adjusted_state_act_pair);
    let argmaxR = 
        argmaxM(stapu.R.view(), &pi, rowblock, rcolblock, 
            &stapu.adjusted_state_act_pair);
    let init = *stapu.state_map.get(&stapu.initial_state).unwrap();
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

/*
-------------------------------------------------------------------
|                              HELPER Fns                         |
|                                                                 |
-------------------------------------------------------------------
*/

fn compute_max_tot_exp_cost(M: &[f32], num_agents: usize, num_tasks: usize) -> f32 {
    let mut tot_value = 0.0; 
    for t in 0..num_tasks {
        let max_a = M[t * num_agents..(t + 1) * num_agents].iter()
            .fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
        //println!("task: {}, max agent: {}", t, max_a);
        tot_value += max_a;
    }
    tot_value
}
