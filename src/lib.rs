#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
pub mod model;
pub mod agent;
pub mod task;
pub mod envs;
pub mod algorithms;
pub mod sparse;
pub mod tests;
//use envs::{message::MessageSender, warehouse::Warehouse};
use model::scpm::SCPM;
use pyo3::prelude::*;
use sprs::CsMatBase;
use task::dfa::{DFA, Mission};
use envs::message::*;
use hashbrown::HashMap;
use std::{hash::Hash, time::Instant};
use envs::warehouse::*;
use model::momdp::{MOProductMDP, choose_random_policy};
use rayon::prelude::*;
use sparse::argmax::argmaxM;
use algorithms::dp::{initial_policy, optimal_policy, optimal_values};

use crate::algorithms::hybrid::{hybrid_stage2, hybrid_stage1};
use std::io::prelude::*;
use std::fs::File;

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
    debug: Debug
) -> Vec<f32>
where S: Copy + Clone + std::fmt::Debug + Eq + Hash + Send + Sync + 'static {
    let (models, results) = hybrid_stage1(
        output, num_agents, num_tasks, w.to_vec(), eps, CPU_COUNT, debug
    );
    let allocation = allocation_fn(
        &results, num_tasks, num_agents
    );

    // Then for each allocation we need to make the argmax P, R matrices
    let allocatedArgMax: Vec<(i32, i32, CsMatBase<f32, i32, Vec<i32>, Vec<i32>, Vec<f32>>,
                              CsMatBase<f32, i32, Vec<i32>, Vec<i32>, Vec<f32>>, usize)> 
        = allocation.into_par_iter().map(|(t, a, pi)| {
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
        num_agents, num_tasks, CPU_COUNT, debug);
    
    //println!("result: {:?}", returns);
    returns
}

pub fn gpu_only_solver<S>(
    models_ls: Vec<MOProductMDP<S>>,
    num_agents: usize,
    num_tasks: usize,
    w: &[f32],
    eps: f32,
    debug: Debug
) -> Vec<f32>
where S: Copy + std::fmt::Debug + Eq + Hash + Send + Sync + 'static {
    

    let mut w_init = vec![0.; num_agents + num_tasks];
    for k in 0..num_agents {
        w_init[k] = 1.;
    }
    
    let t2 = Instant::now();
    let mut results: HashMap<i32, Vec<Option<(i32, Vec<i32>, f32)>>> = HashMap::new();
    for pmdp in models_ls.iter() {
        let mut pi = choose_random_policy(pmdp);
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
                    (0..num_agents).map(|i| if i as i32 == pmdp.agent_id{
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
    match debug {
        Debug::None => { }
        _ => { 
            println!("Time to do stage 1 {}", t2.elapsed().as_secs_f32());
        }
    }
    let allocation = allocation_fn(
        &results, num_tasks, num_agents
    );

    // Then for each allocation we need to make the argmax P, R matrices
    let allocatedArgMax: Vec<(i32, i32, CsMatBase<f32, i32, Vec<i32>, Vec<i32>, Vec<f32>>,
                              CsMatBase<f32, i32, Vec<i32>, Vec<i32>, Vec<f32>>, usize)> 
        = allocation.into_par_iter().map(|(t, a, pi)| {
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
        let r = cuda_multi_obj_solution(P.view(), R.view(), eps, nobjs as i32);
        returns[a as usize] += r[(a as usize) * P.shape().0 + init];
        let kt = num_agents + t as usize;
        returns[num_agents + t as usize] += r[kt * P.shape().0 + init];
    }
    returns
}

pub fn cpu_only_solver<S>(
    models_ls: Vec<MOProductMDP<S>>,
    num_agents: usize,
    num_tasks: usize,
    w: &[f32],
    eps: f32,
    debug: Debug
) -> Vec<f32>
where S: Copy + std::fmt::Debug + Eq + Hash + Send + Sync + 'static {
    let t2 = Instant::now();
    // Input all of the models into the rayon framework
    //
    let output: Vec<(i32, i32, Vec<i32>, f32)> = models_ls.par_iter().map(|pmdp| {

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
        
        initial_policy(initP.view(), initR.view(), &w, eps, 
                    &mut r_v, &mut x, &mut y);

        // taking the initial policy and the value vector for the initial policy
        // what is the optimal policy
        let mut r_v: Vec<f32> = vec![0.; pmdp.R.shape().0];
        let mut y: Vec<f32> = vec![0.; pmdp.P.shape().0];
        let r = optimal_policy(pmdp.P.view(), pmdp.R.view(), &w, eps, 
                    &mut r_v, &mut x, &mut y, &mut pi, 
                    &pmdp.enabled_actions, &pmdp.adjusted_state_act_pair,
                    *pmdp.state_map.get(&pmdp.initial_state).unwrap()
                    );
        (pmdp.task_id, pmdp.agent_id, pi, r)
    }).collect();
    //let mut M: Vec<f32> = vec![0.; num_agents * num_tasks];
    /*let mut Pi: HashMap<(i32, i32), Vec<i32>> = HashMap::new();
    for (i,j,pi,r) in output.drain(..) {
        M[j as usize * num_agents + i as usize] = r;
        Pi.insert((i,j), pi);
    }*/
    let mut results: HashMap<i32, Vec<Option<(i32, Vec<i32>, f32)>>> = HashMap::new();
    for task in 0..num_tasks {
        output.iter().filter(|(t, _, _, _)| *t == task as i32)
            .for_each(|(t, a, pi, r)| {
            match results.get_mut(t) {
                Some(v) => {
                    v[*a as usize] = Some((*a, pi.to_owned(), *r));
                }
                None => {
                    let mut vnew: Vec<Option<(i32, Vec<i32>, f32)>> = vec![None; num_agents];
                    vnew[*a as usize] = Some((*a, pi.to_owned(), *r));
                    results.insert(*t, vnew);
                }
            }
        });
    }
    let allocation = allocation_fn(
        &results, num_tasks, num_agents
    );
    match debug {
        Debug::None => { }
        _ => {
            println!("Time to do stage 1 {}", t2.elapsed().as_secs_f32());
            println!("Total runtime {}", t2.elapsed().as_secs_f32());
        }
    }
    /*
    let allocatedArgMax: Vec<(i32, i32, CsMatBase<f32, i32, Vec<i32>, Vec<i32>, Vec<f32>>,
                              CsMatBase<f32, i32, Vec<i32>, Vec<i32>, Vec<f32>>, usize)> 
        = allocation.into_par_iter().map(|(t, a, pi)| {
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
        let r = optimal_values(P.view(), R.view(), eps, nobjs);
        returns[a as usize] += r[(a as usize) * P.shape().0 + init];
        let kt = num_agents + t as usize;
        returns[num_agents + t as usize] += r[kt * P.shape().0 + init];
    }
    */
    returns
}

/*
-------------------------------------------------------------------
|                          HELPER FUNCTIONS                       |
|                                                                 |
-------------------------------------------------------------------
*/
#[derive(Copy, Clone)]
pub enum Debug {
    None,
    Base,
    Verbose1,
    Verbose2
}

pub fn debug_level(input: i32) -> Debug {
    let debug_input: Debug = match input {
        1 => Debug::Base,
        2 => Debug::Verbose1,
        3 => Debug::Verbose2,
        _ => Debug::None
    };
    debug_input
}

pub fn reverse_key_value_pairs<T, U>(map: &HashMap<T, U>) -> HashMap<U, T> 
where T: Clone + Hash, U: Clone + Hash + Eq {
    map.into_iter().fold(HashMap::new(), 
        |mut acc, (a, b)| {
        acc.insert(b.clone(), a.clone());
        acc
    })
}

/// x is the output from the initial value vector computation
/// enabled actions are per state i.e. A(s)
/// size is the ajust size of the data structure |S| .A(s) forall s in S
pub fn adjust_value_vector(
    x: &[f32], 
    adjusted_s0: &[i32], 
    enabled_actions: &[i32], 
    size: usize
) -> Vec<f32> {
    let mut z: Vec<f32> = vec![0.; size];
    for k in 0..x.len() {
        for act in 0..enabled_actions[k] {
            z[(adjusted_s0[k] + act) as usize] = x[k];
        }
    }
    z
}

pub fn product(
    r1: std::ops::Range<usize>, 
    r2: std::ops::Range<usize>
) -> Vec<(usize, usize)> {
    let mut v: Vec<(usize, usize)> = Vec::with_capacity(r1.end * r2.end);
    for k1 in r1.start..r1.end {
        for k2 in r2.start..r2.end {
            v.push((k1, k2));
        } 
    }
    v
}

pub fn allocation_fn(
    vi_output: &HashMap<i32, Vec<Option<(i32, Vec<i32>, f32)>>>,
    num_tasks: usize, 
    num_agents: usize
) -> Vec<(i32, i32, Vec<i32>)>{ 
    let mut allocation: Vec<(i32, i32, Vec<i32>)> = Vec::new();
    let mut task_counts_by_agent: Vec<i32> = vec![0; num_agents];
    for task in 0..num_tasks {
        let vals = vi_output.get(&(task as i32)).unwrap();
        let mut min_best_val: Option<(i32, Vec<i32>, f32)> = None;
        for i in 0..num_agents - 1 {
            // cmp the current val with the next val
            if vals[i].is_some() {
                // always swap
                if vals[i + 1].is_some() {
                    // compare the two otherwise move on
                    if (vals[i].as_ref().unwrap().2 + 
                        task_counts_by_agent[vals[i].as_ref().unwrap().0 as usize] as f32) < 
                        (vals[i + 1].as_ref().unwrap().2 + 
                        task_counts_by_agent[vals[i + 1].as_ref().unwrap().0 as usize] as f32){
                        min_best_val = vals[i].clone();
                        break;
                    } else {
                        min_best_val = vals[i + 1].clone();
                    }
                } else {
                    if min_best_val.is_none() {
                        min_best_val = vals[i].clone();
                    }
                }
            }
        }
        task_counts_by_agent[min_best_val.as_ref().unwrap().0 as usize] += 1;
        allocation.push((
            task as i32,
            min_best_val.as_ref().unwrap().0, 
            min_best_val.as_ref().unwrap().1.to_owned()
        ));
    }
    allocation
}

/// Write a PRISM file which represents a Agent x Task pair. 
pub fn prism_file_generator(
    P: CsMatBase<f32, i32, &[i32], &[i32], &[f32]>,
    state_space: usize,
    adjusted_sidx: &[i32],
    enabled_actions: &[i32],
    initial_state: usize
) -> std::io::Result<()> 
{
    let mut buffer = File::create("prism_experiment.mn")?;
    // document heading
    writeln!(buffer, "mdp")?;
    writeln!(buffer)?;
    writeln!(buffer, "module M1")?;
    writeln!(buffer)?;
    writeln!(buffer, "\tx: [0..{}] init {};", state_space, initial_state)?;
    writeln!(buffer)?;
    // start model description
    // end model description
    // model tail
    for r in 0..state_space {
        for a in 0..enabled_actions[r] {
            write!(buffer, "\t[] x={} -> ", r)?;
            // write the transition
            let k = (P.indptr().raw_storage()[(adjusted_sidx[r] + a) as usize + 1] - 
                P.indptr().raw_storage()[(adjusted_sidx[r]+ a) as usize]) as usize;
            if k > 0 {
                for j in 0..k {
                    // data 
                    let c = P.indices()[P.indptr().raw_storage()[(adjusted_sidx[r] + a) as usize] as usize + j];
                    let p = P.data()[
                        P.indptr().raw_storage()[(adjusted_sidx[r] + a) as usize] as usize + j
                    ];
                    if j == 0 && j < k - 1 {
                        write!(buffer, "{}:(x'={}) ", p, c)?;
                    } else if j == 0 && j == k - 1 {
                        write!(buffer, "{}:(x'={});", p, c)?;
                    } else if j == k - 1 {
                        write!(buffer, "+ {}:(x'={});", p, c)?;
                    } else {
                        write!(buffer, "+ {}:(x'={}) ", p, c)?;
                    }
                }
                write!(buffer, "\n")?;
            }
        }
    }
    writeln!(buffer)?;
    writeln!(buffer, "endmodule")?;
    buffer.flush()?;
    Ok(())
}

/*
-------------------------------------------------------------------
|                     SPARSE MATRIX DEFINITIONS                   |
|                                                                 |
-------------------------------------------------------------------
*/


#[pyclass]
#[derive(Clone, Debug)]
pub struct CxxMatrixf32 {
    #[pyo3(get)]
    pub nzmax: i32,
    #[pyo3(get)]
    pub m: i32, // number of rows
    #[pyo3(get)]
    pub n: i32, // number of cols
    #[pyo3(get)]
    pub p: Vec<i32>,
    #[pyo3(get)]
    pub i: Vec<i32>,
    #[pyo3(get)]
    pub x: Vec<f32>,
    #[pyo3(get)]
    pub nz: i32, //non zero entries
}

#[repr(C)]
pub struct CSparse {
    nzmax: ::std::os::raw::c_int,
    m: ::std::os::raw::c_int,
    n: ::std::os::raw::c_int,
    p: *const ::std::os::raw::c_int,
    i: *const ::std::os::raw::c_int,
    x: *const f32,
    nz: i32
}

pub type cmat = CSparse;

impl CxxMatrixf32 {
    pub fn make(nzmax: i32, m: i32, n: i32, _nz: i32) -> Self {
        CxxMatrixf32 { nzmax, m, n, p: Vec::new(), 
            i: Vec::new(), x: Vec::new(), nz: 0 }
    }

    pub fn new() -> Self {
        CxxMatrixf32 { 
            nzmax: 0, 
            m: 0, 
            n: 0, 
            p: Vec::new(), 
            i: Vec::new(), 
            x: Vec::new(), 
            nz: 0 
        }
    }

    pub fn triple_entry(&mut self, i: i32, j: i32, val: f32) {
        self.i.push(i);
        self.p.push(j);
        self.x.push(val);
        self.nz += 1;
    }
}

/*
-------------------------------------------------------------------
|                         PYTHON INTERFACE                        |
|                                                                 |
-------------------------------------------------------------------
*/
#[pymodule]
fn hybrid(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<DFA>()?;
    m.add_class::<Mission>()?;
    m.add_class::<MessageSender>()?;
    m.add_class::<SCPM>()?;
    m.add_class::<Warehouse>()?;
    m.add_function(wrap_pyfunction!(test_build, m)?)?;
    m.add_function(wrap_pyfunction!(test_initial_policy, m)?)?;
    m.add_function(wrap_pyfunction!(test_cuda_initial_policy, m)?)?;
    m.add_function(wrap_pyfunction!(test_threaded_initial_policy, m)?)?;
    m.add_function(wrap_pyfunction!(test_cudacpu_opt_pol, m)?)?;
    m.add_function(wrap_pyfunction!(msg_test_gpu_stream, m)?)?;
    m.add_function(wrap_pyfunction!(experiment_gpu_cpu_binary_thread, m)?)?;
    m.add_function(wrap_pyfunction!(warehouse_build_test, m)?)?;
    m.add_function(wrap_pyfunction!(test_warehouse_policy_optimisation,m)?)?;
    m.add_function(wrap_pyfunction!(test_warehouse_model_size, m)?)?;
    m.add_function(wrap_pyfunction!(test_warehouse_gpu_policy_optimisation, m)?)?;
    m.add_function(wrap_pyfunction!(test_warehouse_CPU_only, m)?)?;
    m.add_function(wrap_pyfunction!(test_warehouse_GPU_no_stream, m)?)?;
    m.add_function(wrap_pyfunction!(test_warehouse_gpu_only, m)?)?;
    m.add_function(wrap_pyfunction!(msg_test_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(test_warehouse_hybrid, m)?)?;
    m.add_function(wrap_pyfunction!(test_make_prism_file, m)?)?;
    Ok(())
}

/*
-------------------------------------------------------------------
|                           CUDA INTERFACE                        |
|                                                                 |
-------------------------------------------------------------------
*/
extern "C" {
    fn warm_up_gpu();

    fn initial_policy_value(
        pm: i32,
        pn: i32,
        pnz: i32,
        pi: *const i32,
        pj: *const i32,
        px: *const f32,
        pi_size: i32,
        rm: i32,
        rn: i32,
        rnz: i32,
        ri: *const i32,
        rj: *const i32,
        rx: *const f32,
        ri_size: i32,
        x: *mut f32,
        y: *mut f32,
        w: *const f32,
        rmv: *mut f32,
        unstable: *mut i32,
        eps: f32
    );

    fn policy_value_stream(
        p_init_m: i32,
        p_init_n: i32,
        p_init_nz: i32,
        p_init_i: *const i32,
        p_init_j: *const i32,
        p_init_x: *const f32,
        p_init_i_size: i32,
        p_m: i32,
        p_n: i32,
        p_nz: i32,
        p_i: *const i32,
        p_j: *const i32,
        p_x: *const f32,
        p_i_size: i32,
        r_init_m: i32,
        r_init_n: i32,
        r_init_nz: i32,
        r_init_i: *const i32,
        r_init_j: *const i32,
        r_init_x: *const f32,
        r_init_i_size: i32,
        rm: i32,
        rn: i32,
        rnz: i32,
        ri: *const i32,
        rj: *const i32,
        rx: *const f32,
        ri_size: i32,
        x_init: *mut f32,
        y_init: *mut f32,
        w_init: *const f32,
        rmv_init: *mut f32,
        y: *mut f32,
        rmv: *mut f32,
        w: *const f32,
        unstable: *mut i32,
        Pi: *mut i32,
        enabled_actions: *const i32,
        adj_sidx: *const i32,
        stable: *mut f32,
        eps: f32
    );

    fn policy_optimisation(
        Pi: *const i32,
        pm: i32,
        pn: i32, 
        pnz: i32,
        pi: *const i32,
        pj: *const i32,
        px: *const f32,
        rm: i32,
        rn: i32,
        rnz: i32,
        ri: *const i32,
        rj: *const i32,
        rx: *const f32,
        x: *mut f32,
        y: *mut f32,
        rmv: *mut f32,
        w: *const f32,
        eps: f32,
        block_size: i32,
        enabled_actions: *const i32,
        adj_sidx: *const i32,
        stable: *mut f32
    );

    fn multi_obj_solution(
        pm: i32,
        pn: i32,
        pnz: i32,
        pi: *const i32,
        pj: *const i32,
        px: *const f32,
        pi_size: i32,
        rm: i32,
        rn: i32,
        rnz: i32,
        ri: *const i32,
        rj: *const i32,
        rx: *const f32,
        ri_size: i32,
        eps: f32,
        nobjs: i32,
        x: *mut f32,
        w: *mut f32,
        z: *mut f32,
        unstable: *mut i32
    );
}

pub fn cuda_warm_up_gpu() {
    unsafe {
        warm_up_gpu();
    }
}

pub fn cuda_initial_policy_value(
    P: CsMatBase<f32, i32, &[i32], &[i32], &[f32]>,
    R: CsMatBase<f32, i32, &[i32], &[i32], &[f32]>,
    w: &[f32],
    eps: f32,
    x: &mut [f32],
    y: &mut [f32],
    rmv: &mut [f32],
    unstable: &mut [i32]
) {
    let (pm, pn) = P.shape();
    let pnz = P.nnz() as i32;
    let (rm, rn) = R.shape();
    let rnz = R.nnz() as i32;  
    unsafe {
        initial_policy_value(
            pm as i32, 
            pn as i32, 
            pnz, 
            P.indptr().raw_storage().as_ptr(), 
            P.indices().as_ptr(), 
            P.data().as_ptr(), 
            P.indptr().raw_storage().len() as i32,
            rm as i32, 
            rn as i32, 
            rnz, 
            R.indptr().raw_storage().as_ptr(), 
            R.indices().as_ptr(), 
            R.data().as_ptr(), 
            R.indptr().raw_storage().len() as i32,
            x.as_mut_ptr(), 
            y.as_mut_ptr(), 
            w.as_ptr(), 
            rmv.as_mut_ptr(), 
            unstable.as_mut_ptr(),
            eps
        )
    }
}

pub fn cuda_initial_policy_value_pinned_graph(
    Pinit: CsMatBase<f32, i32, &[i32], &[i32], &[f32]>,
    Rinit: CsMatBase<f32, i32, &[i32], &[i32], &[f32]>,
    P: CsMatBase<f32, i32, &[i32], &[i32], &[f32]>,
    R: CsMatBase<f32, i32, &[i32], &[i32], &[f32]>,
    w_init: &[f32],
    w: &[f32],
    eps: f32,
    x_init: &mut [f32],
    y_init: &mut [f32],
    rmv_init: &mut [f32],
    y: &mut [f32],
    rmv: &mut [f32],
    unstable: &mut [i32],
    Pi: &mut [i32],
    enabled_actions: &[i32],
    adj_sidx: &[i32],
    stable: &mut [f32],
) {
    // Matrix under initial scheduler
    let (p_init_m, p_init_n) = Pinit.shape();
    let p_init_nz = Pinit.nnz() as i32;
    let (r_init_m, r_init_n) = Rinit.shape();
    let r_init_nz = Rinit.nnz() as i32;  
    // Complete
    let (p_m, p_n) = P.shape();
    let p_nz = P.nnz() as i32;
    let (r_m, r_n) = R.shape();
    let r_nz = R.nnz() as i32;
    unsafe {
        policy_value_stream(
            p_init_m as i32, 
            p_init_n as i32, 
            p_init_nz, 
            Pinit.indptr().raw_storage().as_ptr(), 
            Pinit.indices().as_ptr(), 
            Pinit.data().as_ptr(), 
            Pinit.indptr().raw_storage().len() as i32,
            p_m as i32, 
            p_n as i32, 
            p_nz, 
            P.indptr().raw_storage().as_ptr(), 
            P.indices().as_ptr(), 
            P.data().as_ptr(), 
            P.indptr().raw_storage().len() as i32,
            r_init_m as i32, 
            r_init_n as i32, 
            r_init_nz, 
            Rinit.indptr().raw_storage().as_ptr(), 
            Rinit.indices().as_ptr(), 
            Rinit.data().as_ptr(), 
            Rinit.indptr().raw_storage().len() as i32,
            r_m as i32, 
            r_n as i32, 
            r_nz, 
            R.indptr().raw_storage().as_ptr(), 
            R.indices().as_ptr(), 
            R.data().as_ptr(), 
            R.indptr().raw_storage().len() as i32,
            x_init.as_mut_ptr(), 
            y_init.as_mut_ptr(), 
            w_init.as_ptr(), 
            rmv_init.as_mut_ptr(), 
            y.as_mut_ptr(),
            rmv.as_mut_ptr(),
            w.as_ptr(),
            unstable.as_mut_ptr(),
            Pi.as_mut_ptr(),
            enabled_actions.as_ptr(),
            adj_sidx.as_ptr(),
            stable.as_mut_ptr(),
            eps
        )
    }

    
}

pub fn cuda_policy_optimisation(
    P: CsMatBase<f32, i32, &[i32], &[i32], &[f32]>,
    R: CsMatBase<f32, i32, &[i32], &[i32], &[f32]>,
    w: &[f32],
    eps: f32,
    Pi: &[i32],
    x: &mut [f32],
    y: &mut [f32],
    rmv: &mut [f32],
    enabled_actions: &[i32],
    adjusted_sidx: &[i32],
    initial_state: usize,
    stable: &mut [f32]
) -> f32 {
    let (pm, pn) = P.shape();
    let pnz = P.nnz() as i32;
    let (rm, rn) = R.shape();
    let rnz = R.nnz() as i32;
    let block_size = x.len() as i32;
    unsafe {
        policy_optimisation(
            Pi.as_ptr(), 
            pm as i32, 
            pn as i32, 
            pnz, 
            P.indptr().raw_storage().as_ptr(), 
            P.indices().as_ptr(), 
            P.data().as_ptr(), 
            rm as i32, 
            rn as i32, 
            rnz, 
            R.indptr().raw_storage().as_ptr(), 
            R.indices().as_ptr(), 
            R.data().as_ptr(), 
            x.as_mut_ptr(), 
            y.as_mut_ptr(), 
            rmv.as_mut_ptr(), 
            w.as_ptr(), 
            eps, 
            block_size, 
            enabled_actions.as_ptr(),
            adjusted_sidx.as_ptr(),
            stable.as_mut_ptr()
        );
        x[initial_state]
    }
}

pub fn cuda_multi_obj_solution(
    P: CsMatBase<f32, i32, &[i32], &[i32], &[f32]>,
    R: CsMatBase<f32, i32, &[i32], &[i32], &[f32]>,
    eps: f32,
    nobjs: i32
) -> Vec<f32>{
    let (p_m, p_n) = P.shape();
    let p_nz = P.nnz() as i32;
    let (r_m, r_n) = R.shape();
    let r_nz = R.nnz() as i32;
    let mut x: Vec<f32> = vec![0.; p_n];
    let x_ = &mut x;
    let mut z: Vec<f32> = vec![0.; p_m * nobjs as usize];
    let z_ = &mut z;
    let mut w: Vec<f32> = vec![0.; nobjs as usize];
    let w_ = &mut w;
    let mut unstable: Vec<i32> = vec![0; p_m * nobjs as usize];
    let unstable_ = &mut unstable;
    unsafe {
        multi_obj_solution(
            p_m as i32,
            p_n as i32,
            p_nz,
            P.indptr().raw_storage().as_ptr(),
            P.indices().as_ptr(),
            P.data().as_ptr(),
            P.indptr().raw_storage().len() as i32,
            r_m as i32,
            r_n as i32,
            r_nz,
            R.indptr().raw_storage().as_ptr(),
            R.indices().as_ptr(),
            R.data().as_ptr(),
            R.indptr().raw_storage().len() as i32,
            eps, 
            nobjs,
            x_.as_mut_ptr(),
            w_.as_mut_ptr(),
            z_.as_mut_ptr(),
            unstable_.as_mut_ptr()
        )
    }
    z
}
/*
-------------------------------------------------------------------
|                     C_SPARSE MATRIX FUNCTIONS                   |
|                                                                 |
-------------------------------------------------------------------
*/

extern "C" {
    fn create_csr(m: i32, n: i32, nz: i32, i: *const i32, p: *const i32, x: *const f32) -> *const cmat;
    fn sp_spfree(sp: *const cmat);
    fn gaxpy(A: *const cmat, x: *const f32, y: *mut f32);
}

pub fn ffi_spfree(A: *const cmat) {
    unsafe{
        sp_spfree(A)
    }
}

pub fn ffi_create_csr(M: &CxxMatrixf32) -> *const cmat {
    unsafe {
        create_csr(M.m, M.n, M.nz, 
            M.i.as_ptr(), M.p.as_ptr(), M.x.as_ptr())
    }
}

pub fn ffi_gaxpy(A: *const cmat, x: &[f32], y: &mut [f32]) {
    unsafe { 
        gaxpy(A, x.as_ptr(), y.as_mut_ptr());
    }
}
