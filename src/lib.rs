#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
pub mod model;
pub mod agent;
pub mod task;
pub mod envs;
pub mod algorithms;
pub mod sparse;
pub mod tests;
pub mod solvers;



use agent::env::Env;
//use envs::{message::MessageSender, warehouse::Warehouse};
use model::scpm::SCPM;
use pyo3::prelude::*;
use rand::Rng;
use sprs::CsMatBase;
use task::dfa::{DFA, Mission};
use envs::{message::*, warehouse::*, example::*, msg_variants::*, dst::*};
use hashbrown::HashMap;
use std::hash::Hash;
use std::time::Instant;

use std::io::prelude::*;
use std::fs::File;

use crate::{model::mostapu::{StapuState, MOSTAPU_bfs}, solvers::motap::stapu_solver};

pub fn generic_motap_mostapu_cpu<E>(
    model: &SCPM,
    env: &E,
    w: Vec<f32>,
    eps: f32,
    debug: i32,
    max_iter: usize,
    max_unstable: i32,
    initial_agent_states: &[i32]
) where E: Env<i32> {
println!("--------------------------");
println!("   MOTAP MOSTAPU CPU TEST ");
println!("--------------------------");

    let t1 = Instant::now();
    let dbug = debug_level(debug);
    let initial_state = StapuState {
        s: env.get_init_state(0),
        Q: (0..model.tasks.size).map(
            |t| model.tasks.get_task(t).initial_state
            ).collect(),
        agentid: 0,
        active_tasks: None,
        remaining: (0..model.tasks.size as i32).collect()
    };
    let alloc_acts: Vec<i32> = (0..model.tasks.size as i32 + 1).collect();
    let agent_acts: Vec<i32> = model.actions.iter()
        .map(|a| a + model.tasks.size as i32 + 1)
        .collect();
    let actions = [&alloc_acts[..], &agent_acts[..]].concat();
    let stapu = MOSTAPU_bfs(
        initial_state, 
        env, 
        model.num_agents, 
        model.tasks.size, 
        model.tasks.size + model.num_agents, 
        model, 
        &actions,
        initial_agent_states
    );
    /*let mut acc: Vec<usize> = Vec::new();
    for state in stapu.states.iter() {
        match state.active_tasks {
            Some(t) => { 
                let dfa = model.tasks.get_task( t as usize);
                if dfa.accepting.contains(&state.Q[t as usize]) {
                    acc.push(*stapu.state_map.get(state).unwrap());
                }
            }
            None => { }
        }
    }*/
    //
    match dbug {
        Debug::None => { }
        _ => { 
            println!("Time to create mamdp model: {}\n|S|: {},\n|P|: {}", 
                t1.elapsed().as_secs_f32(),
                stapu.states.len(),
                stapu.P.nnz()
            );
        }
    }

    let result = stapu_solver(&stapu, model.num_agents, 
        model.tasks.size, &w, eps, dbug, max_iter, max_unstable);
    match dbug {
        Debug::None => { }
        _ => { 
            println!("Total runtime {}", t1.elapsed().as_secs_f32());
            println!("result: {:?}", result);
        }
    }
}


/*
-------------------------------------------------------------------
|                          HELPER FUNCTIONS                       |
|                                                                 |
-------------------------------------------------------------------
*/
#[derive(Copy, Clone, PartialEq)]
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

fn prism_trans_label(val: i32) -> String {
    match val {
        0 => { "a".to_string() }
        1 => {"b".to_string() }
        _ => { "b".to_string() }
    }
}
/// Write a PRISM file which represents a Agent x Task pair. 
pub fn prism_file_generator(
    P: CsMatBase<f32, i32, &[i32], &[i32], &[f32]>,
    state_space: usize,
    adjusted_sidx: &[i32],
    enabled_actions: &[i32],
    initial_state: usize,
    accepting_states: &[usize],
    mapping: &[i32]
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
            write!(buffer, "\t[{}] x={} -> ", prism_trans_label(mapping[r]), r)?;
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
    writeln!(buffer)?;
    write!(buffer, "label \"accepting\" = x=")?;
    let mut count = 0;
    for state in accepting_states.iter() {
        if count == 0 {
            write!(buffer, "{}", state)?;
        } /*else if count == accepting_states.len() - 1 {
            write!(buffer, " | x={};", state)?;
        } */
        else { 
            write!(buffer, " | x={}", state)?;
        }
        count += 1;
    }
    write!(buffer, ";")?;
    writeln!(buffer)?;
    writeln!(buffer)?;
    writeln!(buffer, "rewards")?;
    writeln!(buffer, "\t[a] true: 0;")?;
    writeln!(buffer, "\t[b] true: 1;")?;
    writeln!(buffer, "endrewards")?;
    buffer.flush()?;
    Ok(())
}

/// Write a PRISM file which represents a Agent x Task pair. 
pub fn motap_prism_file_generator(
    P: CsMatBase<f32, i32, &[i32], &[i32], &[f32]>,
    state_space: usize,
    adjusted_sidx: &[i32],
    enabled_actions: &[i32],
    initial_state: usize,
    accepting_states: &[usize],
    obj_rew: &HashMap<i32, i32>,
    num_agents: i32,
    num_tasks: i32
) -> std::io::Result<()> 
{
    let mut buffer = File::create(format!("prism_A{}_T{}.mn", num_agents, num_tasks))?;
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
            match obj_rew.get(&(r as i32)) {
                Some(k) => {
                    write!(buffer, "\t[{}] x={} -> ", format!("k{}", k), r)?;
                }
                None => {
                    write!(buffer, "\t[] x={} -> ", r)?;
                }
            }
            
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
    writeln!(buffer)?;
    write!(buffer, "label \"accepting\" = x=")?;
    let mut count = 0;
    for state in accepting_states.iter() {
        if count == 0 {
            write!(buffer, "{}", state)?;
        } /*else if count == accepting_states.len() - 1 {
            write!(buffer, " | x={};", state)?;
        } */
        else { 
            write!(buffer, " | x={}", state)?;
        }
        count += 1;
    }
    write!(buffer, ";")?;
    writeln!(buffer)?;
    for k in 0..num_agents {
        writeln!(buffer)?;
        writeln!(buffer, "rewards \"obj{}\"", k)?;
        writeln!(buffer, "\t[k{}] true: 1;", k)?;
        writeln!(buffer, "endrewards")?;
    }
    buffer.flush()?;
    Ok(())
}

pub fn prism_explicit_tra_file_generator(
    P: CsMatBase<f32, i32, &[i32], &[i32], &[f32]>,
    state_space: usize,
    adjusted_sidx: &[i32],
    enabled_actions: &[i32]
) -> std::io::Result<()> 
{
    let mut buffer = File::create("prism.model.tra")?;
    // document heading
    writeln!(buffer, "{} {} {}", P.shape().1, P.shape().0, P.nnz())?;
    // start model description
    // end model description
    // model tail
    for r in 0..state_space {
        for a in 0..enabled_actions[r] {
            //write!(buffer, "\t[] x={} -> ", r)?;
            // write the transition
            let k = (P.indptr().raw_storage()[(adjusted_sidx[r] + a) as usize + 1] - 
                P.indptr().raw_storage()[(adjusted_sidx[r]+ a) as usize]) as usize;
            if k > 0 {
                for j in 0..k {
                    let c = P.indices()[P.indptr().raw_storage()[(adjusted_sidx[r] + a) as usize] as usize + j];
                    let p = P.data()[
                        P.indptr().raw_storage()[(adjusted_sidx[r] + a) as usize] as usize + j
                    ];
                    writeln!(buffer, "{} {} {} {}", r, a, c, p)?;
                }
                //write!(buffer, "\n")?;
            }
        }
    }
    buffer.flush()?;
    Ok(())
}

pub fn prism_explicit_staterew_file_generator(
    r: &[f32],
    state_space: usize,
    adjusted_sidx: &[i32]
) -> std::io::Result<()> 
{
    let mut buffer = File::create("prism.model.rew")?;
    // document heading
    let nnz = r.iter().filter(|x| **x > 0.).count();
    writeln!(buffer, "{} {}", state_space, nnz)?;
    // start model description
    // end model description
    // model tail
    for ii in 0..state_space {
        if r[adjusted_sidx[ii] as usize] != 0. {
            writeln!(buffer, "{} {}", ii , -r[adjusted_sidx[ii] as usize])?;
            //println!("{} {}", ii, -r[adjusted_sidx[ii] as usize]);
        }
    }
    buffer.flush()?;
    Ok(())
}

pub fn prism_explicit_transrew_file_generator(
    state_space: usize,
    P: CsMatBase<f32, i32, &[i32], &[i32], &[f32]>,
    enabled_actions: &[i32],
    adjusted_sidx: &[i32],
    rewards: &[f32],
) -> std::io::Result<()> {
    let mut buffer = File::create("prism.model.trans.rew")?;
    // document heading
    writeln!(buffer, "{} {} {}", P.shape().1, P.shape().0, P.nnz())?;
    // start model description
    // end model description
    // model tail
    for r in 0..state_space {
        for a in 0..enabled_actions[r] {
            //write!(buffer, "\t[] x={} -> ", r)?;
            // write the transition
            let k = (P.indptr().raw_storage()[(adjusted_sidx[r] + a) as usize + 1] - 
                P.indptr().raw_storage()[(adjusted_sidx[r]+ a) as usize]) as usize;
            if k > 0 {
                for j in 0..k {
                    let c = P.indices()[P.indptr().raw_storage()[(adjusted_sidx[r] + a) as usize] as usize + j];
                    let rew = rewards[(adjusted_sidx[r] + a) as usize];
                    writeln!(buffer, "{} {} {} {}", r, a, c, rew)?;
                }
                //write!(buffer, "\n")?;
            }
        }
    }
    buffer.flush()?;
    Ok(())
}

pub fn prism_explicit_label_file_generator(
    init_state: usize,
    acc: &[usize]
) -> std::io::Result<()> 
{
    let mut buffer = File::create("prism.model.lab")?;
    // document heading
    writeln!(buffer, "0=\"init\" 1=\"done\"")?;
    // start model description
    // end model description
    // model tail
    writeln!(buffer, "{}: 0", init_state)?;
    for k in acc.iter() {
        writeln!(buffer, "{}: 1", *k)?;
    }
    buffer.flush()?;
    Ok(())
}

pub fn choose_random_policy(state_size: usize, enabled_actions: &[i32]) -> Vec<i32> {
    let mut pi: Vec<i32> = vec![-1; state_size];
    let mut rng = rand::thread_rng();
    for s in 0..state_size {
        let rand_act = rng
            .gen_range(0..enabled_actions[s]);
        pi[s] = rand_act;
    }
    pi
}

pub fn storm_explicit_tra_file_generator(
    P: CsMatBase<f32, i32, &[i32], &[i32], &[f32]>,
    state_space: usize,
    adjusted_sidx: &[i32],
    enabled_actions: &[i32],
    name: &str
) -> std::io::Result<()> 
{
    let mut buffer = File::create(name)?;
    // document heading
    //writeln!(buffer, "{} {} {}", P.shape().1, P.shape().0, P.nnz())?;
    writeln!(buffer, "mdp")?;
    // start model description
    // end model description
    // model tail
    for r in 0..state_space {
        for a in 0..enabled_actions[r] {
            //write!(buffer, "\t[] x={} -> ", r)?;
            // write the transition
            let k = (P.indptr().raw_storage()[(adjusted_sidx[r] + a) as usize + 1] - 
                P.indptr().raw_storage()[(adjusted_sidx[r]+ a) as usize]) as usize;
            if k > 0 {
                for j in 0..k {
                    let c = P.indices()[P.indptr().raw_storage()[(adjusted_sidx[r] + a) as usize] as usize + j];
                    let p = P.data()[
                        P.indptr().raw_storage()[(adjusted_sidx[r] + a) as usize] as usize + j
                    ];
                    writeln!(buffer, "{} {} {} {}", r, a, c, p)?;
                }
                //write!(buffer, "\n")?;
            }
        }
    }
    buffer.flush()?;
    Ok(())
}

pub fn storm_explicit_staterew_file_generator(
    r: &[f32],
    state_space: usize,
    adjusted_sidx: &[i32],
    name: &str
) -> std::io::Result<()> 
{
    let mut buffer = File::create(name)?;
    // document heading
    //let nnz = r.iter().filter(|x| **x > 0.).count();
    //writeln!(buffer, "{} {}", state_space, nnz)?;
    // start model description
    // end model description
    // model tail
    for ii in 0..state_space {
        if r[adjusted_sidx[ii] as usize] != 0. {
            writeln!(buffer, "{} {}", ii , -r[adjusted_sidx[ii] as usize])?;
            //println!("{} {}", ii, -r[adjusted_sidx[ii] as usize]);
        }
    }
    buffer.flush()?;
    Ok(())
}

pub fn storm_explicit_label_file_generator(
    init_state: usize,
    acc: &[usize]
) -> std::io::Result<()> 
{
    let mut buffer = File::create("storm.w.lab")?;
    // document heading
    writeln!(buffer, "#DECLARATION")?;
    writeln!(buffer, "init done0")?;
    writeln!(buffer, "#END")?;
    // start model description
    // end model description
    // model tail
    writeln!(buffer, "{} init", init_state)?;
    for k in acc.iter() {
        writeln!(buffer, "{} done0", *k)?;
    }
    buffer.flush()?;
    Ok(())
}

pub fn storm_ctmdp_explicit_label_file_generator(
    init_state: usize,
    acc: &HashMap<i32, Vec<usize>>,
    num_tasks: usize,
    name: &str
) -> std::io::Result<()> 
{
    let mut buffer = File::create(name)?;
    // document heading
    writeln!(buffer, "#DECLARATION")?;
    write!(buffer, "init ")?;
    for k in 0..num_tasks {
        write!(buffer, "done{} ", k)?;
    }
    writeln!(buffer)?;
    writeln!(buffer, "#END")?;
    // start model description
    // end model description
    // model tail
    writeln!(buffer, "{} init", init_state)?;
    for task in 0..num_tasks {
        for done in acc.get(&(task as i32)).unwrap().iter() {
            writeln!(buffer, "{} done{}", *done, task)?;
        }
    }
    buffer.flush()?;
    Ok(())
}

fn new_target(
    hullset: Vec<Vec<f32>>, 
    weights: Vec<Vec<f32>>, 
    target: Vec<f32>,
    num_agents: usize,
    constraint_threshold: Vec<f32>
    //l: usize,
    //m: usize,
    //n: usize,
    //iteration: usize,
    //cstep: f64,
    //pstep: f64
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let new_target_script = include_str!("algorithms/eucl.py");
    let result: Vec<f32> = Python::with_gil(|py| -> PyResult<Vec<f32>> {
        let lpnewtarget = PyModule::from_code(py, new_target_script, "", "")?;
        let lpnewtarget_result = lpnewtarget.getattr("eucl_new_target")?.call1((
            hullset,
            weights,
            target,
            //l,
            //m,
            num_agents,
            constraint_threshold
            //iteration,
            //cstep,
            //pstep
        ))?.extract()?;
        Ok(lpnewtarget_result)
    }).unwrap();
    Ok(result)
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
    m.add_class::<Example>()?;
    m.add_class::<SCPM>()?;
    m.add_class::<Warehouse>()?;
    // Exp 3 Variants
    m.add_class::<MAS>()?;
    m.add_class::<Model>()?;
    // STAPU Exp
    m.add_class::<DSTMAS>()?;
    m.add_class::<DSTModel>()?;
    //
    m.add_function(wrap_pyfunction!(test_build, m)?)?;
    // MAMDP model building
    m.add_function(wrap_pyfunction!(test_build_mamdp, m)?)?;
    // SCPM Model building test
    m.add_function(wrap_pyfunction!(test_build_scpm, m)?)?;
    // STAPU Model building test
    m.add_function(wrap_pyfunction!(test_build_stapu, m)?)?;
    m.add_function(wrap_pyfunction!(dst_build_test, m)?)?;
    //
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
    m.add_function(wrap_pyfunction!(test_warehouse_GPU_no_stream, m)?)?;
    m.add_function(wrap_pyfunction!(test_warehouse_gpu_only, m)?)?;
    m.add_function(wrap_pyfunction!(test_warehouse_single_CPU, m)?)?;
    m.add_function(wrap_pyfunction!(msg_test_cpu, m)?)?;
    // motap cpu test
    m.add_function(wrap_pyfunction!(motap_msg_test_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(motap_test_cpu, m)?)?;
    // mamdp cpu test
    m.add_function(wrap_pyfunction!(motap_mamdp_msg_test_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(motap_mamdp_test_cpu, m)?)?;
    // SCPM cpu test
    m.add_function(wrap_pyfunction!(motap_scpm_test_cpu, m)?)?;
    // 
    // STAPU
    m.add_function(wrap_pyfunction!(motap_stapu_test_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(dst_stapu_model, m)?)?;
    //
    m.add_function(wrap_pyfunction!(test_warehouse_hybrid, m)?)?;
    m.add_function(wrap_pyfunction!(test_make_prism_file, m)?)?;
    m.add_function(wrap_pyfunction!(warehouse_make_prism_file, m)?)?;
    m.add_function(wrap_pyfunction!(test_ctmdp_build, m)?)?; 
    m.add_function(wrap_pyfunction!(test_warehouse_ctmdp, m)?)?;
    m.add_function(wrap_pyfunction!(synthesis_test, m)?)?;
    m.add_function(wrap_pyfunction!(test_warehouse_dec, m)?)?;
    m.add_function(wrap_pyfunction!(example_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(ex_synthesis, m)?)?;
    // motap scheduler synthesis
    m.add_function(wrap_pyfunction!(motap_mamdp_synthesis_test, m)?)?;
    m.add_function(wrap_pyfunction!(motap_mamdp_synth, m)?)?;
    // virtual goal experiments
    m.add_function(wrap_pyfunction!(motap_mamdp_synth, m)?)?;
    m.add_function(wrap_pyfunction!(motap_synth, m)?)?;
    m.add_function(wrap_pyfunction!(motap_mamdp_synth, m)?)?;
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
        eps: f32,
        max_iter: i32, 
        max_unstable: i32
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
        eps: f32,
        max_iter: i32, 
        max_unstable: i32
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
        stable: *mut f32,
        max_iter: i32
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
        unstable: *mut i32,
        max_iter: i32, 
        max_unstable: i32
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
    unstable: &mut [i32],
    max_iter: i32, 
    max_unstable: i32
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
            eps,
            max_iter, 
            max_unstable
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
    max_iter: i32,
    max_unstable: i32
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
            eps,
            max_iter, 
            max_unstable
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
    stable: &mut [f32],
    max_iter: i32
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
            stable.as_mut_ptr(),
            max_iter
        );
        x[initial_state]
    }
}

pub fn cuda_multi_obj_solution(
    P: CsMatBase<f32, i32, &[i32], &[i32], &[f32]>,
    R: CsMatBase<f32, i32, &[i32], &[i32], &[f32]>,
    eps: f32,
    nobjs: i32,
    max_iter: i32,
    max_unstable: i32
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
            unstable_.as_mut_ptr(),
            max_iter, 
            max_unstable
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
