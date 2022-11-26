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
use sprs::{CsMatBase, CsMatViewI};
use task::dfa::{DFA, Mission};
use envs::message::*;
use hashbrown::HashMap;
use std::hash::Hash;
use envs::warehouse::*;

//extern crate blas_src;
//extern crate cblas_sys;

/*
-------------------------------------------------------------------
|                          HELPER FUNCTIONS                       |
|                                                                 |
-------------------------------------------------------------------
*/

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
        /*println!("min best value: ({}, {})", 
            min_best_val.as_ref().unwrap().0, min_best_val.as_ref().unwrap().2
        );
        println!("task counts: {:?}", task_counts_by_agent);
        */
        allocation.push((
            task as i32,
            min_best_val.as_ref().unwrap().0, 
            min_best_val.as_ref().unwrap().1.to_owned()
        ));
    }
    allocation
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
    m.add_function(wrap_pyfunction!(test_warehouse_GPU_only, m)?)?;
    m.add_function(wrap_pyfunction!(test_gpu_stream, m)?)?;
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
        storage: *mut f32,
        eps: f32,
        nobjs: i32
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
    storage: &mut [f32],
    eps: f32,
    nobjs: i32
) {
    let (p_m, p_n) = P.shape();
    let p_nz = P.nnz() as i32;
    let (r_m, r_n) = R.shape();
    let r_nz = R.nnz() as i32;
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
            storage.as_mut_ptr(),
            eps, 
            nobjs
        )
    }
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
