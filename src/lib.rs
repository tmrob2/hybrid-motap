#![allow(non_snake_case)]
pub mod model;
pub mod agent;
pub mod task;
pub mod envs;
pub mod algorithms;
pub mod sparse;
pub mod tests;

use envs::message::MessageSender;
use model::scpm::SCPM;
use pyo3::prelude::*;
use task::dfa::{DFA, Mission};
use envs::message::*;
use hashbrown::HashMap;
use std::{hash::Hash};

extern crate blas_src;
extern crate cblas_sys;





/*
-------------------------------------------------------------------
|                          HELPER FUNCTIONS                       |
|                                                                 |
-------------------------------------------------------------------
*/

pub fn reverse_key_value_pairs<T, U>(map: &HashMap<T, U>) -> HashMap<U, T> 
where T: Clone + Hash, U: Clone + Hash + Eq {
    map.into_iter().fold(HashMap::new(), |mut acc, (a, b)| {
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

#[pymodule]
fn hybrid(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<DFA>()?;
    m.add_class::<Mission>()?;
    m.add_class::<MessageSender>()?;
    m.add_class::<SCPM>()?;
    m.add_function(wrap_pyfunction!(test_build, m)?)?;
    m.add_function(wrap_pyfunction!(test_initial_policy, m)?)?;
    m.add_function(wrap_pyfunction!(test_threaded_initial_policy, m)?)?;
    Ok(())
}

/*
-------------------------------------------------------------------
|                         CBLAS DEFINITIONS                       |
|                                                                 |
-------------------------------------------------------------------
*/
/*pub fn cblas_scopy_ffi(n: i32, x: &[f32], y: &mut [f32]) {
    unsafe {
        cblas_scopy(n, x.as_ptr(), 1, y.as_mut_ptr(), 1);
    }
}

pub fn cblas_sscal_ffi(n: i32, alpha: f32, x: &mut [f32]) {
    unsafe {
        cblas_sscal(n, alpha, x.as_mut_ptr(), 1);
    }
}

pub fn cblas_ddot_ffi(n: i32, x: &[f32], y: &mut [f32]) {
    unsafe {
        cblas_sdot(n, x.as_ptr(), 1, y.as_mut_ptr(), 1);
    }
}*/
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






