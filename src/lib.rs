#![allow(non_snake_case)]
pub mod threading;
pub mod model;
pub mod agent;
pub mod task;
pub mod envs;
pub mod algorithms;
pub mod sparse;

use envs::message::MessageSender;
use model::scpm::SCPM;
use pyo3::prelude::*;
use task::dfa::{DFA, Mission};
use envs::message::*;

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
    m.add_function(wrap_pyfunction!(thread_test, m)?)?;
    m.add_function(wrap_pyfunction!(mkl_test, m)?)?;
    m.add_function(wrap_pyfunction!(test_initial_policy, m)?)?;
    //m.add_function(wrap_pyfunction!(csr_impl_test, m)?)?;
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
|                         MKL FUNCTIONS                           |
|                                                                 |
-------------------------------------------------------------------
*/

extern "C" {
    fn test_blas_routine() -> f32;
}

#[pyfunction]
pub fn mkl_test() -> f32 {
    let r: f32; 
    unsafe {
        r = test_blas_routine();
    }
    r
}

extern "C" {
    fn test_mv(
        i: *const i32, 
        p: *const i32, 
        x: *const f32, 
        m: i32,
        n: i32,
        x: *const f32,
        y: *mut f32
    );
}

pub fn mkl_test_mv(M: &CxxMatrixf32, x: &[f32], y: &mut [f32]) {
    unsafe {
        test_mv(
            M.i.as_ptr(), 
            M.p.as_ptr(), 
            M.x.as_ptr(), 
            M.m,
            M.n,
            x.as_ptr(),
            y.as_mut_ptr()
        )
    }
}

extern "C" {
    fn initial_policy(
        p_row_ptr: *const i32,
        p_col_ptr: *const i32,
        p_vals: *const f32,
        pm: i32,
        pn: i32,
        r_row_ptr: *const i32,
        r_col_ptr: *const i32,
        r_vals: *const f32,
        rm: i32,
        rn: i32,
        r_v: *mut f32,
        x: *mut f32,
        y: *mut f32,
        epsilon: f32
    );
}

pub fn cpu_intial_policy(
    P: &CxxMatrixf32,
    R: &CxxMatrixf32,
    r_v: &mut [f32],
    x: &mut [f32],
    y: &mut [f32],
    epsilon: f32
) {
    unsafe { 
        initial_policy(
            P.i.as_ptr(), 
            P.p.as_ptr(), 
            P.x.as_ptr(), 
            P.m, 
            P.n, 
            R.i.as_ptr(), 
            R.p.as_ptr(), 
            P.x.as_ptr(), 
            R.m, 
            R.n, 
            r_v.as_mut_ptr(), 
            x.as_mut_ptr(), 
            y.as_mut_ptr(), 
            epsilon
        )
    }
}

/*
-------------------------------------------------------------------
|                             TESTS                               |
|                                                                 |
-------------------------------------------------------------------
*/

#[cfg(test)]
mod tests {
    use crate::mkl_test;
    use crate::CxxMatrixf32;
    use crate::mkl_test_mv;
    #[test]
    fn mkl() {
        mkl_test();
        //assert_eq!(result, 10.);
    }

    #[test]
    fn mkl_spblas() {

        println!("TESTING SQUARE MATRIX");
        let row: Vec<i32> = vec![0, 3, 5, 8, 11, 13];
        let col: Vec<i32> = vec![0, 1, 3, 0, 1, 2, 3, 4, 0, 2, 3, 1, 4]; 
        let val: Vec<f32> = vec![1., -1., -3., -2., 5., 4., 6., 4., -4.,
                                 2., 7., 8., -5.];
        let nzmax = 13 + 1;
        let m = 5;
        let n = 5;
        let x: Vec<f32> = vec![3.0, 2.0, 5.0, 4.0, 1.0];
        let mut y: Vec<f32> = vec![0., 0., 0., 0., 0.];

        let M = &CxxMatrixf32 { 
            nzmax, 
            m, 
            n, 
            p: col, 
            i: row, 
            x: val, 
            nz: 13
        };

        mkl_test_mv(M, &x, &mut y);

        assert_eq!(y, vec![-11., 4., 48., 26., 11.]);

        println!("TESTING NON-SQUARE MATRIX");

        let row: Vec<i32> = vec![0, 2, 4, 7, 8];
        let col: Vec<i32> = vec![0, 1, 1, 3, 2, 3, 4, 5];
        let val: Vec<f32> = vec![10., 20., 30., 40., 50., 60., 70., 80.];
        let nzmax = 9;
        let m = 4;
        let n = 6;
        let x: Vec<f32> = vec![1., 2., 3., 4., 5., 6.];
        let mut y: Vec<f32> = vec![0., 0., 0., 0.];

        let M2 = CxxMatrixf32 {
            nzmax,
            m,
            n,
            p: col,
            i: row,
            x: val,
            nz: 8
        };

        mkl_test_mv(&M2, &x, &mut y);

        assert_eq!(y, vec![50., 220., 740., 480.]);

    }
}



