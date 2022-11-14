#![allow(non_snake_case)]
use crate::CxxMatrixf32;
use pyo3::prelude::*;

#[pyfunction]
pub fn compress(m: CxxMatrixf32) -> CxxMatrixf32 {
    // consume the triple matrix and return a CSR matrix
    let mut mcsr = CxxMatrixf32::make(
        m.nzmax,
        m.m,
        m.n,
        m.nz
    );

    // first get the column counts
    let mut Ci = vec![0; m.m as usize + 1]; 
    let mut Cp = vec![0; m.nz as usize]; 
    let mut Cx = vec![0.; m.nz as usize]; 

    //println!("m.p:{:?}", m.p);
    
    for k in 0..m.nz as usize {
        Cx[k] = m.x[k];
        Cp[k] = m.p[k];
        Ci[m.i[k] as usize + 1] += 1;
    }

    for i in 0..m.m as usize {
        Ci[i + 1] += Ci[i];
    }

    mcsr.i = Ci;
    mcsr.p = Cp;
    mcsr.x = Cx;
    mcsr
}

pub fn mkl_csr_row_start_end(m: &CxxMatrixf32) -> (Vec<i32>, Vec<i32>) {    
    let mut start: Vec<i32> =Vec::new();
    let mut end: Vec<i32> = Vec::new();
    for k in 0..m.i.len() - 1 {
        start.push(m.i[k as usize]);
        end.push(m.i[k as usize + 1]);
    }  
    (start, end)
}