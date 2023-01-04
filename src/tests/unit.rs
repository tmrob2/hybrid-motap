#![allow(unused_imports)]
/*
-------------------------------------------------------------------
|                             TESTS                               |
|                                                                 |
-------------------------------------------------------------------
*/

use std::sync::mpsc::channel;
use threadpool::ThreadPool;
use rayon;
use crate::{CxxMatrixf32, ffi_create_csr, ffi_gaxpy, ffi_spfree};

#[test]
fn cpcsr_create() -> Result<(), Box<dyn std::error::Error>> {
    let row: Vec<i32> = vec![0, 3, 5, 8, 11, 13];
    let col: Vec<i32> = vec![0, 1, 3, 0, 1, 2, 3, 4, 0, 2, 3, 1, 4]; 
    let val: Vec<f32> = vec![1., -1., -3., -2., 5., 4., 6., 4., -4.,
                                2., 7., 8., -5.];
    let nzmax = 13 + 1;
    let m = 5;
    let n = 5;
    let _x: Vec<f32> = vec![3.0, 2.0, 5.0, 4.0, 1.0];
    let mut _y: Vec<f32> = vec![0., 0., 0., 0., 0.];

    let M = &CxxMatrixf32 { 
        nzmax, 
        m, 
        n, 
        p: col, 
        i: row, 
        x: val, 
        nz: 13
    };

    let C = ffi_create_csr(M);
    ffi_spfree(C);
    Ok(())
}

#[test]
fn spmv() {

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

    let C = ffi_create_csr(M);
    ffi_gaxpy(C, &x, &mut y);
    ffi_spfree(C);

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

    let C = ffi_create_csr(&M2);
    ffi_gaxpy(C, &x, &mut y);
    ffi_spfree(C);

    assert_eq!(y, vec![50., 220., 740., 480.]);

}

#[test]
fn test_splib_threads() {
    let pool = ThreadPool::new(2);
    let (tx, rx) = channel();
    for _t1 in 0..10 {
        let tx_ = tx.clone();
        pool.execute(move || {
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

            let C = ffi_create_csr(M);
            ffi_gaxpy(C, &x, &mut y);
            ffi_spfree(C);

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

            let C = ffi_create_csr(&M2);
            ffi_gaxpy(C, &x, &mut y);
            ffi_spfree(C);

            assert_eq!(y, vec![50., 220., 740., 480.]);
            tx_.send(1).unwrap();
        });
    }
    let mut output = vec![0; 10];
    for k in 0..10 {
        output[k] = rx.recv().unwrap();
    }
    assert_eq!(output, vec![1; 10]);
}


#[test]
fn test_cb_channel() {
    let pool = ThreadPool::new(2);
    let (tx, rx) = channel();
    for _ in 0..10 {
        let tx_ = tx.clone();
        pool.execute(move || {
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

            let C = ffi_create_csr(M);
            ffi_gaxpy(C, &x, &mut y);
            ffi_spfree(C);

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

            let C = ffi_create_csr(&M2);
            ffi_gaxpy(C, &x, &mut y);
            ffi_spfree(C);

            assert_eq!(y, vec![50., 220., 740., 480.]);
            tx_.send(1).unwrap();
        });
    }
    drop(tx);
    let output: Vec<_> = rx.iter().collect();
    assert_eq!(output, vec![1; 10]);
}

#[test]
fn test_rayon_threads() {
    // construct a vector of MDP, task pairs as a cartesian product from the model
    // then run a parallel iter over this 
    use rayon::prelude::*;
    let v: Vec<_> = (0..10).into_par_iter().map(|_| {
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

        let C = ffi_create_csr(M);
        ffi_gaxpy(C, &x, &mut y);
        ffi_spfree(C);

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

        let C = ffi_create_csr(&M2);
        ffi_gaxpy(C, &x, &mut y);
        ffi_spfree(C);

        assert_eq!(y, vec![50., 220., 740., 480.]);
        1
    }).collect();

    assert_eq!(v, vec![1; 10]);
    
}