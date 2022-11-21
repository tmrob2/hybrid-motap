/*
-------------------------------------------------------------------
|                        integration tests                        |
|                                                                 |
-------------------------------------------------------------------
*/

// BLAS Tests are currently commented out. Will need to uncomment and
// load cblas/blis src in Cargo.toml file
use crossbeam_channel::unbounded;
use threadpool::ThreadPool;
use std::sync::mpsc::channel;
//use cblas_sys::{cblas_dgemm, CBLAS_LAYOUT, CBLAS_TRANSPOSE};
use crate::{CxxMatrixf32, ffi_create_csr, ffi_gaxpy, ffi_spfree};
use sprs::CsMat;
use sprs::prod::mul_acc_mat_vec_csr;

/*pub fn test_blis_threads() {

    let EVENTS = 100;
    let pool = ThreadPool::new(16);
    let (tx, rx) = channel();
    for _ in 0..EVENTS {
        let tx_ = tx.clone();
        pool.execute(move || {

            let (m, n, k) = (2, 4, 3);
            let a = vec![
                1.0, 4.0,
                2.0, 5.0,
                3.0, 6.0,
            ];
            let b = vec![
                1.0, 5.0,  9.0,
                2.0, 6.0, 10.0,
                3.0, 7.0, 11.0,
                4.0, 8.0, 12.0,
            ];
            let mut c = vec![
                2.0, 7.0,
                6.0, 2.0,
                0.0, 7.0,
                4.0, 2.0,
            ];

            unsafe {
                cblas_dgemm(CBLAS_LAYOUT::CblasColMajor, 
                    CBLAS_TRANSPOSE::CblasNoTrans, 
                    CBLAS_TRANSPOSE::CblasNoTrans,
                    m, n, k, 1.0, a.as_ptr(), m, 
                    b.as_ptr(), k, 1.0, c.as_mut_ptr(), m);
            }
            tx_.send(1).unwrap();
        });
    }
    let mut output = vec![0; EVENTS];
    for k in 0..EVENTS {
        output[k] = rx.recv().unwrap();
    }
}*/

/*pub fn test_cb_threads(num_threads: usize) {

    let EVENTS = 1000;
    let pool = ThreadPool::new(num_threads);
    let (tx, rx) = unbounded();
    for _ in 0..EVENTS {
        let tx_ = tx.clone();
        pool.execute(move || {

            let (m, n, k) = (2, 4, 3);
            let a = vec![
                1.0, 4.0,
                2.0, 5.0,
                3.0, 6.0,
            ];
            let b = vec![
                1.0, 5.0,  9.0,
                2.0, 6.0, 10.0,
                3.0, 7.0, 11.0,
                4.0, 8.0, 12.0,
            ];
            let mut c = vec![
                2.0, 7.0,
                6.0, 2.0,
                0.0, 7.0,
                4.0, 2.0,
            ];

            /*unsafe {
                cblas_dgemm(CBLAS_LAYOUT::CblasColMajor, 
                    CBLAS_TRANSPOSE::CblasNoTrans, 
                    CBLAS_TRANSPOSE::CblasNoTrans,
                    m, n, k, 1.0, a.as_ptr(), m, 
                    b.as_ptr(), k, 1.0, c.as_mut_ptr(), m);
            }*/
            tx_.send(1).unwrap();
        });
    }
    drop(tx);
    let _output: Vec<_> = rx.iter().collect();
}*/

use rayon::prelude::*;
use ndarray::prelude::*;
pub fn test_rayon_threads(num: usize) {
    let _v: Vec<_> = (0..num).into_par_iter().map(|_| {
        let (_m, _n, _k) = (2, 4, 3);

        let a = arr2(&[[1.0, 4.0], 
                       [2.0, 5.0],
                       [3.0, 6.0]]);
        
        let b = arr2(&[[ 1.0, 5.0,  9.0],
                      [ 2.0, 6.0, 10.0],
                      [3.0, 7.0, 11.0],
                      [4.0, 8.0, 12.0]]);

        let c = arr2(&[[2.0, 7.0],
                          [6.0, 2.0],
                          [0.0, 7.0],
                          [4.0, 2.0]]);

        let _z = b.dot(&a) + c;
        1
    }).collect();
}

use std::thread;

/*pub fn test_explicit_threads_size2() {

    let TOTAL_SENDS: usize = 100000;

    let (s1, r1) = unbounded();
    let (s2, r2) = unbounded();

    thread::spawn(move || {
        for _ in 0..TOTAL_SENDS / 2 {
            let (m, n, k) = (2, 4, 3);
            let a = vec![
                1.0, 4.0,
                2.0, 5.0,
                3.0, 6.0,
            ];
            let b = vec![
                1.0, 5.0,  9.0,
                2.0, 6.0, 10.0,
                3.0, 7.0, 11.0,
                4.0, 8.0, 12.0,
            ];
            let mut c = vec![
                2.0, 7.0,
                6.0, 2.0,
                0.0, 7.0,
                4.0, 2.0,
            ];

            unsafe {
                cblas_dgemm(CBLAS_LAYOUT::CblasColMajor, 
                    CBLAS_TRANSPOSE::CblasNoTrans, 
                    CBLAS_TRANSPOSE::CblasNoTrans,
                    m, n, k, 1.0, a.as_ptr(), m, 
                    b.as_ptr(), k, 1.0, c.as_mut_ptr(), m);
            }
            s1.send(1).unwrap();
        }
        drop(s1);
    });
    
    thread::spawn(move || {
        for _ in 0..TOTAL_SENDS / 2 {
            let (m, n, k) = (2, 4, 3);
            let a = vec![
                1.0, 4.0,
                2.0, 5.0,
                3.0, 6.0,
            ];
            let b = vec![
                1.0, 5.0,  9.0,
                2.0, 6.0, 10.0,
                3.0, 7.0, 11.0,
                4.0, 8.0, 12.0,
            ];
            let mut c = vec![
                2.0, 7.0,
                6.0, 2.0,
                0.0, 7.0,
                4.0, 2.0,
            ];

            unsafe {
                cblas_dgemm(CBLAS_LAYOUT::CblasColMajor, 
                    CBLAS_TRANSPOSE::CblasNoTrans, 
                    CBLAS_TRANSPOSE::CblasNoTrans,
                    m, n, k, 1.0, a.as_ptr(), m, 
                    b.as_ptr(), k, 1.0, c.as_mut_ptr(), m);
            }
            s2.send(1).unwrap();
        }
        drop(s2);
    });

    // Receive all messages currently in the channel.
    let _v1: Vec<_> = r1.iter().collect();
    let _v2: Vec<_> = r2.iter().collect();
}*/

pub fn matrix_mul_test() {
    /*let (m, n, k) = (2, 4, 3);
    let a = vec![
        1.0, 4.0,
        2.0, 5.0,
        3.0, 6.0,
    ];
    let b = vec![
        1.0, 5.0,  9.0,
        2.0, 6.0, 10.0,
        3.0, 7.0, 11.0,
        4.0, 8.0, 12.0,
    ];
    let mut c = vec![
        2.0, 7.0,
        6.0, 2.0,
        0.0, 7.0,
        4.0, 2.0,
    ];*/

    let a = arr2(&[[1.0, 4.0], 
        [2.0, 5.0],
        [3.0, 6.0]]);

    let b = arr2(&[[ 1.0, 5.0,  9.0],
        [ 2.0, 6.0, 10.0],
        [3.0, 7.0, 11.0],
        [4.0, 8.0, 12.0]]);

    let mut c = arr2(&[[2.0, 7.0],
            [6.0, 2.0],
            [0.0, 7.0],
            [4.0, 2.0]]);

    c = b.dot(&a) + c;

    /*unsafe {
        cblas_dgemm(CBLAS_LAYOUT::CblasColMajor, 
            CBLAS_TRANSPOSE::CblasNoTrans, 
            CBLAS_TRANSPOSE::CblasNoTrans,
            m, n, k, 1.0, a.as_ptr(), m, 
            b.as_ptr(), k, 1.0, c.as_mut_ptr(), m);
    }*/
}

pub fn test_explicit_threads_size4() {

    let TOTAL_SENDS: usize = 100000;

    let (s1, r1) = unbounded();
    let (s2, r2) = unbounded();
    let (s3, r3) = unbounded();
    let (s4, r4) = unbounded();

    thread::spawn(move || {
        for _ in 0..TOTAL_SENDS / 4 {
            matrix_mul_test();
            s1.send(1).unwrap();
        }
        drop(s1);
    });
    
    thread::spawn(move || {
        for _ in 0..TOTAL_SENDS / 4 {
            matrix_mul_test();
            s2.send(1).unwrap();
        }
        drop(s2);
    });

    thread::spawn(move || {
        for _ in 0..TOTAL_SENDS / 4 {
            matrix_mul_test();
            s3.send(1).unwrap();
        }
        drop(s3);
    });

    thread::spawn(move || {
        for _ in 0..TOTAL_SENDS / 4 {
            matrix_mul_test();
            s4.send(1).unwrap();
        }
        drop(s4);
    });

    // Receive all messages currently in the channel.
    let _v1: Vec<_> = r1.iter().collect();
    let _v2: Vec<_> = r2.iter().collect();
    let _v3: Vec<_> = r3.iter().collect();
    let _v4: Vec<_> = r4.iter().collect();
} 

pub fn test_explicit_threads_size8() {

    let TOTAL_SENDS: usize = 1_000_000;
    let THREADS = 10;

    let (s1, r1) = std::sync::mpsc::channel();

    for _k in 0..THREADS {
        let s1_ = s1.clone();
        thread::spawn(move || {
            for _ in 0..TOTAL_SENDS / THREADS {
                matrix_mul_test();
                s1_.send(1).unwrap();
            }
        });
    }

    // Receive all messages currently in the channel.
    for _ in 0..TOTAL_SENDS {
        r1.recv().unwrap();
    }

} 

pub fn csparse_mv() {

    // SQUARE MATRIX

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

    // NON SQUARE MATRIX 

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
}

pub fn sprs_mv() {

    // SQUARE MATRIX

    let row: Vec<usize> = vec![0, 3, 5, 8, 11, 13];
    let col: Vec<usize> = vec![0, 1, 3, 0, 1, 2, 3, 4, 0, 2, 3, 1, 4]; 
    let val: Vec<f32> = vec![1., -1., -3., -2., 5., 4., 6., 4., -4.,
                                2., 7., 8., -5.];
    let m = 5;
    let n = 5;
    let x: Vec<f32> = vec![3.0, 2.0, 5.0, 4.0, 1.0];
    let mut y: Vec<f32> = vec![0., 0., 0., 0., 0.];

    let a = CsMat::new((m, n), row, col, val);

    mul_acc_mat_vec_csr(a.view(), &x, &mut y);

    // NON SQUARE MATRIX 

    let row: Vec<usize> = vec![0, 2, 4, 7, 8];
    let col: Vec<usize> = vec![0, 1, 1, 3, 2, 3, 4, 5];
    let val: Vec<f32> = vec![10., 20., 30., 40., 50., 60., 70., 80.];
    let m = 4;
    let n = 6;
    let x: Vec<f32> = vec![1., 2., 3., 4., 5., 6.];
    let mut y: Vec<f32> = vec![0., 0., 0., 0.];

    let b = CsMat::new((m, n), row, col, val);
    mul_acc_mat_vec_csr(b.view(), &x, &mut y);
}

pub fn test_rayon_spmv_c(num: usize) {
    let _v: Vec<_> = (0..num).into_par_iter().map(|_| {
        csparse_mv();
        1
    }).collect();
}

pub fn test_rayon_spmv_sprs(num: usize) {
    let _v: Vec<_> = (0..num).into_par_iter().map(|_| {
        sprs_mv();
        1
    }).collect();
}

