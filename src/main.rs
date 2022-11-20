#![allow(non_snake_case)]
extern crate blas_src;
extern crate cblas_sys;

use cblas_sys::{cblas_dgemm, CBLAS_LAYOUT, CBLAS_TRANSPOSE};
//use crossbeam_channel::unbounded;
use std::{thread, time::Instant};
use sprs::CsMat;
use sprs::prod::mul_acc_mat_vec_csr;
use std::env;

pub fn matrix_mul_test() {
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
}

fn main() {
    let mut t1 = Instant::now();
    let TOTAL_SENDS: usize = 1000;
    let THREADS = 2;

    let (s1, r1) = std::sync::mpsc::channel();

    for _ in 0..TOTAL_SENDS {
        matrix_mul_test();
    }

    println!("Elapsed time using CPU code: {}", t1.elapsed().as_secs_f32());
    t1 = Instant::now();

    for k in 0..THREADS {
        let s1_ = s1.clone();
        thread::spawn(move || {
            for _ in 0..TOTAL_SENDS / THREADS {
                matrix_mul_test();
                s1_.send(format!("hello from thread {k}")).unwrap();
            }
        });
    }

    // Receive all messages currently in the channel.
    for _ in 0..TOTAL_SENDS {
        r1.recv().unwrap();
    }
    //println!("v1: {:?}", v1);

    println!("Elapsed time for threading: {}", t1.elapsed().as_secs_f32());

    println!("RAYON PROPERTIES");
    println!("{:?}", rayon::current_num_threads());
    let v = env::var("RAYON_NUM_THREADS").expect("$RAYON_NUM_THREADS is not set");
    println!("Rayon threads in use: {:?}", v);


    let row: Vec<usize> = vec![0, 3, 5, 8, 11, 13];
    let col: Vec<usize> = vec![0, 1, 3, 0, 1, 2, 3, 4, 0, 2, 3, 1, 4]; 
    let val: Vec<f32> = vec![1., -1., -3., -2., 5., 4., 6., 4., -4.,
                                2., 7., 8., -5.];
    //let nzmax = 13 + 1;
    let m = 5;
    let n = 5;
    let x: Vec<f32> = vec![3.0, 2.0, 5.0, 4.0, 1.0];
    let mut y: Vec<f32> = vec![0., 0., 0., 0., 0.];

    let a = CsMat::new((m, n), row, col, val);

    println!("A: {:?}", a.to_dense());

    let v = a.view();

    mul_acc_mat_vec_csr(a.view(), &x, &mut y);

    println!("y: {:?}", y);

}