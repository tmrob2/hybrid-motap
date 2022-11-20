/* 
TODO start with computing the initial value of the product

The requirements for the initial value include the matrix P,
matrix R an initial vector x, y to capture the values,

A special w vector which only inlcudes weights for the agents

A randomly initialised proper policy

An epsilon threshold for ending value iteration
*/

use sprs::{CsMatBase, prod::mul_acc_mat_vec_csr};
use ndarray::prelude::*;

const MAX_ITERATIONS: usize = 10;

fn abs_max_diff(x: &[f32], y: &[f32]) -> f32 {
    let mut eps = 0.;
    for k in 0..y.len() {
        if (y[k] - x[k]).abs() > eps {
            eps = (y[k] - x[k]).abs();
        }
    }
    eps
}

fn action_comparison(
    y: &mut [f32],
    enabled_actions: &[i32],
    adj_sidx: &[i32],
    xnew: &mut [f32],
    xold: &mut [f32],
    pi: &mut [i32],
    epsilon: f32
) -> (bool, f32) {
    let mut max_eps: f32 = 0.;
    let mut policy_stable: bool = true;
    for r in 0..xnew.len() {
        let mut max_value = -f32::INFINITY;
        let mut argmax_a: i32 = -1;
        for a in 0..enabled_actions[r] {
            if y[(adj_sidx[r] + a) as usize] > max_value {
                max_value = y[(adj_sidx[r] + a) as usize];
                argmax_a = a;
            }
        }
        xnew[r] = max_value;
        if max_value - xold[r] > epsilon {
            policy_stable = false;
            pi[r] = argmax_a;
            max_eps = max_value - xold[r];
        }
    }
    (policy_stable, max_eps)
}

pub fn initial_policy(
    P: CsMatBase<f32, usize, &[usize], &[usize], &[f32]>, // This is a view of the transition matrix CSR fmt
    R: CsMatBase<f32, usize, &[usize], &[usize], &[f32]>, // This is a view of the rewards matrix CSR fmt
    w: &[f32], 
    epsilon: f32, 
    r_v: &mut [f32],
    x: &mut [f32],
    y: &mut [f32]
) {
    let mut eps: f32 = 1.0;
    // First compute the matrix vector dot product R.w
    //println!("|w|: {}, |x|: {}, |y|: {}, |r|: {}", w.len(), x.len(), y.len(), r_v.len());
    //println!("{:?}", R.to_dense());
    mul_acc_mat_vec_csr(R, w, &mut *r_v);
    let mut k: usize = 0;

    while k < MAX_ITERATIONS && eps > epsilon {

        // compute the sparse matrix vector dot product of P.x and add it to r_v to
        // i.e. y = r + P.x
        for (i, y_) in y.iter_mut().enumerate() {
            *y_ = r_v[i];
        }
        //println!("y before  P.v: \n{:?}", y);
        mul_acc_mat_vec_csr(P, &*x, &mut *y);
        //println!("y after P.v:\n{:?}", y);        

        // check the difference between x and y
        eps = abs_max_diff(&x, &y);

        for (i, x_) in x.iter_mut().enumerate() {
            *x_ = y[i];
        }
        k += 1;
    }

}

pub fn optimal_policy(
    P: CsMatBase<f32, usize, &[usize], &[usize], &[f32]>, // This is a view of the transition matrix CSR fmt
    R: CsMatBase<f32, usize, &[usize], &[usize], &[f32]>, // This is a view of the rewards matrix CSR fmt
    w: &[f32], 
    epsilon: f32, 
    r_v: &mut [f32],
    x: &mut [f32], // x is the output from the intial policy evaluation
    y: &mut [f32], // the size of x and y will be different. 
    pi: &mut [i32], // the size of the policy will be equivalent to |x|
    enabled_actions: &[i32],
    adj_sidx: &[i32]
) {
    let mut eps = 1.0;
    let mut policy_stable = false;
    let mut xtmp = vec![0.; x.len()];
    // First thing is to construct the sparse rewards matrix R.w dot product
    mul_acc_mat_vec_csr(R, w, &mut *r_v);
    //let mut k: usize = 0;

    while !policy_stable {

        // compute the sparse matrix vector dot product of P.x and add it to r_v to
        // i.e. y = r + P.x
        for (i, y_) in y.iter_mut().enumerate() {
            *y_ = r_v[i];
        }
        //println!("y before  P.v: \n{:?}", y);
        mul_acc_mat_vec_csr(P, &*x, &mut *y);

        (policy_stable, eps) = action_comparison(y, enabled_actions, adj_sidx, 
                                            &mut xtmp, x, pi, epsilon);
        for (i, x_) in x.iter_mut().enumerate() {
            *x_ = xtmp[i];
        }

        println!("eps: {}", eps);
    }
}