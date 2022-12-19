/* 
TODO start with computing the initial value of the product

The requirements for the initial value include the matrix P,
matrix R an initial vector x, y to capture the values,

A special w vector which only inlcudes weights for the agents

A randomly initialised proper policy

An epsilon threshold for ending value iteration
*/

use sprs::{CsMatBase, prod::mul_acc_mat_vec_csr};

const MAX_ITERATIONS: usize = 1000;
const MAX_UNSTABLE: i32 = 50;

fn abs_max_diff(
    x: &[f32], y: &mut [f32], 
    epsold_: &mut [f32], unstable: &mut [i32]) -> f32 {
    let mut eps = 0.;
    for k in 0..y.len() {
        if (y[k] - x[k]).abs() < epsold_[k] || y[k] == 0. {
            unstable[k] = 0;
        } else {
            unstable[k] += 1;    
        }
        epsold_[k] = (y[k] - x[k]).abs();
        if (y[k] - x[k]).abs() > eps {
            //println!("y: {}, x: {}, k: {}", y[k], x[k], k);
            eps = (y[k] - x[k]).abs();
        }
        if unstable[k] > MAX_UNSTABLE && y[k] < 0. {
            y[k] = -f32::INFINITY;
        } 
    }
    eps
}

fn mo_abs_max_diff(
    x: &[f32], 
    y: &mut [f32], 
    epsold_: &mut [f32], 
    unstable: &mut [i32],
    k: usize,
    m: usize
) -> f32 {
    let mut eps = 0.;
    for ii in 0..y.len() {
        if (y[ii] - x[ii]).abs() < epsold_[k * m +  ii] || y[ii] == 0. {
            unstable[k * m + ii] = 0;
        } else {
            unstable[k * m + ii] += 1;    
        }
        epsold_[k * m + ii] = (y[ii] - x[ii]).abs();
        if (y[ii] - x[ii]).abs() > eps {
            //println!("y: {}, x: {}, k: {}", y[k], x[k], k);
            eps = (y[ii] - x[ii]).abs();
        }
        if unstable[k * m + ii] > MAX_UNSTABLE && y[ii] < 0. {
            y[ii] = -f32::INFINITY;
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

fn action_comparison2(
    y: &mut [f32],
    enabled_actions: &[i32],
    adj_sidx: &[i32],
    xold: &mut [f32],
    pi: &mut [i32],
) -> (bool, f32) {
    let mut max_eps: f32 = 0.;
    let mut policy_stable: bool = true;
    for r in 0..xold.len() {
        //let mut max_value = -f32::INFINITY;
        //let mut argmax_a: i32 = -1;
        for a in 0..enabled_actions[r] {
            if y[(adj_sidx[r] + a) as usize] > xold[r] {
                //max_value = y[(adj_sidx[r] + a) as usize];
                if y[(adj_sidx[r] + a) as usize] - xold[r] > max_eps {
                    //println!("max eps new: {}", max_eps)
                    max_eps = y[(adj_sidx[r] + a) as usize] - xold[r];
                }
                xold[r] = y[(adj_sidx[r] + a) as usize];
                //argmax_a = a;
                pi[r] = a;
                policy_stable = false;
            }
        }
        /*xnew[r] = max_value;
        if max_value - xold[r] > epsilon {
            policy_stable = false;
            pi[r] = argmax_a;
            max_eps = max_value - xold[r];
        }*/
    }
    (policy_stable, max_eps)
}

pub fn initial_policy(
    P: CsMatBase<f32, i32, &[i32], &[i32], &[f32]>, // This is a view of the transition matrix CSR fmt
    R: CsMatBase<f32, i32, &[i32], &[i32], &[f32]>, // This is a view of the rewards matrix CSR fmt
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
    let mut epsold_ = vec![0.; x.len()];
    let mut unstable = vec![0; x.len()];
    mul_acc_mat_vec_csr(R, w, &mut *r_v);
    let mut k: usize = 0;
    let mut i;

    while k < MAX_ITERATIONS && eps > epsilon && eps != f32::INFINITY {

        // compute the sparse matrix vector dot product of P.x and add it to r_v to
        // i.e. y = r + P.x
        i = 0;
        for y_ in y.iter_mut() {
            *y_ = r_v[i];
            i += 1;
        }
        //println!("y before  P.v: \n{:?}", y);
        mul_acc_mat_vec_csr(P, &*x, &mut *y);
        
        // check the difference between x and y
        eps = abs_max_diff(&x, y, &mut epsold_, &mut unstable);
        
        i = 0;
        for x_ in x.iter_mut() {
            *x_ = y[i];
            i += 1;
        }
        //println!("eps: {}, x:\n{:.2?}", eps, x);        
        k += 1;
    }
}

pub fn optimal_policy(
    P: CsMatBase<f32, i32, &[i32], &[i32], &[f32]>, // This is a view of the transition matrix CSR fmt
    R: CsMatBase<f32, i32, &[i32], &[i32], &[f32]>, // This is a view of the rewards matrix CSR fmt
    w: &[f32], 
    epsilon: f32, 
    r_v: &mut [f32],
    x: &mut [f32], // x is the output from the intial policy evaluation
    y: &mut [f32], // the size of x and y will be different. 
    pi: &mut [i32], // the size of the policy will be equivalent to |x|
    enabled_actions: &[i32],
    adj_sidx: &[i32],
    initial_state_index: usize
) -> f32 {
    let mut _eps = 1.0;
    let mut policy_stable = false;
    //let mut xtmp = vec![0.; x.len()];
    let mut epsold_ = vec![0.; x.len()];
    let mut unstable = vec![0; x.len()];
    // First thing is to construct the sparse rewards matrix R.w dot product
    mul_acc_mat_vec_csr(R, w, &mut *r_v);
    //let mut k: usize = 0;
    let mut i: usize;
    let mut num_loops = 0;
    while !policy_stable && num_loops < MAX_ITERATIONS {
    //while _eps > epsilon && num_loops < MAX_ITERATIONS {

        // compute the sparse matrix vector dot product of P.x and add it to r_v to
        // i.e. y = r + P.x
        i = 0;
        for y_ in y.iter_mut() {
            *y_ = r_v[i];
            i += 1;
        }
        //println!("y before  P.v: \n{:?}", y);
        mul_acc_mat_vec_csr(P, &*x, &mut *y);

        (policy_stable, _eps) = action_comparison2(y, enabled_actions, adj_sidx, x, pi);
        
        i = 0;
        /*for x_ in x.iter_mut() {
            *x_ = xtmp[i];
            i += 1;
        }*/

        //println!("eps: {}", _eps);
        
        num_loops += 1;
    }
    x[initial_state_index]
}

pub fn optimal_values(
    P: CsMatBase<f32, i32, &[i32], &[i32], &[f32]>, // This is a view of the transition matrix CSR fmt
    R: CsMatBase<f32, i32, &[i32], &[i32], &[f32]>, // This is a view of the rewards matrix CSR fmt
    epsilon: f32, 
    nobjs: usize
) -> Vec<f32> {
    // transitions
    let m = P.shape().0;
    let n = P.shape().1;
    let mut x: Vec<f32> = vec![0.; n];
    let mut y: Vec<f32> = vec![0.; m];
    let mut XStorage: Vec<f32> = vec![0.; m * nobjs];
    // rewards
    
    let mut r_v: Vec<f32> = vec![0.; R.shape().0];
    let mut RStorage: Vec<f32> = vec![0.; m * nobjs];
    let mut epsold_ = vec![0.; m * nobjs];
    let mut unstable = vec![0; m * nobjs];

    for k in 0..nobjs {
        let mut w: Vec<f32> = vec![0.; nobjs];
        w[k] = 1.0;
        // having RStorage saves us from having to reininit r to 
        // zeors ever loop of k
        for (i, r) in r_v.iter_mut().enumerate() {
            *r = RStorage[k * m + i];
        }
        mul_acc_mat_vec_csr(R, w, &mut *r_v);
        for (i, r) in RStorage[k * m..(k + 1) *m].iter_mut().enumerate() {
            *r = r_v[i];
        }
    }

    let mut ii: usize = 0;
    let mut k_eps = 0.;
    let mut eps = 1.0;
    while ii < MAX_ITERATIONS && eps > epsilon && eps != f32::INFINITY {
        eps = 0.;
        for k in 0..nobjs {
            for (i, y_) in y.iter_mut().enumerate() {
                *y_ = RStorage[k * m + i];
                x[i] = XStorage[k * m + i];
            }
            // compute the sparse matrix vector dot product of P.x and add it to r_v to
            // i.e. y = r + P.x
            mul_acc_mat_vec_csr(P, &*x, &mut *y);
            
            // check the difference between x and y
            k_eps = mo_abs_max_diff(&x, &mut y, &mut epsold_, &mut unstable, k, m);
            for (i, x_) in XStorage[k * m.. (k + 1) *m].iter_mut().enumerate() {
                *x_ = y[i];
            }
            /*if k == 7 {
                println!("\n{:?}", &XStorage[k * m.. (k + 1) *m]);
            }*/
            eps = f32::max(k_eps, eps);
            //println!("eps: {}", eps);
        }
        ii += 1;
    }
    XStorage
}