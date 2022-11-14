/* 
TODO start with computing the initial value of the product

The requirements for the initial value include the matrix P,
matrix R an initial vector x, y to capture the values,

A special w vector which only inlcudes weights for the agents

A randomly initialised proper policy

An epsilon threshold for ending value iteration
*/

use crate::CxxMatrixf32;

pub fn initial_value(
    P: &CxxMatrixf32, 
    R: &CxxMatrixf32,
    init_policy: &[i32],
    w: &[f32],
    epsilon: f32 
) {
    let x: Vec<f32> = vec![0.; P.n as usize]; // Notice that x, y have different sizes
    let y: Vec<f32> = vec![0.; P.m as usize];


}
