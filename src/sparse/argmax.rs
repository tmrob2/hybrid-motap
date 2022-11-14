#![allow(non_snake_case)]
use hashbrown::HashMap;

use crate::CxxMatrixf32;
use super::compress;

/// argmax on a compressed row matrix
/// TODO this function is not quite correct because it doesn' take into accounr the 
/// new compressed action matrix data structure. 
/// 
/// 
/// Notes:
/// 
/// Need the starting row for each of the states, we can get this information from
/// the product MDP
/// 
/// Using the index of the policy to represent the row, the action chosen will 
/// be the offest from the initial state starting point in the HashMap input into
/// this function
/// 
/// The value we take from the larger matrix will then be the offset sidx, sidx and 
/// the probability of transition.
pub fn argmaxM(
    M: &CxxMatrixf32, 
    pi: &[i32], 
    row_block: i32, 
    col_block: i32,
    sidx_offset: &HashMap<i32, i32>
) -> CxxMatrixf32 {
    /*let ridx = pi.iter()
        .enumerate()
        .map(|(i, x)| i as i32 + *x * row_block)
        .collect::<Vec<i32>>();
    */
    let mut newM = CxxMatrixf32::new();
    let mut nz = 0;

    assert_eq!(row_block, pi.len() as i32);

    for r in 0..row_block as usize {
        //let rowlookup = ridx[r] as usize;
        let mut rowlookup = *sidx_offset.get(&(r as i32)).unwrap() as usize;
        // now which action is it?
        rowlookup += pi[r] as usize;
        let k = (M.i[rowlookup + 1] - M.i[rowlookup]) as usize;
        if k > 0 {
            for j_ in 0..k {
                if M.x[M.i[rowlookup] as usize + j_] != 0. {
                    newM.triple_entry(
                        r as i32, 
                        M.p[M.i[rowlookup] as usize + j_] - pi[r] * col_block, 
                        M.x[M.i[rowlookup] as usize + j_]
                    );
                    nz += 1;
                }
            }
        }
    }

    newM.nz = nz;
    newM.m = row_block;
    newM.n = col_block;
    newM.nzmax = nz;

    newM = compress::compress(newM);
    newM
}