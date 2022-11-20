#![allow(non_snake_case)]
use sprs::{CsMatBase, TriMat};

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
    M: CsMatBase<f32, usize, &[usize], &[usize], &[f32]>, 
    pi: &[i32], 
    row_block: i32, 
    col_block: i32,
    sidx_offset: &[i32]
) -> CsMatBase<f32, usize, Vec<usize>, Vec<usize>, Vec<f32>> {

    //assert_eq!(row_block, pi.len() as i32);

    //println!("Argmax choices");
    //println!("row ptr: {:?}", M.indptr().raw_storage());
    //println!("col ind: {:?}", M.indices());
    //println!("vals: {:?}", M.data());
    let mut rows: Vec<usize> = Vec::new();
    let mut cols: Vec<usize> = Vec::new();
    let mut vals: Vec<f32> = Vec::new();

    for r in 0..row_block as usize {
        //let rowlookup = ridx[r] as usize;
        let mut rowlookup = sidx_offset[r] as usize;
        // now which action is it?
        rowlookup += pi[r] as usize;
        //println!("r: {} => r0: {}", r, rowlookup);
        let k = (M.indptr().raw_storage()[rowlookup + 1] - M.indptr().raw_storage()[rowlookup]) as usize;
        if k > 0 {
            for j_ in 0..k {
                // TODO we might need to interpret the j_ index here
                if M.data()[M.indptr().raw_storage()[rowlookup] as usize + j_] != 0. {
                    rows.push(r);
                    cols.push(M.indices()[M.indptr().raw_storage()[rowlookup]] + j_);
                    vals.push(M.data()[M.indptr().raw_storage()[rowlookup] + j_]);
                    /*newM.triple_entry(
                        r as i32, // the row will be different
                        M.p[M.i[rowlookup] as usize + j_], // the column will be the same column 
                        M.x[M.i[rowlookup] as usize + j_]
                    );*/
                }
            }
        }
    }

    let T = TriMat::from_triplets(
        (row_block as usize, col_block as usize), rows, cols, vals
    );
    T.to_csr()
}