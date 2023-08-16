//use lp_modeler::format::lp_format::LpFileFormat;
use ordered_float::OrderedFloat;
use hashbrown::{HashSet, HashMap};
use crate::{model::{momdp::MOProductMDP, centralised::CTMDP, mamdp::MOMAMDP}, 
solvers::morap::cpu_only_solver, Debug, solvers::morap::gpu_only_solver, 
solvers::morap::hybrid_solver, solvers::{morap::ctmdp_gpu_solver, motap::motap_cpu_only_solver}, 
solvers::{morap::ctmdp_cpu_solver, motap::mamdp_cpu_solver}, new_target};
use std::{hash::Hash, time::Instant};
//use lp_modeler::solvers::{MiniLpSolver, SolverTrait};
//use lp_modeler::dsl::*;
use minilp::{Problem, OptimizationDirection, ComparisonOp};
use std::io::Write;
use termcolor::{Color, ColorChoice, ColorSpec, StandardStream, WriteColor};

const MAX_ITERATIONS: usize = 100;

pub enum HardwareChoice {
    Hybrid(usize), 
    CPU,
    GPU
}

pub fn scheduler_synthesis<S>(
    mut models_ls: Vec<MOProductMDP<S>>,
    num_agents: usize,
    num_tasks: usize,
    mut w: Vec<f32>,
    target: &[f32],
    epsilon1: f32, // value iteration threshold
    _epsilon2: f32, // scheduler synthesis threshold,
    cfg: HardwareChoice, // the system configuration for solving the problem
    dbug: Debug,
    max_iter: usize,
    max_unstable: i32,
    constraint_threshold: &[f32]
) -> Result<(HashMap<usize, Vec<f32>>, Vec<f32>, usize), String> 
where S: Copy + std::fmt::Debug + Eq + Hash + Send + Sync + 'static {
    // the first step of the scheduler synthesis is to process the
    // next w, starting with the input w
    //let nobjs: usize = num_agents + num_tasks;
    let mut tdown = target.to_vec();
    //let mut tup = vec![-f32::INFINITY; nobjs];
    let mut W: HashSet<Vec<OrderedFloat<f32>>> = HashSet::new();
    let mut Phi_: HashMap<usize, Vec<f32>> = HashMap::new();
    let mut A: HashMap<Vec<OrderedFloat<f32>>, Vec<f32>> = HashMap::new();
    //let largest_target_cost = compute_largest_cost(num_agents, &target[..]);
    // compute w's until we find the same w. 
    
    // find the value of the first w

    let mut r: Vec<f32>;
    //let mut w_prev = w.to_vec();
    let mut l = 0;
    let mut update_solution: bool; 
    let mut run_times: Vec<f32> = Vec::new();
    let mut rt: f32;
    let mut distance: f32 = 1.0;

    //while compute_eucl_dist(&tup, &tdown) > epsilon2 {
    for _ in 0..MAX_ITERATIONS {
        if !W.is_empty() {
            // compute a new y to find a new w
            let wnew = min_hyperplane_point_projection(&Phi_, &tdown);
            match wnew {
                Ok((d, w_ok)) => { 
                    (distance, w) = (d, w_ok);
                    match dbug {
                        Debug::Verbose1 => { println!("new w: {:?}", w); }
                        _ => { }
                    }
                }
                Err(e) => { 
                    match e {
                        minilp::Error::Infeasible => { return Ok((Phi_, run_times, l)); }
                        _ => { 
                            println!("Err: {:?}", e);
                            return Err(e.to_string()); 
                        }
                    }
                }
            }   
        }
        // distance convergence
        if distance < _epsilon2 {
            println!("================");
            println!("t[up] == t[down]");
            println!("================");
            return Ok((Phi_, run_times, l));
        }
        // insert the new w
        let w_: Vec<OrderedFloat<f32>> = w.iter().map(|x| OrderedFloat(*x)).collect();
        match W.insert(w_.to_vec()) {
            false => { 
                println!("target not achievable");
                update_solution = false;
                
            }
            true => { update_solution = true; }
        };
        // run a particular hardware selection
        let t1 = Instant::now();
        match cfg {
            HardwareChoice::Hybrid(CPU_COUNT) => { 
                (r, models_ls, rt) = hybrid_solver(models_ls, num_agents, num_tasks, &w, 
                    epsilon1, CPU_COUNT, dbug, max_iter, max_unstable);
                    run_times.push(rt);
                }
            HardwareChoice::GPU => { 
                //println!("calling gpu solver");
                (r, rt) = gpu_only_solver(&models_ls, num_agents, num_tasks, 
                    &w, epsilon1, dbug, max_iter as i32, max_unstable);
                run_times.push(rt);
            }
            HardwareChoice::CPU => { 
                (r, rt) = cpu_only_solver(&models_ls, num_agents, 
                    num_tasks, &w, epsilon1, dbug, max_iter, max_unstable);
                run_times.push(rt);
            }
        };
        match dbug {
            Debug::Base => { 
                println!("Time to do stage 1 + stage 2: {}", t1.elapsed().as_secs_f32());
            }
            Debug::Verbose1 => {
                println!("Time to do stage 1 + stage 2: {}", t1.elapsed().as_secs_f32());
                println!("r[{}]: {:?}", l, r);
            }
            _ => { }
        }
        
        if update_solution {
            match dbug {
                Debug::Verbose1 => {
                    println!("inserting w: {:?}, r: {:?} into A", w, r);
                }
                _ => { }
            }
            A.insert(w_, r.to_vec());
            Phi_.insert(l, r.to_vec());
            l += 1;
        }

        let c1 = dot(&w, &r);
        let c2 = dot(&w, &tdown); 
        if c1 < c2 && c2 - c1 > 1e-4 {
            // find a new tdown
            println!("wr: {} < wt: {}", c1, c2);
            // we have to do more here to find a new target
            let mut weights: Vec<Vec<f32>> = Vec::new();
            let mut hullset: Vec<Vec<f32>> = Vec::new();
            //let hullpoints = hullset.len();
            for (tw, tr) in A.iter() {
                weights.push(tw.iter().map(|x| x.into_inner()).collect());
                hullset.push(tr.to_vec());
            }
            match dbug {
                Debug::Verbose1 => {
                    println!("W: {:?}\nX: {:?}\nt: {:?}\nC: {:?}", 
                        weights, hullset, target, constraint_threshold);
                }
                _ => { }
            }
            match new_target(hullset, weights, target.to_vec(), num_agents, constraint_threshold.to_vec()) {
                Ok(z) => { 
                    tdown = z;
                    match dbug {
                        Debug::Verbose1 => {
                            println!("new target: {:?}", tdown);
                        }
                        _ => { }
                    }
                    
                }
                Err(e) => { println!("Err: {:?}", e); return Ok((Phi_, run_times, l)); }
            };  
        }
    }
    Ok((Phi_, run_times, l))
}

pub fn ctmdp_scheduler_synthesis<S>(
    ctmdp: CTMDP<S>,
    num_agents: usize,
    num_tasks: usize,
    mut w: Vec<f32>,
    target: &[f32],
    epsilon1: f32, // value iteration threshold
    _epsilon2: f32, // scheduler synthesis threshold,
    cfg: HardwareChoice, // the system configuration for solving the problem
    dbug: Debug,
    max_iter: usize,
    max_unstable: i32,
    constraint_threshold: &[f32]
) -> Result<(HashMap<usize, Vec<f32>>, Vec<f32>, usize), String> 
where S: Copy + std::fmt::Debug + Eq + Hash + Send + Sync + 'static {
    // the first step of the scheduler synthesis is to process the
    // next w, starting with the input w
    //let nobjs: usize = num_agents + num_tasks;
    let mut tdown = target.to_vec();
    //let mut tup = vec![-f32::INFINITY; nobjs];
    let mut W: HashSet<Vec<OrderedFloat<f32>>> = HashSet::new();
    let mut Phi_: HashMap<usize, Vec<f32>> = HashMap::new();
    let mut A: HashMap<Vec<OrderedFloat<f32>>, Vec<f32>> = HashMap::new();
    // compute w's until we find the same w. 
    
    // find the value of the first w

    let mut r: Vec<f32>;
    //let mut w_prev = w.to_vec();
    let mut l = 0;
    let mut update_solution: bool; 
    let mut run_times: Vec<f32> = Vec::new();
    let mut rt: f32;
    let mut distance: f32 = 1.0;    
    //while compute_eucl_dist(&tup, &tdown) > epsilon2 {
    for _ in 0..MAX_ITERATIONS {
        let t1 = Instant::now();
        if !W.is_empty() {
            // compute a new y to find a new w
            let wnew = min_hyperplane_point_projection(&Phi_, &tdown);
            match wnew {
                Ok((d, w_ok)) => { 
                    (distance, w) = (d, w_ok);
                    match dbug {
                        Debug::Verbose1 => { println!("new w: {:?}", w); }
                        _ => { }
                    }
                }
                Err(e) => { 
                    match e {
                        minilp::Error::Infeasible => { return Ok((Phi_, run_times, l)); }
                        _ => { 
                            println!("Err: {:?}", e);
                            return Err(e.to_string()); 
                        }
                    }
                }
            }   
        }
        if distance < _epsilon2 {
            println!("================");
            println!("t[up] == t[down]");
            println!("================");
            return Ok((Phi_, run_times, l));
        }
        // insert the new w
        let w_: Vec<OrderedFloat<f32>> = w.iter().map(|x| OrderedFloat(*x)).collect();
        match W.insert(w_.to_vec()) {
            false => { 
                println!("target not achievable");
                update_solution = false;
                
            }
            true => { update_solution = true; }
        };
        // run a particular hardware selection
        match cfg {
            HardwareChoice::Hybrid(_) => { 
                println!("Hybrid not supported for CTMDP");
                break;
            }
            HardwareChoice::GPU => { 
                (r, rt) = ctmdp_gpu_solver(&ctmdp, num_agents, num_tasks, 
                    &w, epsilon1, dbug, max_iter as i32, max_unstable);
                run_times.push(rt);
            }
            HardwareChoice::CPU => { 
                (r, rt) = ctmdp_cpu_solver(&ctmdp, num_agents, 
                    num_tasks, &w, epsilon1, dbug, max_iter, max_unstable);
                run_times.push(rt);
            }
        };
        match dbug {
            Debug::Base => { 
                println!("Time to do stage 1 + stage 2: {}", t1.elapsed().as_secs_f32());
            }
            Debug::Verbose1 => {
                println!("Time to do stage 1 + stage 2: {}", t1.elapsed().as_secs_f32());
                println!("r[{}]: {:?}", l, r);
            }
            _ => { }
        }
        
        if update_solution {
            match dbug {
                Debug::Verbose1 => {
                    println!("inserting w: {:?}, r: {:?} into A", w, r);
                }
                _ => { }
            }
            A.insert(w_, r.to_vec());
            Phi_.insert(l, r.to_vec());
            l += 1;
        }

        let c1 = dot(&w, &r);
        let c2 = dot(&w, &tdown); 
        if c1 < c2 {
            // find a new tdown
            println!("wr: {} < wt: {}", c1, c2);
            // we have to do more here to find a new target
            let mut weights: Vec<Vec<f32>> = Vec::new();
            let mut hullset: Vec<Vec<f32>> = Vec::new();
            //let hullpoints = hullset.len();
            for (tw, tr) in A.iter() {
                weights.push(tw.iter().map(|x| x.into_inner()).collect());
                hullset.push(tr.to_vec());
            }
            match new_target(hullset, weights, target.to_vec(), num_agents, constraint_threshold.to_vec()) {
                Ok(z) => { tdown = z;  }
                Err(e) => { println!("Err: {:?}", e); return Ok((Phi_, run_times, l)); }
            };
        }
    }
    //Ok((Phi_, run_times, l))
    Err("Max iter reached. No solution converged.".to_string())
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

//---------------------------------------------------------------------------------
//                         MOTAP SCHEDULER SYNTHESIS
//---------------------------------------------------------------------------------
pub fn mamdp_scheduler_synthesis<S>(
    mamdp: &MOMAMDP<S>,
    num_agents: usize,
    num_tasks: usize,
    mut w: Vec<f32>,
    target: &[f32],
    epsilon1: f32, // value iteration threshold
    _epsilon2: f32, // scheduler synthesis threshold,
    dbug: Debug,
    max_iter: usize,
    max_unstable: i32,
    constraint_threshold: &[f32]
) -> Result<(HashMap<usize, Vec<f32>>, Vec<f32>, usize, Option<Vec<f32>>), String> 
where S: Copy + std::fmt::Debug + Eq + Hash + Send + Sync + 'static {
    // the first step of the scheduler synthesis is to process the
    // next w, starting with the input w
    //let nobjs: usize = num_agents + num_tasks;
    let mut tdown = target.to_vec();
    //let mut tup = vec![-f32::INFINITY; nobjs];
    let mut W: HashSet<Vec<OrderedFloat<f32>>> = HashSet::new();
    let mut Phi_: HashMap<usize, Vec<f32>> = HashMap::new();
    let mut A: HashMap<Vec<OrderedFloat<f32>>, Vec<f32>> = HashMap::new();
    let mut stdout = StandardStream::stdout(ColorChoice::Always);
    // compute w's until we find the same w. 
    
    // find the value of the first w

    let mut r: Vec<f32>;
    //let mut w_prev = w.to_vec();
    let mut l = 0;
    let mut update_solution: bool; 
    let mut run_times: Vec<f32> = Vec::new();
    let mut rt: f32;
    let mut distance: f32 = 1.0;    
    //while compute_eucl_dist(&tup, &tdown) > epsilon2 {
    for _ in 0..MAX_ITERATIONS {
        let t1 = Instant::now();
        if !W.is_empty() {
            // compute a new y to find a new w
            let wnew = min_hyperplane_point_projection(&Phi_, &tdown);
            match wnew {
                Ok((d, w_ok)) => { 
                    (distance, w) = (d, w_ok);
                    match dbug {
                        Debug::Verbose1 => { println!("new w: {:?}", w); }
                        _ => { }
                    }
                }
                Err(e) => { 
                    match e {
                        minilp::Error::Infeasible => { 
                            match randomised_alloc_fn(&Phi_, &tdown[..]) {
                                Ok(v) => {
                                    return Ok((Phi_, run_times, l, Some(v))); 
                                }
                                Err(e) => {
                                    println!("Received error on building randomised allocation function!: {}", e);
                                    return Ok((Phi_, run_times, l, None));
                                }
                            };
                        }
                        _ => { 
                            println!("Err: {:?}", e);
                            return Err(e.to_string()); 
                        }
                    }
                }
            }   
        }
        if distance < _epsilon2 {
            println!("================");
            println!("t[up] == t[down]");
            println!("================");
            match randomised_alloc_fn(&Phi_, &tdown[..]) {
                Ok(v) => {
                    return Ok((Phi_, run_times, l, Some(v))); 
                }
                Err(e) => {
                    println!("Received error on building randomised allocation function!: {}", e);
                    return Ok((Phi_, run_times, l, None));
                }
            };
        }
        // insert the new w
        let w_: Vec<OrderedFloat<f32>> = w.iter().map(|x| OrderedFloat(*x)).collect();
        match W.insert(w_.to_vec()) {
            false => { 
                println!("target not achievable");
                update_solution = false;
                
            }
            true => { update_solution = true; }
        };
        // run a particular hardware selection
        (r, rt) = mamdp_cpu_solver(&mamdp, num_agents, 
            num_tasks, &w, epsilon1, dbug, max_iter, max_unstable);
        run_times.push(rt);
        match dbug {
            Debug::Base => { 
                println!("Time to do stage 1 + stage 2: {}", t1.elapsed().as_secs_f32());
            }
            Debug::Verbose1 => {
                println!("Time to do stage 1 + stage 2: {}", t1.elapsed().as_secs_f32());
                stdout.set_color(ColorSpec::new().set_fg(Some(Color::Red))).unwrap();
                writeln!(&mut stdout, "r[{}]: {:?}", l, r).unwrap();
                stdout.reset().unwrap();
            }
            _ => { }
        }
        
        if update_solution {
            match dbug {
                Debug::Verbose1 => {
                    println!("inserting w: {:?}, r: {:?} into A", w, r);
                }
                _ => { }
            }
            A.insert(w_, r.to_vec());
            Phi_.insert(l, r.to_vec());
            l += 1;
        }

        let c1 = dot(&w, &r);
        let c2 = dot(&w, &tdown); 
        if c1 < c2 && c2 - c1 > 1e-4 {
            // find a new tdown
            println!("wr: {} < wt: {}", c1, c2);
            // we have to do more here to find a new target
            let mut weights: Vec<Vec<f32>> = Vec::new();
            let mut hullset: Vec<Vec<f32>> = Vec::new();
            //let hullpoints = hullset.len();
            for (tw, tr) in A.iter() {
                weights.push(tw.iter().map(|x| x.into_inner()).collect());
                hullset.push(tr.to_vec());
            }
            match dbug {
                Debug::Verbose1 => {
                    /*println!("W: {:?}\nX: {:?}\nt: {:?}\nC: {:?}", 
                        weights, hullset, target, constraint_threshold);*/
                }
                _ => { }
            }
            match new_target(hullset, weights, target.to_vec(), num_agents, constraint_threshold.to_vec()) {
                Ok(z) => { 
                    tdown = z;
                    match dbug {
                        Debug::Verbose1 => {
                            //println!("new target: {:?}", tdown);
                            stdout.set_color(ColorSpec::new().set_fg(Some(Color::Green))).unwrap();
                            writeln!(&mut stdout, "new target: {:?}", tdown).unwrap();
                            stdout.reset().unwrap();
                        }
                        _ => { }
                    }
                    
                }
                Err(e) => { 
                    println!("Err: {:?}", e); 
                    match randomised_alloc_fn(&Phi_, &tdown[..]) {
                        Ok(v) => {
                            return Ok((Phi_, run_times, l, Some(v))); 
                        }
                        Err(e) => {
                            println!("Received error on building randomised allocation function!: {}", e);
                            return Ok((Phi_, run_times, l, None));
                        }
                    };
                }
            };  
        }
    }
    //Ok((Phi_, run_times, l))
    Err("Max iter reached. No solution converged.".to_string())
}

pub fn motap_scheduler_synthesis<S>(
    models_ls: Vec<MOProductMDP<S>>,
    num_agents: usize,
    num_tasks: usize,
    mut w: Vec<f32>,
    target: &[f32],
    epsilon1: f32, // value iteration threshold
    _epsilon2: f32, // scheduler synthesis threshold,
    dbug: Debug,
    max_iter: usize,
    max_unstable: i32,
    constraint_threshold: &[f32]
) -> Result<(HashMap<usize, Vec<f32>>, Vec<f32>, usize, Option<Vec<f32>>), String> 
where S: Copy + std::fmt::Debug + Eq + Hash + Send + Sync + 'static {
    // the first step of the scheduler synthesis is to process the
    // next w, starting with the input w
    //let nobjs: usize = num_agents + num_tasks;
    let mut tdown = target.to_vec();
    //let mut tup = vec![-f32::INFINITY; nobjs];
    let mut W: HashSet<Vec<OrderedFloat<f32>>> = HashSet::new();
    let mut Phi_: HashMap<usize, Vec<f32>> = HashMap::new();
    let mut A: HashMap<Vec<OrderedFloat<f32>>, Vec<f32>> = HashMap::new();
    let mut stdout = StandardStream::stdout(ColorChoice::Always);
    //let largest_target_cost = compute_largest_cost(num_agents, &target[..]);
    // compute w's until we find the same w. 
    
    // find the value of the first w

    let mut r: Vec<f32>;
    //let mut w_prev = w.to_vec();
    let mut l = 0;
    let mut update_solution: bool; 
    let mut run_times: Vec<f32> = Vec::new();
    let mut rt: f32;
    let mut distance: f32 = 1.0;

    //while compute_eucl_dist(&tup, &tdown) > epsilon2 {
    for _ in 0..MAX_ITERATIONS {
        if !W.is_empty() {
            // compute a new y to find a new w
            let wnew = min_hyperplane_point_projection(&Phi_, &tdown);
            match wnew {
                Ok((d, w_ok)) => { 
                    (distance, w) = (d, w_ok);
                    match dbug {
                        Debug::Verbose1 => { println!("new w: {:?}", w); }
                        _ => { }
                    }
                }
                Err(e) => { 
                    match e {
                        minilp::Error::Infeasible => { 
                            match randomised_alloc_fn(&Phi_, &tdown[..]) {
                                Ok(v) => {
                                    return Ok((Phi_, run_times, l, Some(v))); 
                                }
                                Err(e) => {
                                    println!("Received error on building randomised allocation function!: {}", e);
                                    return Ok((Phi_, run_times, l, None));
                                }
                            };
                            
                        }
                        _ => { 
                            println!("Err: {:?}", e);
                            return Err(e.to_string()); 
                        }
                    }
                }
            }   
        }
        // distance convergence
        if distance < _epsilon2 {
            println!("================");
            println!("t[up] == t[down]");
            println!("================");
            match randomised_alloc_fn(&Phi_, &tdown[..]) {
                Ok(v) => {
                    //println!("received something");
                    return Ok((Phi_, run_times, l, Some(v)));
                 }
                Err(e) => {
                    println!("Received error on building randomised allocation function!: {}", e);
                    return Ok((Phi_, run_times, l, None));
                }
            };
            
        }
        // insert the new w
        let w_: Vec<OrderedFloat<f32>> = w.iter().map(|x| OrderedFloat(*x)).collect();
        match W.insert(w_.to_vec()) {
            false => { 
                println!("target not achievable");
                update_solution = false;
                
            }
            true => { update_solution = true; }
        };
        // run a particular hardware selection
        (r, rt) = motap_cpu_only_solver(
            &models_ls, num_agents, num_tasks, &w, epsilon1, dbug, max_iter, max_unstable
        );
        run_times.push(rt);
        let t1 = Instant::now();
        
        match dbug {
            Debug::Base => { 
                println!("Time to do stage 1 + stage 2: {}", t1.elapsed().as_secs_f32());
            }
            Debug::Verbose1 => {
                println!("Time to do stage 1 + stage 2: {}", t1.elapsed().as_secs_f32());
                stdout.set_color(ColorSpec::new().set_fg(Some(Color::Red))).unwrap();
                writeln!(&mut stdout, "r[{}]: {:?}", l, r).unwrap();
                stdout.reset().unwrap();
            }
            _ => { }
        }
        
        if update_solution {
            match dbug {
                Debug::Verbose1 => {
                    println!("inserting w: {:?}, r: {:?} into A", w, r);
                }
                _ => { }
            }
            A.insert(w_, r.to_vec());
            Phi_.insert(l, r.to_vec());
            l += 1;
        }

        let c1 = dot(&w, &r);
        let c2 = dot(&w, &tdown); 
        if c1 < c2 && c2 - c1 > 1e-4 {
            // find a new tdown
            println!("wr: {} < wt: {}", c1, c2);
            // we have to do more here to find a new target
            let mut weights: Vec<Vec<f32>> = Vec::new();
            let mut hullset: Vec<Vec<f32>> = Vec::new();
            //let hullpoints = hullset.len();
            for (tw, tr) in A.iter() {
                weights.push(tw.iter().map(|x| x.into_inner()).collect());
                hullset.push(tr.to_vec());
            }
            /*match dbug {
                Debug::Verbose1 => {
                    println!("W: {:?}\nX: {:?}\nt: {:?}\nC: {:?}", 
                        weights, hullset, target, constraint_threshold);
                }
                _ => { }
            }*/
            match new_target(hullset, weights, target.to_vec(), num_agents, constraint_threshold.to_vec()) {
                Ok(z) => { 
                    tdown = z;
                    match dbug {
                        Debug::Verbose1 => {
                            //println!("new target: {:?}", tdown);
                            stdout.set_color(ColorSpec::new().set_fg(Some(Color::Green))).unwrap();
                            writeln!(&mut stdout, "new target: {:?}", tdown).unwrap();
                            stdout.reset().unwrap();
                        }
                        _ => { }
                    }
                    
                }
                Err(e) => { 
                    println!("Err: {:?}", e); 
                    match randomised_alloc_fn(&Phi_, &tdown[..]) {
                        Ok(v) => {
                            return Ok((Phi_, run_times, l, Some(v)));
                         }
                        Err(e) => {
                            println!("Received error on building randomised allocation function!: {}", e);
                            return Ok((Phi_, run_times, l, None));
                        }
                    };
                }
            };  
        }
    }
    match randomised_alloc_fn(&Phi_, &tdown[..]) {
        Ok(v) => { 
            return Ok((Phi_, run_times, l, Some(v)));
        }
        Err(e) => {
            println!("Received error on building randomised allocation function!: {}", e);
            return Ok((Phi_, run_times, l, None))
        }
    };
    
}

fn min_hyperplane_point_projection(
    hullset: &HashMap<usize, Vec<f32>>,
    target: &[f32]
) -> Result<(f32, Vec<f32>), minilp::Error> {

    let mut problem = Problem::new(OptimizationDirection::Maximize);

    let mut vars: HashMap<String, (usize, _)> = HashMap::new();

    for i in 0..target.len() {
        vars.insert(
            format!("w{}", i), 
            (i, problem.add_var(0.0, (0.00001, 1.0)))
        );
    }
    vars.insert(
        "d".to_string(), 
        (0, problem.add_var(1.0, (0., f64::INFINITY)))
    );

    //println!("hullset: {:?}", hullset);
    
    for j in 0..hullset.len() {
        //println!("j: {}", j);
        let r = &hullset.get(&j).unwrap()[..];
        problem.add_constraint(vars.iter()
            .map(|(k, (idx, var))| if k != "d" {
                    (*var, (target[*idx] - r[*idx]) as f64)
                } else {
                    (*var, -1.0)
                }
            ),
            ComparisonOp::Ge,
            0.0
        );
    }

    problem.add_constraint(vars.iter()
        .map(|(k, (_, var))| if k != "d" {
                (*var, 1.0)
            } else {
                (*var, 0.0)
            }
        ),
        ComparisonOp::Eq,
        1.0
    );

    let mut wnew: Vec<f32> = vec![0.; target.len()];
    let distance: f32;
    match problem.solve() {
        Ok(sol) => {
            distance = sol[vars.get("d").unwrap().1] as f32;
            println!("Distance: {:?}", distance);
            for i in 0..target.len() {
                wnew[i] = sol[vars.get(&format!("w{}", i)).unwrap().1] as f32;
            }
        }
        Err(e) => { 
            match e {
                minilp::Error::Infeasible => { 
                    println!("No new separating hyperplane possible");
                }
                _ => { }
            }
            return Err(e);
        }
    }

    Ok((distance, wnew))
}   


fn randomised_alloc_fn(
    hullset: &HashMap<usize, Vec<f32>>,
    target: &[f32]
) -> Result<Vec<f32>, minilp::Error> {

    let mut problem = Problem::new(OptimizationDirection::Maximize);

    let mut vars: HashMap<String, (usize, _)> = HashMap::new();

    for i in 0..hullset.len() {
        vars.insert(
            format!("w{}", i), 
            (i, problem.add_var(0.0, (0.0, 1.0)))
        );
    }

    /*vars.insert(
        "d".to_string(), 
        (0, problem.add_var(1.0, (0., 0.)))
    );*/

    for i in 0..target.len() {
        problem.add_constraint(
            vars.iter().map(|(_, (k, v))| 
                (*v, hullset.get(k).unwrap()[i] as f64)
            ),
            ComparisonOp::Ge,
            target[i] as f64 - 0.01
        )
    }

    problem.add_constraint(vars.iter()
        .map(|(_, (_, var))| (*var, 1.0)),
        ComparisonOp::Eq,
        1.0
    );

    

    let mut wnew: Vec<f32> = vec![0.; hullset.len()];
    match problem.solve() {
        Ok(sol) => {
            for i in 0..hullset.len() {
                wnew[i] = sol[vars.get(&format!("w{}", i)).unwrap().1] as f32;
            }
            println!("v: {:?}", wnew);
        }
        Err(e) => { 
            match e {
                minilp::Error::Infeasible => { 
                    // this should not happen
                    println!("No weighted combination possible");
                }
                _ => { }
            }
            return Err(e);
        }
    }

    Ok(wnew)
}  