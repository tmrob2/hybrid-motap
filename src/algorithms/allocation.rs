use lp_modeler::dsl::*;
//use lp_modeler::format::lp_format::LpFileFormat;
use lp_modeler::solvers::{SolverTrait, GlpkSolver, Status};
use hashbrown::HashMap;

pub fn allocation_fn(C: &[f32], n_agents: usize, n_tasks: usize, penalty: f32, 
    exp_val: f32) -> Result<Vec<i32>, String> {
    // let C be a vector of costs where c_ij is the cost of agent i for task j
    // this vector will be a 2d array flattened into a 1d array
    let agents: Vec<i32> = (0..n_agents as i32).collect();
    let tasks: Vec<i32> = (0..n_tasks as i32).collect();
    let task_ratio = 1.0_f32.max(n_tasks as f32 / n_agents as f32);

    let mut problem = LpProblem::new(
        "Allocation", 
        lp_modeler::dsl::LpObjective::Maximize
    );
    // Define the variables
    let vars: HashMap<(i32, i32), LpBinary> = agents.iter()
        .flat_map(|&a| tasks.iter()
            .map(move |&t| {
                let key = (a, t);
                let value = LpBinary::new(&format!("A{}_T{}", a, t));
                (key, value)
            }))
        .collect();
    let revvars: HashMap<String, (i32, i32)> = agents.iter()
        .flat_map(|&a| tasks.iter()
            .map(move |&t| {
                let key = format!("A{}_T{}", a, t);
                let value = (a, t);
                (key, value)
            }))
        .collect();
    let softvars: HashMap<i32, LpInteger> = agents.iter().map(|&a| {
        (a, LpInteger::new(&format!("z_{}", a)))
    }).collect();
    
    // Define the Objective Function
    let obj_vec: Vec<LpExpression> = {
        vars.iter().map(|(&(a, t), bin)| {
            let coef = C[n_agents * t as usize + a as usize];
            coef * bin
        })
    }.collect();
    let softpen: Vec<LpExpression> = {
        softvars.iter().map(|(_, var)| {
            -penalty * var
        })
    }.collect();

    problem += obj_vec.sum() + softpen.sum();
    
    // Define Constraints
    for &a in agents.iter() {
        problem += (sum(&tasks, |&t| vars.get(&(a, t)).unwrap())
            - softvars.get(&a).unwrap()).le(task_ratio)
    }
    for &t in tasks.iter() {
        problem += sum(&agents, |&a| vars.get(&(a, t)).unwrap()).equal(1);
    }

    for &a in agents.iter() {
        problem += softvars.get(&a).unwrap().ge(0);
    }

    // define the maximimsation constraint
    problem += obj_vec.sum().ge(exp_val - 1e-3);

    // Run the solver
    let solver = GlpkSolver::new();
    
    /*let result = problem.write_lp("problem.lp");
    match result{
        Ok(_) => println!("Written to file"),
        Err(msg) => println!("{}", msg)
    }*/
    
    let result = solver.run(&problem);
    
    let status = &result.as_ref().unwrap().status;
    let results = &result.as_ref().unwrap().results;
    let mut alloc: Vec<i32> = vec![0; n_tasks];
    for (var_name, var_value) in results {
        let int_var_value = *var_value as u32;
        if int_var_value == 1 && &var_name[..1] != "z" {
            //println!("{} = {}", var_name, int_var_value);
            let (a, t) = revvars.get(var_name).unwrap();
            alloc[*t as usize] = *a;
        }
    }
    println!("Status: {:?}", status);
    match status {
        Status::Optimal => { return Ok(alloc); }
        _ => { return Err(format!("Allocation function not solved with: {:?}", status)); }
    }

}