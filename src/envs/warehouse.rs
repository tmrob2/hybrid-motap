use pyo3::prelude::*;
use hashbrown::{HashSet, HashMap};
use array_macro::array;
use pyo3::exceptions::PyValueError;
use sprs::CsMatBase;
use crate::agent::env::Env;
use crate::{product, cuda_initial_policy_value, cuda_policy_optimisation, cuda_warm_up_gpu, cuda_initial_policy_value_pinned_graph, warm_up_gpu, allocation_fn, cuda_multi_obj_solution};
use crate::sparse::argmax::argmaxM;
use crate::model::momdp::{product_mdp_bfs, choose_random_policy};
use crate::model::scpm::SCPM;
use crate::model::momdp::MOProductMDP;
use crate::algorithms::dp::{initial_policy, optimal_policy};
use std::time::Instant;
use rayon::prelude::*;

type Point = (i8, i8);
type State = (Point, u8, Option<Point>);


//-----------------
// Python Interface
//-----------------
#[pyclass]
pub struct Warehouse {
    pub current_task: Option<usize>,
    pub size: usize,
    pub nagents: usize,
    pub feedpoints: Vec<Point>,
    #[pyo3(get)]
    pub racks: HashSet<Point>,
    pub original_racks: HashSet<Point>,
    pub agent_initial_locs: Vec<Point>,
    pub action_to_dir: HashMap<u8, [i8; 2]>,
    pub task_racks_start: HashMap<i32, Point>, // What is this?
    pub task_racks_end: HashMap<i32, Point>, // What is this?
    pub task_feeds: HashMap<i32, Point>,
    pub words: [String; 25],
    pub psuccess: f32
}

#[pymethods]
impl Warehouse {
    #[new]
    fn new(
        size: usize, 
        nagents: usize, 
        feedpoints: Vec<Point>, 
        initial_locations: Vec<Point>,
        actions_to_dir: Vec<[i8; 2]>,
        psuccess: f32
    ) -> Self {
        // call the rack function
        let mut action_map: HashMap<u8, [i8; 2]> = HashMap::new();
        for (i, act) in actions_to_dir.into_iter().enumerate() {
            action_map.insert(i as u8, act);
        }
        let w0: [String; 5] = ["RS".to_string(), "RE".to_string(), "F".to_string(), "FR".to_string(), "NFR".to_string()];
        let w1: [String; 5] = ["P".to_string(), "D".to_string(), "CR".to_string(), "CNR".to_string(), "NC".to_string()];
        let mut words: [String; 25] = array!["".to_string(); 25];
        let mut count: usize = 0;
        for wa in w0.iter() {
            for wb in w1.iter() {
                words[count] = format!("{}_{}", wa, wb);
                count += 1;
            }
        }
        //let actions: [u8; 7] = array![i => i as u8; 7];
        let mut new_env = Warehouse {
            current_task: None,
            size,
            nagents,
            feedpoints,
            racks: HashSet::new(),
            original_racks: HashSet::new(),
            agent_initial_locs: initial_locations,
            action_to_dir: action_map,
            task_racks_start: HashMap::new(),
            task_racks_end: HashMap::new(),
            task_feeds: HashMap::new(),
            words,
            psuccess
        };
        // TODO finish testing of different minimal warehouse layouts.
        new_env.racks = new_env.place_racks(size);
        // test: => new_env.racks.insert((0, 1));
        //new_env.racks.insert((0, 1));
        //new_env.racks.insert((0, 2));
        new_env
    }

    fn set_psuccess(&mut self, pnew: f32) {
        self.psuccess = pnew;
    }

    fn update_rack(&mut self, new_rack_pos: Point, old_rack_pos: Point) {
        println!("Recieved a rack update: new: {:?}, old: {:?}", new_rack_pos, old_rack_pos);
        self.racks.remove(&old_rack_pos);
        self.racks.insert(new_rack_pos);
    }

    fn add_task_rack_start(&mut self, task: i32, rack: Point) {
        self.task_racks_start.insert(task, rack);
    }

    fn remove_task_rack_start(&mut self, task: i32) {
        self.task_racks_start.remove(&task);
    }

    fn add_task_rack_end(&mut self, task: i32, rack: Point) {
        self.task_racks_end.insert(task, rack);
    }

    fn remove_task_rack_end(&mut self, task: i32) {
        self.task_racks_start.remove(&task);
    }

    
    fn add_task_feed(&mut self, task: i32, feed: Point) {
        self.task_feeds.insert(task, feed);
    }
    
    fn remove_task_feed(&mut self, task: i32) {
        self.task_feeds.remove(&task);
    }

    fn step(&self, state: State, action: u8, task_id: i32) -> PyResult<Vec<(State, f32, String)>> {
        let v = match self.step_(state, action, task_id){
            Ok(result) => { result }
            Err(e) => {
                return Err(PyValueError::new_err(format!(
                    "Could not step the environment forward => {:?}", e
                )))
            }
        };
        Ok(v)
    }

    fn get_words(&self) -> [String; 25] {
        self.words.clone()
    }
}

impl Warehouse {
    fn place_racks(&self, size: usize) 
        -> HashSet<Point> {
        let mut racks: HashSet<Point> = HashSet::new();
        let mut count: usize = 0;
        assert!(size > 5);
        assert!(size > 4);
        for c in 2..size - 2 {
            if count < 2 {
                for r in 2..size - 2 { 
                    racks.insert((c as i8, r as i8));
                }
                count += 1;
            } else {
                count = 0;
            }
        }
        //racks.insert((1, 1));
        racks
    }

    fn pos_addition(
        &self,
        pos: &Point, 
        dir: &[i8; 2],
        size_max: i8
    ) -> Point {
        let pos_: [i8; 2] = [pos.0, pos.1];
        let mut add_ = Self::new_coord_from(pos_.iter().zip(dir).map(|(a, b)| a + b));
        for a in add_.iter_mut() {
            if *a < 0 {
                *a = 0;
            } else if *a > size_max - 1 {
                *a = size_max - 1;
            }
        }
        (add_[0], add_[1])
    }

    fn new_coord_from<F: Iterator<Item=i8>>(src: F) -> [i8; 2] {
        let mut result = [0; 2];
        for (rref, val) in result.iter_mut().zip(src) {
            *rref = val;
        }
        result
    }

    fn process_word(&self, current_state: &State, new_state: &State, current_task: i32) -> String {
        // The 'character' of the 'word' should be the rack position
        // i.e. did the agent move to the rack corresponding to the task
        // 
        // The second 'character' in the 'word' should be the status of whether the agent
        // picked something up or not
        let mut word: [&str; 2] = [""; 2];
        if current_state.1 == 0 && new_state.1 == 1 {
            // add pick up
            word[1] = "P";
        } else if current_state.1 == 1 && new_state.1 == 0 {
            // add drop
            word[1] = "D";
        } else if current_state.1 == 1 && new_state.1 == 1 {
            //let current_task = self.get_current_task();
            if self.racks.contains(&new_state.0) {
                // this is a problem...but only if the rack is not the task rack
                if *self.task_racks_start.get(&(current_task)).unwrap() == new_state.0 {
                    word[1] = "CNR";
                } else if self.task_racks_end.get(&(current_task)).is_some() && 
                    *self.task_racks_end.get(&(current_task)).unwrap() == new_state.0 {
                    // this is fine
                    word[1] = "CNR";
                } else {
                    word[1] = "CR";
                }
            } else {
                word[1] = "CNR";
            }
        } else {
            word[1] = "NC";
        }

    
        if new_state.0 == *self.task_racks_start.get(&current_task).unwrap() {
            // then the rack is in position
            word[0] = "RS";
        } else if self.task_racks_end.get(&current_task).is_some() && 
            new_state.0 == *self.task_racks_end.get(&(current_task as i32)).unwrap()
        {
            word[0] = "RE";
        } else if new_state.0 == *self.task_feeds.get(&current_task)
            .expect(&format!("Could not find task: {}", current_task)) {
            word[0] = "F";
        } else if current_state.2.is_some() && 
            current_state.2.unwrap() == new_state.0 && current_state.1 == 0 {
            word[0] = "RS";
        } else {
            word[0] = "NFR";
        }
        format!("{}_{}", word[0], word[1])
    }
}

impl Env<State> for Warehouse {
    fn step_(&self, state: State, action: u8, task_id: i32) -> Result<Vec<(State, f32, String)>, String> {
        let psuccess :f32 = self.psuccess;
        let mut v: Vec<(State, f32, String)> = Vec::new();
        if vec![0, 1, 2, 3].contains(&action) {
            let direction: &[i8; 2] = self.action_to_dir.get(&action).unwrap();
            
            let agent_new_loc = self.pos_addition( 
                &state.0, 
                direction, 
                self.size as i8, 
            );
            
            if state.1 == 0 {
                let new_state = (agent_new_loc, 0, state.2);
                // TODO we need to convert the words to how they should be represented with 
                // respect to the DFA because this is a labelled MDP
                let w = self.process_word(&state, &new_state, task_id);
                v.push((new_state, 1.0, w));
            } else {
                // an agent is restricted to move with corridors
                // check that the agent is in a corridor, if not, then
                // it cannot proceed in a direction that is not a corridor
                if self.racks.contains(&state.0) {
                    if self.racks.contains(&agent_new_loc) {
                        let w = self.process_word(&state, &state, task_id);
                        v.push((state, 1.0, w));
                    } else {
                        // Define the failure scenario
                        let success_rack_pos = Some(agent_new_loc);
                        let success_state: State = (agent_new_loc, 1, success_rack_pos);
                        let success_word = self.process_word(&state, &success_state, task_id);
                        //println!("{:?}, {} -> {:?} => {}", 
                        //    state, action, success_state, success_word
                        //);
                        let fail_state: State = (agent_new_loc, 0, state.2);
                        let fail_word = self.process_word(&state, &fail_state, task_id);

                        v.push((success_state, psuccess, success_word));
                        v.push((fail_state, 1. - psuccess, fail_word));
                    }
                } else {
                    // Define the failure scenario
                    let success_rack_pos = Some(agent_new_loc);
                    let success_state: State = (agent_new_loc, 1, success_rack_pos);
                    let success_word = self.process_word(&state, &success_state, task_id);
                    //println!("{:?}, {} -> {:?} => {}", 
                    //    state, action, success_state, success_word
                    //);
                    let fail_state: State = (agent_new_loc, 0, state.2);
                    let fail_word = self.process_word(&state, &fail_state, task_id);
                    
                    v.push((success_state, psuccess, success_word));
                    v.push((fail_state, 1. - psuccess, fail_word));
                }
            };
        } else if action == 4 {
            if state.1 == 0 {
                // if the agent is in a rack position then it may carry a rack
                // OR is the agent is not in a rack position but is superimposed on 
                // a rack that is in a corridor then it may pick up the rack. 
                let cmp_state = state.2;
                if self.racks.contains(&state.0) {
                    let new_rack_pos = Some(state.0);
                    let new_state = (state.0, 1, new_rack_pos);
                    let word = self.process_word(&state, &new_state, task_id);
                    v.push((new_state, 1.0, word));
                } else if cmp_state.is_some() {
                    if cmp_state.unwrap() == state.0 {
                        let new_rack_pos = Some(state.0);
                        let new_state = (state.0, 1, new_rack_pos);
                        let word = self.process_word(&state, &new_state, task_id);
                        v.push((new_state, 1.0, word));
                    } else {
                        let new_state = (state.0, 0, state.2);
                        let word = self.process_word(&state, &new_state, task_id);
                        v.push((new_state, 1.0, word));
                    }
                } else {
                    let new_state = (state.0, 0, state.2);
                    let word = self.process_word(&state, &new_state, task_id);
                    v.push((new_state, 1.0, word));
                }
            } else {
                let word = self.process_word(&state, &state, task_id);
                v.push((state, 1.0, word));
            }
            //println!("{:?} -> {:?}", state, v);
        } else if action == 5 {
            if state.1 == 1 {
                // this agent is currently carrying something
                // therefore, drop the rack at the current agent position
                let new_state = (state.0, 0, None);
                let word = self.process_word(&state, &new_state, task_id);
                v.push((new_state, 1.0, word));
            } else {
                let word = self.process_word(&state, &state, task_id);
                v.push((state, 1.0, word));
            }
        } else if action == 6 {
            // do nothing
            let word = self.process_word(&state, &state, task_id);
            v.push((state, 1.0, word));
        } else {
            return Err("action not registered".to_string())
        }
        Ok(v)
    }

    fn get_init_state(&self, agent: usize) -> State {
        // with the current task construct the initial state
        let agent_pos = self.agent_initial_locs[agent as usize];
        (agent_pos, 0, None)
    }

    fn set_task(&mut self, task_id: usize) {
        self.current_task = Some(task_id);
    }

    fn get_action_space(&self) -> Vec<i32> {
        todo!()
    }

    fn get_states_len(&self) -> usize {
        todo!()
    }
}

#[pyfunction]
pub fn warehouse_build_test(
    model: &SCPM,
    env: &mut Warehouse,
    w: Vec<f32>,
    eps: f32,
    //mut outputs: MDPOutputs
) -> () //(Vec<f32>, MDPOutputs)
where Warehouse: Env<State> {
    let t1 = Instant::now();
    println!("Constructing MDP of agent environment");
    //model.construct_products(&mut mdp);
    let pmdp = product_mdp_bfs(
        (env.get_init_state(0), 0), 
        env,
        &model.tasks.get_task(0), 
        0, 
        0, 
        model.num_agents, 
        model.num_agents + model.tasks.size,
        &model.actions
    );
    println!("Elapsed time to build warehouse model: {:?}", t1.elapsed().as_secs_f32());
    println!("ProdMDP |S|: {}", pmdp.states.len());
    println!("ProdMPD |P|: {}", pmdp.P.nnz());
}

#[pyfunction]
pub fn test_warehouse_policy_optimisation(
    model: &SCPM,
    env: &mut Warehouse,
    w: Vec<f32>,
    eps: f32
) -> () //(Vec<f32>, MDPOutputs)
where Warehouse: Env<State> {
    let t1 = Instant::now();
println!(
"--------------------------\n
        CPU TEST           \n
--------------------------"
);
    println!("Constructing MDP of agent environment");
    //model.construct_products(&mut mdp);
    let pmdp = product_mdp_bfs(
        (env.get_init_state(0), 0), 
        env,
        &model.tasks.get_task(0), 
        0, 
        0, 
        model.num_agents, 
        model.num_agents + model.tasks.size,
        &model.actions
    );

    let mut w_init = vec![0.; model.num_agents + model.tasks.size];
    for k in 0..model.num_agents {
        w_init[k] = 1.;
    }
    
    
    println!("Elapsed time to build warehouse model: {:?}", t1.elapsed().as_secs_f32());
    println!("ProdMDP |S|: {}", pmdp.states.len());
    println!("ProdMPD |P|: {}", pmdp.P.nnz());

    let mut pi = choose_random_policy(&pmdp);

    let rowblock = pmdp.states.len() as i32;
    let pcolblock = rowblock as i32;
    let rcolblock = (model.num_agents + model.tasks.size) as i32;
    let initP = 
        argmaxM(pmdp.P.view(), &pi, rowblock, pcolblock, &pmdp.adjusted_state_act_pair);
    let initR = 
        argmaxM(pmdp.R.view(), &pi, rowblock, rcolblock, &pmdp.adjusted_state_act_pair);

    // using the random policy determine the value of the intial policy
    let mut r_v: Vec<f32> = vec![0.; initR.shape().0 as usize];
    let mut x: Vec<f32> = vec![0.; initP.shape().1 as usize];
    let mut y: Vec<f32> = vec![0.; initP.shape().0 as usize];

    let t2 = Instant::now();
    
    initial_policy(initP.view(), initR.view(), &w_init, eps, &mut r_v, &mut x, &mut y);

    println!("Build + initial policy: {:?} (s)", t1.elapsed().as_secs_f32());
    println!("initial policy only: {:?} (s)", t2.elapsed().as_secs_f32());
    println!("initial policy value: {:?}", x[*pmdp.state_map.get(&pmdp.initial_state).unwrap()]);
    
    let t3 = Instant::now();
    let mut r_v: Vec<f32> = vec![0.; pmdp.R.shape().0];
    let mut y: Vec<f32> = vec![0.; pmdp.P.shape().0];

    let val = optimal_policy(pmdp.P.view(), pmdp.R.view(), &w, eps, 
        &mut r_v, &mut x, &mut y, &mut pi, &pmdp.enabled_actions, 
        &pmdp.adjusted_state_act_pair, 
        *pmdp.state_map.get(&pmdp.initial_state).unwrap()
    );
    println!("Optimal policy computation: {:?} (s) = {}", t3.elapsed().as_secs_f32(), val);
}

#[pyfunction]
pub fn test_warehouse_model_size(
    model: &SCPM,
    env: &mut Warehouse,
    w: Vec<f32>,
    eps: f32,
    //mut outputs: MDPOutputs
) -> () //(Vec<f32>, MDPOutputs)
where Warehouse: Env<State> {
    let t1 = Instant::now();
    println!("Constructing MDP of agent environment");
    //model.construct_products(&mut mdp);
    let pairs = 
        product(0..model.num_agents, 0..model.tasks.size);
    let output: Vec<MOProductMDP<State>> = pairs.into_par_iter().map(|(a, t)| {
        let pmdp = product_mdp_bfs(
            (env.get_init_state(a), 0), 
            env,
            &model.tasks.get_task(t), 
            a as i32, 
            t as i32, 
            model.num_agents, 
            model.num_agents + model.tasks.size,
            &model.actions
        );
        pmdp
    }).collect(); 
    println!("Time to create {} models: {:?}", output.len(), t1.elapsed().as_secs_f32())
}

#[pyfunction]
pub fn test_warehouse_gpu_policy_optimisation(
    model: &SCPM,
    env: &mut Warehouse,
    w: Vec<f32>,
    eps: f32
) -> () //(Vec<f32>, MDPOutputs)
where Warehouse: Env<State> {
    // always call warm up first
    cuda_warm_up_gpu();
    let t1 = Instant::now();
println!(
"--------------------------\n
        GPU TEST           \n
--------------------------"
);
    println!("Constructing MDP of agent environment");
    //model.construct_products(&mut mdp);
    let pmdp = product_mdp_bfs(
        (env.get_init_state(0), 0), 
        env,
        &model.tasks.get_task(0), 
        0, 
        0, 
        model.num_agents, 
        model.num_agents + model.tasks.size,
        &model.actions
    );

    let mut w_init = vec![0.; model.num_agents + model.tasks.size];
    for k in 0..model.num_agents {
        w_init[k] = 1.;
    }
    
    println!("Elapsed time to build warehouse model: {:?}", t1.elapsed().as_secs_f32());
    println!("ProdMDP |S|: {}", pmdp.states.len());
    println!("ProdMPD |P|: {}", pmdp.P.nnz());

    let mut pi = choose_random_policy(&pmdp);

    let rowblock = pmdp.states.len() as i32;
    let pcolblock = rowblock as i32;
    let rcolblock = (model.num_agents + model.tasks.size) as i32;
    let initP = 
        argmaxM(pmdp.P.view(), &pi, rowblock, pcolblock, &pmdp.adjusted_state_act_pair);
    let initR = 
        argmaxM(pmdp.R.view(), &pi, rowblock, rcolblock, &pmdp.adjusted_state_act_pair);

    // using the random policy determine the value of the intial policy
    let mut r_v: Vec<f32> = vec![0.; initR.shape().0 as usize];
    let mut x: Vec<f32> = vec![0.; initP.shape().1 as usize];
    let mut y: Vec<f32> = vec![0.; initP.shape().0 as usize];
    let mut unstable: Vec<i32> = vec![0; initP.shape().0 as usize];

    let t2 = Instant::now();
    
    cuda_initial_policy_value(initP.view(), initR.view(), &w_init, eps, 
                              &mut r_v, &mut x, &mut y, &mut unstable);

    println!("Build + initial policy: {:?} (s)", t1.elapsed().as_secs_f32());
    println!("initial policy only: {:?} (s)", t2.elapsed().as_secs_f32());
    println!("initial policy value: {:?}", x[*pmdp.state_map.get(&pmdp.initial_state).unwrap()]);

    let t3 = Instant::now();
    let mut gpux = x.to_vec();
    let mut r_v: Vec<f32> = vec![0.; pmdp.R.shape().0];
    let mut y: Vec<f32> = vec![0.; pmdp.P.shape().0];
    let mut stable: Vec<f32> = vec![0.; x.len()];

    println!("|x|: {}", x.len());
    println!("|y|: {}", pmdp.P.shape().0);
    println!("P cols: {}", pmdp.P.shape().1);
    let gpuval = cuda_policy_optimisation(pmdp.P.view(), pmdp.R.view(), &w, eps, &mut pi,
        &mut gpux, &mut y, &mut r_v, 
        &pmdp.enabled_actions, &pmdp.adjusted_state_act_pair, 
        *pmdp.state_map.get(&pmdp.initial_state).unwrap(),
        &mut stable
    );
    println!("Optimal policy computation: {:?} (s) = {}", 
        t3.elapsed().as_secs_f32(), gpuval);
}

#[pyfunction]
pub fn test_warehouse_CPU_only(
    model: &SCPM,
    env: &mut Warehouse,
    w: Vec<f32>,
    eps: f32,
    //mut outputs: MDPOutputs
) -> () //(Vec<f32>, MDPOutputs)
where Warehouse: Env<State> {
    let t1 = Instant::now();
    println!("Constructing MDP of agent environment");
    //model.construct_products(&mut mdp);
    let pairs = 
        product(0..model.num_agents, 0..model.tasks.size);
    let models_ls: Vec<MOProductMDP<State>> = pairs.into_par_iter().map(|(a, t)| {
        let pmdp = product_mdp_bfs(
            (env.get_init_state(a), 0), 
            env,
            &model.tasks.get_task(t), 
            a as i32, 
            t as i32, 
            model.num_agents, 
            model.num_agents + model.tasks.size,
            &model.actions
        );
        pmdp
    }).collect(); 
    println!("Time to create {} models: {:?}\n|S|: {},\n|P|: {}", 
        models_ls.len(), t1.elapsed().as_secs_f32(),
        models_ls.iter().fold(0, |acc, m| acc + m.P.shape().1),
        models_ls.iter().fold(0, |acc, m| acc + m.P.nnz())
    );

    let t2 = Instant::now();
    // Input all of the models into the rayon framework
    let _output: Vec<f32> = models_ls.into_par_iter().map(|pmdp| {

        let mut pi = choose_random_policy(&pmdp);

        let rowblock = pmdp.states.len() as i32;
        let pcolblock = rowblock as i32;
        let rcolblock = (model.num_agents + model.tasks.size) as i32;
        let initP = 
            argmaxM(pmdp.P.view(), &pi, rowblock, pcolblock, 
                    &pmdp.adjusted_state_act_pair);
        let initR = 
            argmaxM(pmdp.R.view(), &pi, rowblock, rcolblock, 
                    &pmdp.adjusted_state_act_pair);

        let mut r_v: Vec<f32> = vec![0.; initR.shape().0];
        let mut x: Vec<f32> = vec![0.; initP.shape().1];
        let mut y: Vec<f32> = vec![0.; initP.shape().0];
        
        initial_policy(initP.view(), initR.view(), &w, eps, 
                    &mut r_v, &mut x, &mut y);

        // taking the initial policy and the value vector for the initial policy
        // what is the optimal policy
        let mut r_v: Vec<f32> = vec![0.; pmdp.R.shape().0];
        let mut y: Vec<f32> = vec![0.; pmdp.P.shape().0];
        let r = optimal_policy(pmdp.P.view(), pmdp.R.view(), &w, eps, 
                    &mut r_v, &mut x, &mut y, &mut pi, 
                    &pmdp.enabled_actions, &pmdp.adjusted_state_act_pair,
                    *pmdp.state_map.get(&pmdp.initial_state).unwrap()
                    );
        r
    }).collect();
    println!("Time to do stage 1 {}", t2.elapsed().as_secs_f32());
    println!("Total runtime {}", t2.elapsed().as_secs_f32());

}

#[pyfunction]
pub fn test_warehouse_GPU_only(
    model: &SCPM,
    env: &mut Warehouse,
    w: Vec<f32>,
    eps: f32,
    //mut outputs: MDPOutputs
) -> () //(Vec<f32>, MDPOutputs)
where Warehouse: Env<State> {
    let t1 = Instant::now();
    println!("Constructing MDP of agent environment");
    //model.construct_products(&mut mdp);
    let pairs = 
        product(0..model.num_agents, 0..model.tasks.size);
    let models_ls: Vec<MOProductMDP<State>> = pairs.into_par_iter().map(|(a, t)| {
        let pmdp = product_mdp_bfs(
            (env.get_init_state(a), 0), 
            env,
            &model.tasks.get_task(t), 
            a as i32, 
            t as i32, 
            model.num_agents, 
            model.num_agents + model.tasks.size,
            &model.actions
        );
        pmdp
    }).collect(); 
    println!("Time to create {} models: {:?}\n|S|: {},\n|P|: {}", 
        models_ls.len(), t1.elapsed().as_secs_f32(),
        models_ls.iter().fold(0, |acc, m| acc + m.P.shape().1),
        models_ls.iter().fold(0, |acc, m| acc + m.P.nnz())
    );

    let mut w_init = vec![0.; model.num_agents + model.tasks.size];
    for k in 0..model.num_agents {
        w_init[k] = 1.;
    }

    let t2 = Instant::now();
    for pmdp in models_ls.iter() {
        let mut pi = choose_random_policy(pmdp);

        let rowblock = pmdp.states.len() as i32;
        let pcolblock = rowblock as i32;
        let rcolblock = (model.num_agents + model.tasks.size) as i32;
        let initP = 
            argmaxM(pmdp.P.view(), &pi, rowblock, pcolblock, &pmdp.adjusted_state_act_pair);
        let initR = 
            argmaxM(pmdp.R.view(), &pi, rowblock, rcolblock, &pmdp.adjusted_state_act_pair);

        // using the random policy determine the value of the intial policy
        let mut r_v: Vec<f32> = vec![0.; initR.shape().0 as usize];
        let mut x: Vec<f32> = vec![0.; initP.shape().1 as usize];
        let mut y: Vec<f32> = vec![0.; initP.shape().0 as usize];
        let mut unstable: Vec<i32> = vec![0; initP.shape().0 as usize];

        let t2 = Instant::now();
        
        cuda_initial_policy_value(initP.view(), initR.view(), &w_init, eps, 
                                &mut r_v, &mut x, &mut y, &mut unstable);

        let mut gpux = x.to_vec();
        let mut r_v: Vec<f32> = vec![0.; pmdp.R.shape().0];
        let mut y: Vec<f32> = vec![0.; pmdp.P.shape().0];
        let mut stable: Vec<f32> = vec![0.; x.len()];

        let _gpuval = cuda_policy_optimisation(pmdp.P.view(), pmdp.R.view(), &w, eps, &mut pi,
            &mut gpux, &mut y, &mut r_v, 
            &pmdp.enabled_actions, &pmdp.adjusted_state_act_pair, 
            *pmdp.state_map.get(&pmdp.initial_state).unwrap(),
            &mut stable
        );
    }

    println!("Time to do stage 1 {}", t2.elapsed().as_secs_f32());
    println!("Total runtime {}", t2.elapsed().as_secs_f32());

}

#[pyfunction]
pub fn test_gpu_stream(
    model: &SCPM,
    env: &mut Warehouse,
    w: Vec<f32>,
    eps: f32,
) {
    cuda_warm_up_gpu();
    let t1 = Instant::now();
    let pairs = 
        product(0..model.num_agents, 0..model.tasks.size);
    let models_ls: Vec<MOProductMDP<State>> = pairs.into_par_iter().map(|(a, t)| {
        let pmdp = product_mdp_bfs(
            (env.get_init_state(a), 0), 
            env,
            &model.tasks.get_task(t), 
            a as i32, 
            t as i32, 
            model.num_agents, 
            model.num_agents + model.tasks.size,
            &model.actions
        );
        pmdp
    }).collect(); 

    let mut w_init = vec![0.; model.num_agents + model.tasks.size];
    for k in 0..model.num_agents {
        w_init[k] = 1.;
    }

    println!("Time to create {} models: {:?}\n|S|: {},\n|P|: {}", 
        models_ls.len(), t1.elapsed().as_secs_f32(),
        models_ls.iter().fold(0, |acc, m| acc + m.P.shape().1),
        models_ls.iter().fold(0, |acc, m| acc + m.P.nnz())
    );
    let t2 = Instant::now();
    let mut results: HashMap<i32, Vec<Option<(i32, Vec<i32>, f32)>>> = HashMap::new();
    for pmdp in models_ls.iter() {
        let mut pi = choose_random_policy(pmdp);
        let rowblock = pmdp.states.len() as i32;
        let pcolblock = rowblock as i32;
        let rcolblock = (model.num_agents + model.tasks.size) as i32;
        let initP = 
            argmaxM(pmdp.P.view(), &pi, rowblock, pcolblock, &pmdp.adjusted_state_act_pair);
        let initR = 
            argmaxM(pmdp.R.view(), &pi, rowblock, rcolblock, &pmdp.adjusted_state_act_pair);

        let mut r_v_init: Vec<f32> = vec![0.; initR.shape().0 as usize];
        let mut x_init: Vec<f32> = vec![0.; initP.shape().1 as usize];
        let mut y_init: Vec<f32> = vec![0.; initP.shape().0 as usize];
        let mut unstable: Vec<i32> = vec![0; initP.shape().0 as usize];
        let mut stable: Vec<f32> = vec![0.; x_init.len()];
        let mut y: Vec<f32> = vec![0.; pmdp.P.shape().0];
        let mut rmv: Vec<f32> = vec![0.; pmdp.P.shape().0];

        cuda_initial_policy_value_pinned_graph(
            initP.view(), 
            initR.view(), 
            pmdp.P.view(), 
            pmdp.R.view(), 
            &w_init,
            &w, 
            eps, 
            &mut x_init, 
            &mut y_init, 
            &mut r_v_init, 
            &mut y, 
            &mut rmv, 
            &mut unstable, 
            &mut pi, 
            &pmdp.enabled_actions, 
            &pmdp.adjusted_state_act_pair,
            &mut stable,
        );

        match results.get_mut(&pmdp.task_id) {
            Some(v) => { 
                v[pmdp.agent_id as usize] = Some((
                    pmdp.agent_id, 
                    pi.to_owned(), 
                    x_init[*pmdp.state_map.get(&pmdp.initial_state).unwrap()]
                ));
            }
            None => {
                results.insert(
                    pmdp.task_id,
                    (0..model.num_agents).map(|i| if i as i32 == pmdp.agent_id{
                        // insert the current tuple
                        Some((i as i32, 
                        pi.to_owned(),
                        x_init[*pmdp.state_map.get(&pmdp.initial_state).unwrap()]))
                    } else {
                        None
                    }).collect::<Vec<Option<(i32, Vec<i32>, f32)>>>()
                );
            }
        }   
    }
    println!("Time to do stage 1 {}", t2.elapsed().as_secs_f32());
    let allocation = allocation_fn(
        &results, model.tasks.size, model.num_agents
    );

    // Then for each allocation we need to make the argmax P, R matrices
    let allocatedArgMax: Vec<(CsMatBase<f32, i32, Vec<i32>, Vec<i32>, Vec<f32>>,
                              CsMatBase<f32, i32, Vec<i32>, Vec<i32>, Vec<f32>>)> 
        = allocation.into_par_iter().map(|(t, a, pi)| {
        let pmdp: &MOProductMDP<State> = models_ls.iter()
            .filter(|m| m.agent_id == a && m.task_id == t)
            .collect::<Vec<&MOProductMDP<State>>>()[0];
            let rowblock = pmdp.states.len() as i32;
            let pcolblock = rowblock as i32;
            let rcolblock = (model.num_agents + model.tasks.size) as i32;
            let argmaxP = 
                argmaxM(pmdp.P.view(), &pi, rowblock, pcolblock, &pmdp.adjusted_state_act_pair);
            let argmaxR = 
                argmaxM(pmdp.R.view(), &pi, rowblock, rcolblock, &pmdp.adjusted_state_act_pair);
            (argmaxP, argmaxR)
    }).collect();
    let nobjs = model.num_agents + model.tasks.size;
    let (P, R) = allocatedArgMax[0].to_owned();
    let mut storage = vec![0.; P.shape().0 * nobjs];
    cuda_multi_obj_solution(P.view(), R.view(), &mut storage, eps, nobjs as i32);
    println!("Total runtime {}", t2.elapsed().as_secs_f32());
}