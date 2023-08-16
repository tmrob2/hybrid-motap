use pyo3::prelude::*;
use crate::agent::env::Env;
use crate::model::mostapu::MOSTAPU_bfs;
use crate::solvers::motap::{stapu_solver};
use std::time::Instant;
use crate::model::mostapu::StapuState;
use crate::{product, debug_level, motap_prism_file_generator, generic_motap_mostapu_cpu};
use crate::model::momdp::product_mdp_bfs;
use crate::model::scpm::{SCPM, SCPMState, SCPM_bfs};
use crate::model::momdp::MOProductMDP;
use crate::Debug;
use hashbrown::HashMap;

#[pyclass]
#[derive(Clone)]
pub struct DSTModel {
    pub states: Vec<i32>,
    pub initial_state: i32,
    pub action_space: Vec<i32>,
    treasure_coords: HashMap<(i32, i32), String>
}

#[pyclass]
pub struct DSTMAS {
    pub agents: Vec<DSTModel>,
    pub size: usize, 
    pub ordering: Vec<usize>,
    pub action_space: Vec<i32>
}

#[pymethods]
impl DSTMAS {
    #[new]
    pub fn new(action_space: Vec<i32>) -> Self {
        DSTMAS {
            agents: Vec::new(),
            size: 0,
            ordering: Vec::new(),
            action_space
        }
    }

    pub fn create_order(&mut self, order: Vec<usize>) {
        self.ordering = order;
    }

    pub fn add_environment(&mut self, env: DSTModel) {
        self.agents.push(env);
    }

    pub fn get_init_states(&self) -> Vec<i32> {
        let mut init_states: Vec<i32> = Vec::new();
        for agent in self.agents.iter() {
            init_states.push(agent.initial_state);
        }
        init_states
    }
}

#[pymethods]
impl DSTModel {
    #[new]
    pub fn new(init_row: i32, init_col: i32) -> Self {
        let t_ = [((1, 0), "0_7"), ((2, 1), "8_2"), ((3, 2), "11_5"), 
                  ((4, 3), "14_0"), ((4, 4), "15_1"), ((4, 5), "16_1"),
                  ((7, 6), "19_6"), ((7, 7), "20_3"), ((9, 8), "22_4"),
                  ((10, 9), "23_7")];
        let m: HashMap<_, _> = t_.into_iter()
            .map(|(k, v)| (k, v.to_string()))
            .collect();
        DSTModel {
            states: (0..11 * 11).collect(),
            initial_state: 11 * init_row + init_col,
            action_space: (0..4).collect(),
            treasure_coords: m
        }
    }
}

impl Env<i32> for DSTMAS {
    fn step_(&self, s: i32, action: u8, _task_id: i32, agent_id: i32) 
    -> Result<Vec<(i32, f32, String)>, String> {
        // get the coordinates of the state index
        let mut row: i32 = s  / 11;
        let mut col: i32 = s % 11;
        
        match action {
            0 => {
                row = if row > 0 { row - 1} else { 0 };
            }
            1 => {
                row = if row < 11 - 1 { row + 1} else { row };
            }
            2 => { 
                col = if col > 0 { col - 1 } else { 0 };
            }
            3 => {
                col = if col < 11 - 1 { col + 1} else { col };
            } 
            _ => {
            }
        };
        let sprime = 11 * row + col;
        let w = match self.agents[agent_id as usize].treasure_coords.get(&(row, col)) {
            Some(x) => { x.to_string() }
            None => {
                match col {
                    0 => { if row > 1 { "neg".to_string()} else {"".to_string() } }
                    1 => { if row > 2 { "neg".to_string()} else {"".to_string() }}
                    2 => { if row > 3 { "neg".to_string()} else {"".to_string() }}
                    3 => { if row > 4 { "neg".to_string()} else {"".to_string() }}
                    4 => { if row > 4 { "neg".to_string()} else {"".to_string() }}
                    5 => { if row > 4 { "neg".to_string()} else {"".to_string() }}
                    6 => { if row > 7 { "neg".to_string()} else {"".to_string() }}
                    7 => { if row > 7 { "neg".to_string()} else {"".to_string() }}
                    8 => { if row > 9 { "neg".to_string()} else {"".to_string() }}
                    9 => { "".to_string() }
                    10 => { "".to_string() }
                    _ => { "".to_string() }
                }
            }
        };
        Ok(vec![(sprime, 1.0, w.to_string())])
    }

    fn get_init_state(&self, _agent: usize) -> i32 {
        0
    }

    fn set_task(&mut self, _task_id: usize) {
        todo!()
    }

    fn get_action_space(&self) -> Vec<i32> {
        self.action_space.to_vec()
    }

    fn get_states_len(&self, agent_id: i32) -> usize {
        self.agents[self.ordering[agent_id as usize]].states.len()
    }
} 

#[pyfunction]
pub fn dst_stapu_model(
    model: &SCPM,
    env: &DSTMAS,
    w: Vec<f32>,
    epsilon1: f32,
    debug: i32,
    max_iter: usize,
    max_unstable: i32,
    initial_agent_states: Vec<i32>
) { 
    generic_motap_mostapu_cpu(model, env, w, epsilon1, debug, max_iter, max_unstable, &initial_agent_states[..]);
}

#[pyfunction]
pub fn dst_build_test(
    model: &SCPM,
    env: &DSTMAS,
) -> () 
where DSTMAS: Env<i32> {
    let initial_state = StapuState {
        s: env.get_init_state(0),
        Q: (0..model.tasks.size).map(
            |t| model.tasks.get_task(t).initial_state
            ).collect(),
        agentid: 0,
        active_tasks: None,
        remaining: (0..model.tasks.size as i32).collect()
    };
    let alloc_acts: Vec<i32> = (0..model.tasks.size as i32 + 1).collect();
    let agent_acts: Vec<i32> = model.actions.iter()
        .map(|a| a + model.tasks.size as i32 + 1)
        .collect();
    let actions = [&alloc_acts[..], &agent_acts[..]].concat();
    let _scpm = MOSTAPU_bfs(
        initial_state, 
        env, 
        model.num_agents, 
        model.tasks.size, 
        model.tasks.size + model.num_agents, 
        model, 
        &actions,
        &env.get_init_states()[..]
    );
    //println!("state space: {}", _scpm.states.len())

    //println!("R\n:{:?}", _scpm.R.to_dense());
    //println!("P\n:{:?}", _scpm.P.to_dense());
}