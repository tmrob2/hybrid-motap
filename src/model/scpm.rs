use crate::task::dfa::Mission;
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone)]
/// SCPM is an unusual object
/// 
/// For now we will just include the team of agents and a mission
/// because we just want to look out how the channel works
pub struct SCPM {
    pub num_agents: usize,
    //pub agents: Team,
    pub tasks: Mission, 
    // todo: I don't think that we actually require a mission any more, because we will not be storing any data here
    // all of the transition information should be stored in the mdp environment script which will be called via GIL 
    pub actions: Vec<i32>
}

#[pymethods]
impl SCPM{ 
    #[new]
    fn new(mission: Mission, num_agents: usize, actions: Vec<i32>) -> Self {
        //let num_tasks = mission.size;
        SCPM {
            num_agents,
            tasks: mission,
            actions
        }
    }
}