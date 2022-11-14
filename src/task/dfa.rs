use std::collections::{HashMap, HashSet};
use pyo3::prelude::*;
//use serde::{Serialize, Deserialize};
//use serde_json;

#[pyclass]
#[derive(Clone)] 
pub struct DFA {
    #[pyo3(get)]
    pub states: Vec<i32>,
    pub name: String,
    #[pyo3(get)]
    pub initial_state: i32,
    #[pyo3(get)]
    pub accepting: Vec<i32>,
    #[pyo3(get)]
    pub rejecting: Vec<i32>,
    #[pyo3(get)]
    pub done: Vec<i32>,
    pub transitions: HashMap<String, i32>,
    pub current_state: i32,
    words: HashSet<String>
}

/// Supply a convenience trait so that we can simply or create 
/// words in the environment
pub trait Expression<D> {
    fn conversion(&self, q: i32, data: &D) -> String;
}

#[pymethods]
impl DFA {
    #[new]
    pub fn new(
        states: Vec<i32>, 
        initial_state: i32, 
        accepting: Vec<i32>, 
        rejecting: Vec<i32>, 
        done: Vec<i32>
    ) -> Self {
        DFA {
            states, 
            name: "task".to_string(),
            initial_state, 
            accepting, 
            rejecting, 
            done, 
            transitions: HashMap::new(),
            current_state: 0,
            words: HashSet::new()
        }
    }

    fn print_transitions(&self) {
        for transition in self.transitions.iter() {
            println!("{:?}", transition);
        }
    }

    fn add_transition(&mut self, q: i32, w: String, qprime: i32) {
        self.transitions.insert(format!("{}-{}", q, w), qprime);
    }

    fn define_words(&mut self, words: HashSet<String>) {
        for word in words.iter() {
            self.words.insert(word.to_string());
        }
    }

    pub fn get_transition(&self, state: i32, word: &str) -> i32 {
        let qprime = *self.transitions.get(&format!("{}-{}", state, word))
            .expect(&format!("Could not find transition: {}-{}", state, word));
        qprime
    }

    pub fn next(&mut self, state: i32, word: String) -> i32 {
        let qprime = *self.transitions.get(&format!("{}-{}", state, word)).unwrap();
        //println!("state: {}, word: {:?} -> q' {}", state, word, qprime);
        self.current_state = qprime;
        self.current_state
    }

    pub fn check_done(&self) -> u8 {
        if self.done.contains(&self.current_state) {
            2
        } else if self.rejecting.contains(&self.current_state) {
            3
        } else if self.current_state == self.initial_state {
            0
        } else {
            1
        }
    }

    fn reset(&mut self) {
        self.current_state = self.initial_state;
    }

    fn clone(&self) -> Self {
        DFA {
            states: self.states.to_vec(), 
            name: self.name.to_string(),
            initial_state: self.initial_state, 
            accepting: self.accepting.to_vec(), 
            rejecting: self.rejecting.to_vec(), 
            done: self.done.to_vec(), 
            transitions: self.transitions.clone(),
            current_state: 0,
            words: self.words.clone()
        }
    }

    /*fn json_serialize_dfa(&self) -> String {
        serde_json::to_string(&self).unwrap()
    }*/
}

/*#[pyfunction]
#[pyo3(name="task_from_str")]
pub fn json_deserialize_from_string(s: String) -> DFA {
    let task: DFA = serde_json::from_str(&s).unwrap();
    task
}*/

#[pyclass]
#[derive(Clone)]
pub struct Mission {
    pub tasks: Vec<DFA>,
    #[pyo3(get)]
    pub size: usize
}

#[pymethods]
impl Mission {
    #[new]
    fn new() -> Self {
        Mission {
            tasks: Vec::new(),
            size: 0
        }
    }

    fn add_task(&mut self, dfa: DFA) {
        self.tasks.push(dfa);
        self.size += 1;
    }

    fn print_task_transitions(&self) {
        for task in self.tasks.iter() {
            println!("|P|: {:?}", task.transitions.len());
        }
    }

    pub fn get_task(&self, task_idx: usize) -> DFA {
        self.tasks[task_idx].clone()
    }

    pub fn step(&mut self, task: usize, q: i32, word: String) -> i32 {
        self.tasks[task].next(q, word)
    }

    pub fn reset(&mut self, task: usize) {
        self.tasks[task].reset()
    }

    pub fn check_done(&self, task: usize) -> u8 {
        self.tasks[task].check_done()
    }

    pub fn check_mission_complete(&self) -> bool {
        let mut complete: bool = true;
        for task in self.tasks.iter() {
            if task.accepting.contains(&task.current_state) 
                || task.rejecting.contains(&task.current_state) {
                // do nothing 
            } else {
                complete = false;
            }
        }
        return complete
    }

    pub fn release_last_dfa(&mut self) -> Option<DFA> {
        self.tasks.pop()
    }

    pub fn clear_mission(&mut self) {
        self.tasks = Vec::new();
        self.size = 0;
    }
}