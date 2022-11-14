use std::{sync::mpsc, thread};
use crate::model::{scpm::SCPM, momdp::product_mdp_bfs};
use std::hash::Hash;
use crate::agent::env::Env;

pub fn start_threads_compute_model<S, E>(
    model: &SCPM,
    env: E
) where S: Copy + Eq + Hash + Sync + Send + std::fmt::Debug, 
    E: Env<S> + Clone + Sync + Send + 'static {
    let (tx, rx) = mpsc::channel();

    // spawn a single thread and build a model
    let m = model.clone();
    thread::spawn(move || { 
        // build a model and send a 1 to the mpsc channel
        product_mdp_bfs(
            (env.get_init_state(0), 0), 
            env, 
            m.tasks.get_task(0), 
            0, 
            0, 
            m.num_agents, 
            m.num_agents + m.tasks.size, 
            &m.actions
        );
        // Need to be able to compute the value of the product MDP here to see 
        // what is really going on with the threading
        tx.send(1).unwrap(); // TODO probably use ? and return a result from this function
    });
    let received = rx.recv().unwrap();
    println!("Finished building the product MDP: {}", received);
}