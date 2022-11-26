pub trait Env<S> {
    fn step_(&self, s: S, action: u8, task_id: i32) -> Result<Vec<(S, f32, String)>, String>;

    fn get_init_state(&self, agent: usize) -> S;

    fn set_task(&mut self, task_id: usize);

    fn get_action_space(&self) -> Vec<i32>;

    fn get_states_len(&self) -> usize;
}