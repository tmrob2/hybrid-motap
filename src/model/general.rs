pub trait ModelFns<S> {
    fn get_states(&self) -> &[S];

    fn get_enabled_actions(&self) -> &[i32];
}