pub fn henon_map(n_timesteps: usize, a: f64, b: f64, x0: [f64; 2]) -> Vec<[f64; 2]> {
    if n_timesteps == 0 {
        return Vec::new();
    }

    let mut states = vec![[0.0, 0.0]; n_timesteps];
    states[0] = x0;

    for i in 1..n_timesteps {
        states[i][0] = 1.0 - a * states[i - 1][0] * states[i - 1][0] + states[i - 1][1];
        states[i][1] = b * states[i - 1][0];
    }

    states
}
