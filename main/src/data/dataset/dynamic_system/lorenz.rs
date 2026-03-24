fn lorenz_diff(state: [f64; 3], rho: f64, sigma: f64, beta: f64) -> [f64; 3] {
    let x = state[0];
    let y = state[1];
    let z = state[2];
    [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]
}

fn rk4_step(state: [f64; 3], dt: f64, rho: f64, sigma: f64, beta: f64) -> [f64; 3] {
    let k1 = lorenz_diff(state, rho, sigma, beta);
    let s2 = [
        state[0] + 0.5 * dt * k1[0],
        state[1] + 0.5 * dt * k1[1],
        state[2] + 0.5 * dt * k1[2],
    ];
    let k2 = lorenz_diff(s2, rho, sigma, beta);
    let s3 = [
        state[0] + 0.5 * dt * k2[0],
        state[1] + 0.5 * dt * k2[1],
        state[2] + 0.5 * dt * k2[2],
    ];
    let k3 = lorenz_diff(s3, rho, sigma, beta);
    let s4 = [
        state[0] + dt * k3[0],
        state[1] + dt * k3[1],
        state[2] + dt * k3[2],
    ];
    let k4 = lorenz_diff(s4, rho, sigma, beta);

    [
        state[0] + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
        state[1] + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
        state[2] + dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
    ]
}

pub fn lorenz(
    n_timesteps: usize,
    rho: f64,
    sigma: f64,
    beta: f64,
    x0: [f64; 3],
    h: f64,
) -> Vec<[f64; 3]> {
    if n_timesteps == 0 {
        return Vec::new();
    }

    let mut states = Vec::with_capacity(n_timesteps);
    let mut state = x0;
    states.push(state);

    for _ in 1..n_timesteps {
        state = rk4_step(state, h, rho, sigma, beta);
        states.push(state);
    }

    states
}
