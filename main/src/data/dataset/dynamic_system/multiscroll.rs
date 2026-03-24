fn multiscroll_diff(state: [f64; 3], a: f64, b: f64, c: f64) -> [f64; 3] {
    let x = state[0];
    let y = state[1];
    let z = state[2];
    [a * (y - x), (c - a) * x - x * z + c * y, x * y - b * z]
}

fn rk4_step(state: [f64; 3], dt: f64, a: f64, b: f64, c: f64) -> [f64; 3] {
    let k1 = multiscroll_diff(state, a, b, c);
    let s2 = [
        state[0] + 0.5 * dt * k1[0],
        state[1] + 0.5 * dt * k1[1],
        state[2] + 0.5 * dt * k1[2],
    ];
    let k2 = multiscroll_diff(s2, a, b, c);
    let s3 = [
        state[0] + 0.5 * dt * k2[0],
        state[1] + 0.5 * dt * k2[1],
        state[2] + 0.5 * dt * k2[2],
    ];
    let k3 = multiscroll_diff(s3, a, b, c);
    let s4 = [
        state[0] + dt * k3[0],
        state[1] + dt * k3[1],
        state[2] + dt * k3[2],
    ];
    let k4 = multiscroll_diff(s4, a, b, c);

    [
        state[0] + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
        state[1] + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
        state[2] + dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
    ]
}

pub fn multiscroll(n_timesteps: usize, a: f64, b: f64, c: f64, x0: [f64; 3], h: f64) -> Vec<[f64; 3]> {
    if n_timesteps == 0 {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(n_timesteps);
    let mut state = x0;
    out.push(state);

    for _ in 1..n_timesteps {
        state = rk4_step(state, h, a, b, c);
        out.push(state);
    }

    out
}
