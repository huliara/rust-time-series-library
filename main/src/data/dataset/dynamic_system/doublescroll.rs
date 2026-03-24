fn doublescroll_diff(state: [f64; 3], r1: f64, r2: f64, r4: f64, ir: f64, beta: f64) -> [f64; 3] {
    let v1 = state[0];
    let v2 = state[1];
    let i = state[2];

    let dv = v1 - v2;
    let factor = (dv / r2) + ir * (beta * dv).sinh();
    let dv1 = (v1 / r1) - factor;
    let dv2 = factor - i;
    let di = v2 - r4 * i;

    [dv1, dv2, di]
}

fn rk4_step(state: [f64; 3], dt: f64, r1: f64, r2: f64, r4: f64, ir: f64, beta: f64) -> [f64; 3] {
    let k1 = doublescroll_diff(state, r1, r2, r4, ir, beta);
    let s2 = [
        state[0] + 0.5 * dt * k1[0],
        state[1] + 0.5 * dt * k1[1],
        state[2] + 0.5 * dt * k1[2],
    ];
    let k2 = doublescroll_diff(s2, r1, r2, r4, ir, beta);
    let s3 = [
        state[0] + 0.5 * dt * k2[0],
        state[1] + 0.5 * dt * k2[1],
        state[2] + 0.5 * dt * k2[2],
    ];
    let k3 = doublescroll_diff(s3, r1, r2, r4, ir, beta);
    let s4 = [
        state[0] + dt * k3[0],
        state[1] + dt * k3[1],
        state[2] + dt * k3[2],
    ];
    let k4 = doublescroll_diff(s4, r1, r2, r4, ir, beta);

    [
        state[0] + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
        state[1] + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
        state[2] + dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
    ]
}

pub fn doublescroll(
    n_timesteps: usize,
    r1: f64,
    r2: f64,
    r4: f64,
    ir: f64,
    beta: f64,
    x0: [f64; 3],
    h: f64,
) -> Vec<[f64; 3]> {
    if n_timesteps == 0 {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(n_timesteps);
    let mut state = x0;
    out.push(state);

    for _ in 1..n_timesteps {
        state = rk4_step(state, h, r1, r2, r4, ir, beta);
        out.push(state);
    }

    out
}
