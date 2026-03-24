pub fn logistic_map(n_timesteps: usize, r: f64, x0: f64) -> Result<Vec<[f64; 1]>, String> {
    if r <= 0.0 {
        return Err("r should be positive.".to_string());
    }
    if !(0.0 < x0 && x0 < 1.0) {
        return Err("Initial condition x0 should be in ]0;1[.".to_string());
    }
    if n_timesteps == 0 {
        return Ok(Vec::new());
    }

    let mut x = vec![[0.0_f64; 1]; n_timesteps];
    x[0][0] = x0;

    for i in 1..n_timesteps {
        x[i][0] = r * x[i - 1][0] * (1.0 - x[i - 1][0]);
    }

    Ok(x)
}
