use super::_kuramoto_sivashinsky_etdrk4::_kuramoto_sivashinsky_etdrk4;

pub fn _kuramoto_sivashinsky(
    n_timesteps: usize,
    warmup: usize,
    n: usize,
    m: f64,
    x0: Vec<f64>,
    h: f64,
) -> Result<Vec<Vec<f64>>, String> {
    if x0.len() != n {
        return Err(format!("Initial condition x0 should have shape ({n},)."));
    }
    if n_timesteps == 0 {
        return Ok(Vec::new());
    }

    // Real-space surrogate coefficients for ETDRK4-style stepping.
    let mut k = vec![0.0_f64; n];
    for (i, ki) in k.iter_mut().enumerate() {
        let base = if i <= n / 2 { i as f64 } else { (i as f64) - (n as f64) };
        *ki = base / m;
    }
    let l = k
        .iter()
        .map(|ki| ki * ki - ki.powi(4))
        .collect::<Vec<_>>();
    let e = l.iter().map(|li| (h * li).exp()).collect::<Vec<_>>();
    let e2 = l.iter().map(|li| (0.5 * h * li).exp()).collect::<Vec<_>>();
    let q = vec![h; n];
    let f1 = vec![h; n];
    let f2 = vec![0.5 * h; n];
    let f3 = vec![0.5 * h; n];
    let g = k.iter().map(|ki| -0.5 * ki).collect::<Vec<_>>();

    let mut states = Vec::with_capacity(n_timesteps);
    let mut v = x0;
    states.push(v.clone());

    for _ in 1..n_timesteps {
        v = _kuramoto_sivashinsky_etdrk4(&v, &g, &e, &e2, &q, &f1, &f2, &f3);
        states.push(v.clone());
    }

    Ok(states.into_iter().skip(warmup).collect::<Vec<_>>())
}
