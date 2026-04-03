#![allow(dead_code)]

use ndarray::{s, Array1, Array2};
use rand::{rngs::StdRng, Rng, SeedableRng};

#[derive(Debug, Clone)]
pub struct RcConfig {
    pub units: usize,
    pub leak_rate: f32,
    pub spectral_radius: f32,
    pub ridge: f32,
    pub input_scale: f32,
    pub seed: u64,
}

impl Default for RcConfig {
    fn default() -> Self {
        Self {
            units: 200,
            leak_rate: 0.3,
            spectral_radius: 0.9,
            ridge: 1e-6,
            input_scale: 1.0,
            seed: 42,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Rc {
    pub config: RcConfig,
    pub w_in: Option<Array2<f32>>,  // [units, in_dim]
    pub w_res: Option<Array2<f32>>, // [units, units]
    pub w_out: Option<Array2<f32>>, // [out_dim, 1 + units]
    pub state: Option<Array1<f32>>, // [units]
}

impl Rc {
    pub fn new(config: RcConfig) -> Self {
        Self {
            config,
            w_in: None,
            w_res: None,
            w_out: None,
            state: None,
        }
    }

    pub fn fit(&mut self, train_data: &Array2<f32>, warmup: usize) -> Result<(), String> {
        if train_data.nrows() < 2 {
            return Err("train_data must have at least 2 timesteps".to_string());
        }

        let x_train = train_data.slice(s![..-1, ..]).to_owned();
        let y_train = train_data.slice(s![1.., ..]).to_owned();
        let in_dim = x_train.ncols();
        self.ensure_weights(in_dim)?;
        self.reset_state();

        let mut states: Vec<Array1<f32>> = Vec::with_capacity(x_train.nrows());
        for t in 0..x_train.nrows() {
            let u = x_train.slice(s![t, ..]).to_owned();
            let s_t = self.step_state(&u)?;
            states.push(s_t);
        }

        let start = warmup.min(states.len().saturating_sub(1));
        let n_rows = states.len().saturating_sub(start);
        if n_rows == 0 {
            return Err("no rows left after warmup".to_string());
        }

        let mut x_feat = Array2::<f32>::zeros((n_rows, self.config.units + 1));
        for (i, st) in states.iter().skip(start).enumerate() {
            x_feat[[i, 0]] = 1.0;
            for j in 0..self.config.units {
                x_feat[[i, 1 + j]] = st[j];
            }
        }
        let y = y_train.slice(s![start.., ..]).to_owned();

        let yxt = y.t().dot(&x_feat);
        let mut xxt = x_feat.t().dot(&x_feat);
        for i in 0..xxt.nrows().min(xxt.ncols()) {
            xxt[[i, i]] += self.config.ridge;
        }
        let inv = invert_square(&xxt)?;
        self.w_out = Some(yxt.dot(&inv));

        Ok(())
    }

    pub fn forecast(&mut self, context: &Array2<f32>, steps: usize) -> Result<Array2<f32>, String> {
        if context.nrows() == 0 {
            return Err("context must not be empty".to_string());
        }

        let in_dim = context.ncols();
        self.ensure_weights(in_dim)?;
        self.reset_state();

        for t in 0..context.nrows() {
            let u = context.slice(s![t, ..]).to_owned();
            self.step_state(&u)?;
        }

        let mut x = context.slice(s![context.nrows() - 1, ..]).to_owned();
        let out_dim = x.len();
        let mut pred = Array2::<f32>::zeros((steps, out_dim));

        for i in 0..steps {
            self.step_state(&x)?;
            let y = self.readout()?;
            for d in 0..out_dim {
                pred[[i, d]] = y[d];
            }
            x = y;
        }

        Ok(pred)
    }

    fn ensure_weights(&mut self, in_dim: usize) -> Result<(), String> {
        if self.w_in.is_some() && self.w_res.is_some() {
            return Ok(());
        }

        let mut rng = StdRng::seed_from_u64(self.config.seed);
        let units = self.config.units;

        let mut w_in = Array2::<f32>::zeros((units, in_dim));
        for i in 0..units {
            for j in 0..in_dim {
                w_in[[i, j]] = (rng.gen::<f32>() * 2.0 - 1.0) * self.config.input_scale;
            }
        }

        let mut w_res = Array2::<f32>::zeros((units, units));
        for i in 0..units {
            for j in 0..units {
                w_res[[i, j]] = rng.gen::<f32>() * 2.0 - 1.0;
            }
        }

        let radius = spectral_radius_power_iter(&w_res, 50).max(1e-6);
        let scale = self.config.spectral_radius / radius;
        for i in 0..units {
            for j in 0..units {
                w_res[[i, j]] *= scale;
            }
        }

        self.w_in = Some(w_in);
        self.w_res = Some(w_res);
        self.state = Some(Array1::<f32>::zeros(units));
        Ok(())
    }

    fn reset_state(&mut self) {
        if let Some(state) = &mut self.state {
            state.fill(0.0);
        }
    }

    fn step_state(&mut self, u: &Array1<f32>) -> Result<Array1<f32>, String> {
        let Some(w_in) = &self.w_in else {
            return Err("w_in is not initialized".to_string());
        };
        let Some(w_res) = &self.w_res else {
            return Err("w_res is not initialized".to_string());
        };
        let Some(prev_state) = &self.state else {
            return Err("state is not initialized".to_string());
        };

        let mut pre = w_in.dot(u) + w_res.dot(prev_state);
        pre.mapv_inplace(|v| v.tanh());
        let new_state = prev_state * (1.0 - self.config.leak_rate) + pre * self.config.leak_rate;
        self.state = Some(new_state.clone());
        Ok(new_state)
    }

    fn readout(&self) -> Result<Array1<f32>, String> {
        let Some(w_out) = &self.w_out else {
            return Err("model is not fitted".to_string());
        };
        let Some(state) = &self.state else {
            return Err("state is not initialized".to_string());
        };

        let mut feat = Array1::<f32>::zeros(state.len() + 1);
        feat[0] = 1.0;
        for i in 0..state.len() {
            feat[i + 1] = state[i];
        }
        Ok(w_out.dot(&feat))
    }
}

fn spectral_radius_power_iter(a: &Array2<f32>, iters: usize) -> f32 {
    let n = a.nrows();
    let mut v = Array1::<f32>::ones(n);
    for _ in 0..iters {
        let w = a.dot(&v);
        let norm = (w.dot(&w)).sqrt().max(1e-12);
        v = w.mapv(|x| x / norm);
    }
    let w = a.dot(&v);
    w.dot(&v).abs()
}

fn invert_square(a: &Array2<f32>) -> Result<Array2<f32>, String> {
    if a.nrows() != a.ncols() {
        return Err("matrix must be square".to_string());
    }
    let n = a.nrows();
    let mut aug = Array2::<f32>::zeros((n, 2 * n));
    for r in 0..n {
        for c in 0..n {
            aug[[r, c]] = a[[r, c]];
        }
        aug[[r, n + r]] = 1.0;
    }

    for i in 0..n {
        let mut pivot = i;
        let mut max_abs = aug[[i, i]].abs();
        for r in (i + 1)..n {
            let v = aug[[r, i]].abs();
            if v > max_abs {
                max_abs = v;
                pivot = r;
            }
        }
        if max_abs < 1e-12 {
            return Err("matrix is singular".to_string());
        }
        if pivot != i {
            for c in 0..(2 * n) {
                let tmp = aug[[i, c]];
                aug[[i, c]] = aug[[pivot, c]];
                aug[[pivot, c]] = tmp;
            }
        }

        let diag = aug[[i, i]];
        for c in 0..(2 * n) {
            aug[[i, c]] /= diag;
        }

        for r in 0..n {
            if r == i {
                continue;
            }
            let factor = aug[[r, i]];
            if factor.abs() < 1e-20 {
                continue;
            }
            for c in 0..(2 * n) {
                aug[[r, c]] -= factor * aug[[i, c]];
            }
        }
    }

    let mut inv = Array2::<f32>::zeros((n, n));
    for r in 0..n {
        for c in 0..n {
            inv[[r, c]] = aug[[r, n + c]];
        }
    }
    Ok(inv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rc_fit_and_forecast_shape() {
        let mut model = Rc::new(RcConfig {
            units: 64,
            leak_rate: 0.3,
            spectral_radius: 0.9,
            ridge: 1e-4,
            input_scale: 0.8,
            seed: 42,
        });

        let n = 200;
        let mut data = Array2::<f32>::zeros((n, 1));
        for i in 0..n {
            data[[i, 0]] = (i as f32 * 0.04).sin();
        }

        model.fit(&data, 50).unwrap();
        let context = data.slice(s![n - 50.., ..]).to_owned();
        let pred = model.forecast(&context, 20).unwrap();
        assert_eq!(pred.shape(), &[20, 1]);
    }
}
