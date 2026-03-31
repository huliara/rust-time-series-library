#![allow(dead_code)]

use ndarray::{s, Array1, Array2, Axis};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NgrcLoss {
    Mse,
    Mae,
}

#[derive(Debug, Clone)]
pub struct NgrcConfig {
    pub delay: usize,
    pub stride: usize,
    pub poly_order: usize,
    pub ridge_param: f32,
    pub transients: usize,
    pub bias: bool,
    pub loss: NgrcLoss,
}

impl Default for NgrcConfig {
    fn default() -> Self {
        Self {
            delay: 2,
            stride: 1,
            poly_order: 2,
            ridge_param: 1e-6,
            transients: 0,
            bias: true,
            loss: NgrcLoss::Mse,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Ngrc {
    pub config: NgrcConfig,
    pub wout: Option<Array2<f32>>,
}

impl Ngrc {
    pub fn new(config: NgrcConfig) -> Self {
        Self { config, wout: None }
    }

    pub fn fit(&mut self, train_data: &Array2<f32>) -> Result<(), String> {
        if train_data.nrows() < 2 {
            return Err("train_data must have at least 2 timesteps".to_string());
        }

        let x = train_data.slice(s![..-1, ..]).to_owned();
        let y_next = train_data.slice(s![1.., ..]).to_owned();
        let dtrain = &y_next - &x;

        let (lin_features, nlin_features, _) = self.nvar(&x, None)?;
        let wout = match self.config.loss {
            NgrcLoss::Mse => self.tikhonov_regression(&lin_features, &nlin_features, &dtrain)?,
            // LP-based MAE in Python requires an LP solver; here we fallback to ridge for portability.
            NgrcLoss::Mae => self.tikhonov_regression(&lin_features, &nlin_features, &dtrain)?,
        };
        self.wout = Some(wout);
        Ok(())
    }

    pub fn forecast(&self, context: &Array2<f32>, steps: usize) -> Result<Array2<f32>, String> {
        if context.nrows() < self.config.delay + 1 {
            return Err(format!(
                "context is too short: need at least {} rows",
                self.config.delay + 1
            ));
        }
        let n_dim = context.ncols();
        let mut output = Array2::<f32>::zeros((steps, n_dim));

        let warmup = context
            .slice(s![
                context.nrows() - self.config.delay - 1..context.nrows() - 1,
                ..
            ])
            .to_owned();
        let (_, _, mut window) = self.nvar(&warmup, None)?;

        let mut u = context
            .slice(s![context.nrows() - 1..context.nrows(), ..])
            .to_owned();

        for i in 0..steps {
            let (lin, nlin, new_window) = self.nvar(&u, Some(window))?;
            window = new_window;
            let delta = self.predict_from_features(&lin, &nlin)?;
            u = &u + &delta;
            output.slice_mut(s![i..i + 1, ..]).assign(&u);
        }

        Ok(output)
    }

    fn predict_from_features(
        &self,
        lin_features: &Array2<f32>,
        nlin_features: &Array2<f32>,
    ) -> Result<Array2<f32>, String> {
        let Some(wout) = &self.wout else {
            return Err("model is not fitted".to_string());
        };
        let mut tot = ndarray::concatenate![Axis(1), lin_features.view(), nlin_features.view()];
        if self.config.bias {
            let ones = Array2::<f32>::ones((tot.nrows(), 1));
            tot = ndarray::concatenate![Axis(1), ones.view(), tot.view()];
        }
        Ok(tot.dot(&wout.t()))
    }

    fn nvar(
        &self,
        x: &Array2<f32>,
        window: Option<Array2<f32>>,
    ) -> Result<(Array2<f32>, Array2<f32>, Array2<f32>), String> {
        let k = self.config.delay;
        let s_stride = self.config.stride;
        let p = self.config.poly_order;

        if k < 1 || s_stride < 1 || p < 1 {
            return Err("delay/stride/poly_order must be >= 1".to_string());
        }

        let n_steps = x.nrows();
        let n_dim = x.ncols();
        let lin_dim = n_dim * k;

        let monom_idx = combinations_with_replacement(lin_dim, p);
        let nlin_dim = monom_idx.len();

        let win_dim = (k - 1) * s_stride + 1;
        let mut win = match window {
            Some(w) => {
                if w.shape() != [win_dim, n_dim] {
                    return Err(format!(
                        "window shape mismatch: expected ({}, {}), got ({}, {})",
                        win_dim,
                        n_dim,
                        w.nrows(),
                        w.ncols()
                    ));
                }
                w
            }
            None => Array2::<f32>::zeros((win_dim, n_dim)),
        };

        let mut lin_features = Array2::<f32>::zeros((n_steps, lin_dim));
        let mut nlin_features = Array2::<f32>::zeros((n_steps, nlin_dim));

        for i in 0..n_steps {
            for row in 0..(win_dim - 1) {
                let next = win.slice(s![row + 1, ..]).to_owned();
                win.slice_mut(s![row, ..]).assign(&next);
            }
            win.slice_mut(s![win_dim - 1, ..])
                .assign(&x.slice(s![i, ..]).to_owned());

            let mut lin_feat = Array1::<f32>::zeros(lin_dim);
            let mut idx = 0usize;
            for row in (0..win_dim).step_by(s_stride) {
                for col in 0..n_dim {
                    lin_feat[idx] = win[[row, col]];
                    idx += 1;
                }
            }

            let mut nlin_feat = Array1::<f32>::zeros(nlin_dim);
            for (j, comb) in monom_idx.iter().enumerate() {
                let mut prod = 1.0f32;
                for &c in comb {
                    prod *= lin_feat[c];
                }
                nlin_feat[j] = prod;
            }

            lin_features.slice_mut(s![i, ..]).assign(&lin_feat);
            nlin_features.slice_mut(s![i, ..]).assign(&nlin_feat);
        }

        Ok((lin_features, nlin_features, win))
    }

    fn total_feature(
        &self,
        lin_features: &Array2<f32>,
        nlin_features: &Array2<f32>,
    ) -> Array2<f32> {
        let mut tot = ndarray::concatenate![Axis(1), lin_features.view(), nlin_features.view()];
        let trans = self.config.transients.min(tot.nrows());
        tot = tot.slice(s![trans.., ..]).to_owned();
        if self.config.bias {
            let ones = Array2::<f32>::ones((tot.nrows(), 1));
            tot = ndarray::concatenate![Axis(1), ones.view(), tot.view()];
        }
        tot
    }

    fn tikhonov_regression(
        &self,
        lin_features: &Array2<f32>,
        nlin_features: &Array2<f32>,
        target: &Array2<f32>,
    ) -> Result<Array2<f32>, String> {
        let trans = self.config.transients.min(target.nrows());
        let y = target.slice(s![trans.., ..]).to_owned();
        let x = self.total_feature(lin_features, nlin_features);

        if x.nrows() == 0 || y.nrows() == 0 {
            return Err("no rows left after applying transients".to_string());
        }

        let yxt = y.t().dot(&x);
        let mut xxt = x.t().dot(&x);
        for i in 0..xxt.nrows().min(xxt.ncols()) {
            xxt[[i, i]] += self.config.ridge_param;
        }
        let inv = invert_square(&xxt)?;
        Ok(yxt.dot(&inv))
    }
}

fn combinations_with_replacement(n: usize, p: usize) -> Vec<Vec<usize>> {
    fn rec(start: usize, n: usize, p: usize, cur: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if cur.len() == p {
            out.push(cur.clone());
            return;
        }
        for i in start..n {
            cur.push(i);
            rec(i, n, p, cur, out);
            cur.pop();
        }
    }

    let mut out = Vec::new();
    rec(0, n, p, &mut Vec::new(), &mut out);
    out
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
    use ndarray::Array2;

    #[test]
    fn ngrc_fit_and_forecast_shape() {
        let mut model = Ngrc::new(NgrcConfig {
            delay: 2,
            stride: 1,
            poly_order: 2,
            ridge_param: 1e-4,
            transients: 5,
            bias: true,
            loss: NgrcLoss::Mse,
        });

        let n = 120;
        let mut data = Array2::<f32>::zeros((n, 1));
        for i in 0..n {
            data[[i, 0]] = (i as f32 * 0.05).sin();
        }

        model.fit(&data).unwrap();
        let context = data.slice(s![n - 30.., ..]).to_owned();
        let pred = model.forecast(&context, 10).unwrap();
        assert_eq!(pred.shape(), &[10, 1]);
    }
}
