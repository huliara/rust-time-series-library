use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use clap::{Args, ValueEnum};
use ndarray::{s, Array2};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, ValueEnum, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum NgrcLoss {
    Mse,
    Mae,
}

#[derive(Debug, Clone, Deserialize, Serialize, Args)]
pub struct NgrcArgs {
    pub delay: usize,
    pub stride: usize,
    pub poly_order: usize,
    pub ridge_param: f32,
    pub transients: usize,
    pub bias: bool,
    pub loss: NgrcLoss,
}

impl Default for NgrcArgs {
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
pub struct Ngrc<B: Backend> {
    pub config: NgrcArgs,
    pub wout: Option<Tensor<B, 2>>,
    pub device: B::Device,
}

impl<B: Backend> Ngrc<B> {
    pub fn new(config: NgrcArgs, device: B::Device) -> Self {
        Self {
            config,
            wout: None,
            device,
        }
    }

    pub fn fit(&mut self, train_data: &Tensor<B, 2>) -> Result<(), String> {
        let shape = train_data.shape();
        if shape.dims[0] < 2 {
            return Err("train_data must have at least 2 timesteps".to_string());
        }

        let x = train_data
            .clone()
            .slice([0..shape.dims[0] - 1, 0..shape.dims[1]]);
        let y_next = train_data
            .clone()
            .slice([1..shape.dims[0], 0..shape.dims[1]]);
        let dtrain = y_next - x.clone();

        let (lin_features, nlin_features, _) = self.nvar(&x, None)?;
        let wout = match self.config.loss {
            NgrcLoss::Mse => self.tikhonov_regression(&lin_features, &nlin_features, &dtrain)?,
            NgrcLoss::Mae => self.tikhonov_regression(&lin_features, &nlin_features, &dtrain)?,
        };
        self.wout = Some(wout);
        Ok(())
    }

    pub fn forecast(&self, context: &Tensor<B, 2>, steps: usize) -> Result<Tensor<B, 2>, String> {
        let shape = context.shape();
        if shape.dims[0] < self.config.delay + 1 {
            return Err(format!(
                "context is too short: need at least {} rows",
                self.config.delay + 1
            ));
        }
        let n_dim = shape.dims[1];

        let context_start = shape.dims[0].saturating_sub(self.config.delay);
        let warmup = context
            .clone()
            .slice([context_start..shape.dims[0] - 1, 0..n_dim]);
        let (_, _, mut window) = self.nvar(&warmup, None)?;

        let mut u = context
            .clone()
            .slice([shape.dims[0] - 1..shape.dims[0], 0..n_dim]);

        let mut outputs = Vec::new();

        for _ in 0..steps {
            let (lin, nlin, new_window) = self.nvar(&u, Some(window.clone()))?;
            window = new_window;
            let delta = self.predict_from_features(&lin, &nlin)?;
            u = u.clone() + delta;
            outputs.push(u.clone());
        }

        if outputs.is_empty() {
            Ok(Tensor::zeros([steps, n_dim], &self.device))
        } else {
            Ok(Tensor::cat(outputs, 0))
        }
    }

    fn predict_from_features(
        &self,
        lin_features: &Tensor<B, 2>,
        nlin_features: &Tensor<B, 2>,
    ) -> Result<Tensor<B, 2>, String> {
        let wout = self
            .wout
            .as_ref()
            .ok_or_else(|| "model is not fitted".to_string())?;

        let mut tot = Tensor::cat(vec![lin_features.clone(), nlin_features.clone()], 1);
        if self.config.bias {
            let shape = tot.shape();
            let ones = Tensor::ones([shape.dims[0], 1], &self.device);
            tot = Tensor::cat(vec![ones, tot], 1);
        }

        Ok(tot.matmul(wout.clone().transpose()))
    }

    fn nvar(
        &self,
        x: &Tensor<B, 2>,
        window: Option<Tensor<B, 2>>,
    ) -> Result<(Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>), String> {
        // Convert input tensor to ndarray for processing
        let x_data = x.clone().into_data().convert::<f32>();
        let x_vec = x_data.as_slice::<f32>().unwrap_or(&[]).to_vec();
        let x_shape = x.shape();
        let x_ndarray = Array2::from_shape_vec([x_shape.dims[0], x_shape.dims[1]], x_vec)
            .map_err(|e| format!("Failed to create ndarray: {}", e))?;

        let k = self.config.delay;
        let s_stride = self.config.stride;
        let p = self.config.poly_order;

        if k < 1 || s_stride < 1 || p < 1 {
            return Err("delay/stride/poly_order must be >= 1".to_string());
        }

        let n_steps = x_ndarray.nrows();
        let n_dim = x_ndarray.ncols();
        let lin_dim = n_dim * k;

        let monom_idx = combinations_with_replacement(lin_dim, p);
        let nlin_dim = monom_idx.len();

        let win_dim = (k - 1) * s_stride + 1;

        let mut win = if let Some(w) = window {
            let w_data = w.clone().into_data().convert::<f32>();
            let w_vec = w_data.as_slice::<f32>().unwrap_or(&[]).to_vec();
            let w_shape = w.shape();
            if w_shape.dims != [win_dim, n_dim] {
                return Err(format!(
                    "window shape mismatch: expected ({}, {}), got ({}, {})",
                    win_dim, n_dim, w_shape.dims[0], w_shape.dims[1]
                ));
            }
            Array2::from_shape_vec([w_shape.dims[0], w_shape.dims[1]], w_vec)
                .map_err(|e| format!("Failed to create window ndarray: {}", e))?
        } else {
            Array2::<f32>::zeros((win_dim, n_dim))
        };

        let mut lin_features = Array2::<f32>::zeros((n_steps, lin_dim));
        let mut nlin_features = Array2::<f32>::zeros((n_steps, nlin_dim));

        for i in 0..n_steps {
            // Shift window
            for row in 0..(win_dim - 1) {
                let next = win.slice(s![row + 1, ..]).to_owned();
                win.slice_mut(s![row, ..]).assign(&next);
            }

            // Add new row to window
            win.slice_mut(s![win_dim - 1, ..])
                .assign(&x_ndarray.slice(s![i, ..]).to_owned());

            // Extract linear features
            let mut lin_feat = vec![0.0f32; lin_dim];
            let mut idx = 0usize;
            for row in (0..win_dim).step_by(s_stride) {
                for col in 0..n_dim {
                    lin_feat[idx] = win[[row, col]];
                    idx += 1;
                }
            }

            // Extract nonlinear features
            let mut nlin_feat = vec![0.0f32; nlin_dim];
            for (j, comb) in monom_idx.iter().enumerate() {
                let mut prod = 1.0f32;
                for &c in comb {
                    prod *= lin_feat[c];
                }
                nlin_feat[j] = prod;
            }

            // Store features
            for j in 0..lin_dim {
                lin_features[[i, j]] = lin_feat[j];
            }
            for j in 0..nlin_dim {
                nlin_features[[i, j]] = nlin_feat[j];
            }
        }

        // Convert ndarray back to Tensor
        let lin_features_vec = lin_features
            .to_shape([n_steps * lin_dim])
            .map_err(|e| e.to_string())?
            .to_vec();
        let nlin_features_vec = nlin_features
            .to_shape([n_steps * nlin_dim])
            .map_err(|e| e.to_string())?
            .to_vec();
        let win_vec = win
            .to_shape([win_dim * n_dim])
            .map_err(|e| e.to_string())?
            .to_vec();

        let lin_tensor = Tensor::<B, 2>::from_data(
            TensorData::new(
                lin_features_vec,
                burn::tensor::Shape::new([n_steps, lin_dim]),
            ),
            &self.device,
        );
        let nlin_tensor = Tensor::<B, 2>::from_data(
            TensorData::new(
                nlin_features_vec,
                burn::tensor::Shape::new([n_steps, nlin_dim]),
            ),
            &self.device,
        );
        let win_tensor = Tensor::<B, 2>::from_data(
            TensorData::new(win_vec, burn::tensor::Shape::new([win_dim, n_dim])),
            &self.device,
        );

        Ok((lin_tensor, nlin_tensor, win_tensor))
    }

    fn total_feature(
        &self,
        lin_features: &Tensor<B, 2>,
        nlin_features: &Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let mut tot = Tensor::cat(vec![lin_features.clone(), nlin_features.clone()], 1);
        let shape = tot.shape();
        let trans = self.config.transients.min(shape.dims[0]);
        tot = tot.slice([trans..shape.dims[0], 0..shape.dims[1]]);

        if self.config.bias {
            let shape_after = tot.shape();
            let ones = Tensor::ones([shape_after.dims[0], 1], &self.device);
            tot = Tensor::cat(vec![ones, tot], 1);
        }
        tot
    }

    fn tikhonov_regression(
        &self,
        lin_features: &Tensor<B, 2>,
        nlin_features: &Tensor<B, 2>,
        target: &Tensor<B, 2>,
    ) -> Result<Tensor<B, 2>, String> {
        let target_shape = target.shape();
        let trans = self.config.transients.min(target_shape.dims[0]);
        let y = target
            .clone()
            .slice([trans..target_shape.dims[0], 0..target_shape.dims[1]]);
        let x = self.total_feature(lin_features, nlin_features);

        let x_shape = x.shape();
        let y_shape = y.shape();
        if x_shape.dims[0] == 0 || y_shape.dims[0] == 0 {
            return Err("no rows left after applying transients".to_string());
        }

        let yxt = y.clone().transpose().matmul(x.clone());
        let xxt = x.clone().transpose().matmul(x);

        // Add ridge regularization to diagonal
        let xxt_data = xxt.clone().into_data().convert::<f32>();
        let xxt_slice = xxt_data.as_slice::<f32>().unwrap_or(&[]);
        let mut xxt_vec = xxt_slice.to_vec();
        let n = xxt.shape().dims[0];
        for i in 0..n {
            xxt_vec[i * n + i] += self.config.ridge_param;
        }
        let xxt_reg = Tensor::from_data(
            TensorData::new(xxt_vec, burn::tensor::Shape::new([n, n])),
            &self.device,
        );

        let inv = invert_square_tensor::<B>(xxt_reg, &self.device)?;
        Ok(yxt.matmul(inv))
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

fn invert_square_tensor<B: Backend>(
    a: Tensor<B, 2>,
    device: &B::Device,
) -> Result<Tensor<B, 2>, String> {
    let shape = a.shape();
    if shape.dims[0] != shape.dims[1] {
        return Err("matrix must be square".to_string());
    }
    let n = shape.dims[0];

    // Convert to f32 data for computation
    let a_data = a.into_data().convert::<f32>();
    let a_slice = a_data.as_slice::<f32>().unwrap_or(&[]);

    // Create augmented matrix [A | I]
    let mut aug = vec![0.0f32; n * 2 * n];
    for r in 0..n {
        for c in 0..n {
            aug[r * 2 * n + c] = a_slice[r * n + c];
        }
        aug[r * 2 * n + n + r] = 1.0;
    }

    // Gaussian elimination with partial pivoting
    for i in 0..n {
        // Find pivot
        let mut pivot = i;
        let mut max_abs = aug[i * 2 * n + i].abs();
        for r in (i + 1)..n {
            let v = aug[r * 2 * n + i].abs();
            if v > max_abs {
                max_abs = v;
                pivot = r;
            }
        }
        if max_abs < 1e-12 {
            return Err("matrix is singular".to_string());
        }

        // Swap rows
        if pivot != i {
            for c in 0..(2 * n) {
                let tmp = aug[i * 2 * n + c];
                aug[i * 2 * n + c] = aug[pivot * 2 * n + c];
                aug[pivot * 2 * n + c] = tmp;
            }
        }

        // Normalize pivot row
        let diag = aug[i * 2 * n + i];
        for c in 0..(2 * n) {
            aug[i * 2 * n + c] /= diag;
        }

        // Eliminate below and above
        for r in 0..n {
            if r == i {
                continue;
            }
            let factor = aug[r * 2 * n + i];
            if factor.abs() < 1e-20 {
                continue;
            }
            for c in 0..(2 * n) {
                aug[r * 2 * n + c] -= factor * aug[i * 2 * n + c];
            }
        }
    }

    // Extract inverse from augmented matrix
    let mut inv_data = vec![0.0f32; n * n];
    for r in 0..n {
        for c in 0..n {
            inv_data[r * n + c] = aug[r * 2 * n + n + c];
        }
    }

    Ok(Tensor::from_data(
        TensorData::new(inv_data, burn::tensor::Shape::new([n, n])),
        device,
    ))
}
