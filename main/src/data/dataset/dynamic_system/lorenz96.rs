use burn::prelude::Backend;
use clap::Args;
use serde::{Deserialize, Serialize};

use crate::{
    args::time_lengths::TimeLengths,
    data::dataset::{
        dynamic_system::config::{from_series, split_borders},
        init_dynamic_system::InitDynamicSystem,
        init_time_series::InitTimeSeries,
        time_series_dataset::{ExpFlag, TimeSeriesDataset},
    },
};
#[derive(Args, Debug, Clone, Deserialize, Serialize)]
pub struct Lorenz96Config {
    #[arg(long, default_value_t = 10000)]
    pub total_steps: usize,
    #[arg(long, default_value_t = 36)]
    pub dimention: usize,
    #[arg(long, default_value_t = 8.0)]
    pub f: f64,
    #[arg(long, default_value_t = 0.01)]
    pub dt: f64,
    #[arg(long, default_value_t = 0.01)]
    pub h: f64,
    #[arg(long, num_args = 1..)]
    pub initial_value: Vec<f64>,
}

impl Default for Lorenz96Config {
    fn default() -> Self {
        Self {
            total_steps: 10000,
            dimention: 36,
            f: 8.0,
            dt: 0.01,
            h: 0.01,
            initial_value: Vec::new(),
        }
    }
}

impl std::fmt::Display for Lorenz96Config {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "lorenz96_nt{}_dim{}_f{:.2}",
            self.total_steps, self.dimention, self.f
        )
    }
}

impl InitTimeSeries for Lorenz96Config {
    fn split_borders(
        lengths: &TimeLengths,
        total_rows: usize,
    ) -> ((usize, usize, usize), (usize, usize, usize)) {
        split_borders(lengths, total_rows)
    }
}

impl InitDynamicSystem for Lorenz96Config {
    fn init<B: Backend>(
        &self,
        lengths: &TimeLengths,
        flag: ExpFlag,
        device: &B::Device,
    ) -> TimeSeriesDataset<B> {
        let x0 = if self.initial_value.is_empty() {
            None
        } else {
            Some(self.initial_value.clone())
        };
        let series = lorenz96(
            self.total_steps,
            0,
            self.dimention,
            self.f,
            self.dt,
            self.h,
            x0,
        )
        .expect("Failed to generate lorenz96 series");
        from_series(series, lengths, flag, device)
    }
}

fn lorenz96_diff(state: &[f64], f: f64) -> Vec<f64> {
    let n = state.len();
    let mut ds = vec![0.0_f64; n];
    for i in 0..n {
        let ip1 = (i + 1) % n;
        let im1 = (i + n - 1) % n;
        let im2 = (i + n - 2) % n;
        ds[i] = (state[ip1] - state[im2]) * state[im1] - state[i] + f;
    }
    ds
}

fn rk4_step(state: &[f64], dt: f64, f: f64) -> Vec<f64> {
    let k1 = lorenz96_diff(state, f);
    let s2 = state
        .iter()
        .zip(k1.iter())
        .map(|(x, k)| x + 0.5 * dt * k)
        .collect::<Vec<_>>();
    let k2 = lorenz96_diff(&s2, f);
    let s3 = state
        .iter()
        .zip(k2.iter())
        .map(|(x, k)| x + 0.5 * dt * k)
        .collect::<Vec<_>>();
    let k3 = lorenz96_diff(&s3, f);
    let s4 = state
        .iter()
        .zip(k3.iter())
        .map(|(x, k)| x + dt * k)
        .collect::<Vec<_>>();
    let k4 = lorenz96_diff(&s4, f);

    (0..state.len())
        .map(|i| state[i] + dt * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0)
        .collect::<Vec<_>>()
}

pub fn lorenz96(
    n_timesteps: usize,
    warmup: usize,
    n: usize,
    f: f64,
    df: f64,
    h: f64,
    x0: Option<Vec<f64>>,
) -> Result<Vec<Vec<f64>>, String> {
    if n < 4 {
        return Err("N must be >= 4.".to_string());
    }
    if n_timesteps == 0 {
        return Ok(Vec::new());
    }

    let mut state = if let Some(initial) = x0 {
        if initial.len() != n {
            return Err(format!(
                "x0 should have shape ({n},), but has length {}",
                initial.len()
            ));
        }
        initial
    } else {
        let mut init = vec![f; n];
        init[0] = f + df;
        init
    };

    let total_steps = n_timesteps + warmup;
    let mut out = Vec::with_capacity(total_steps);
    out.push(state.clone());

    for _ in 1..total_steps {
        state = rk4_step(&state, h, f);
        out.push(state.clone());
    }

    Ok(out.into_iter().skip(warmup).collect::<Vec<_>>())
}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;

    use crate::{
        args::time_lengths::TimeLengths,
        data::dataset::{
            dynamic_system::config::from_series, time_series_dataset::ExpFlag,
        },
        test_utils::{
            assert_tensor_shape_value::assert_tensor_shape_and_val,
            test_py::execute_dynamic_system_dataset_test,
        },
    };

    use super::lorenz96;

    type B = burn::backend::wgpu::Wgpu;

    fn assert_dataset_matches_python(system_name: &str, series: Vec<Vec<f64>>) {
        let lengths = TimeLengths::default();
        let device = Default::default();

        let py_dataset_result = execute_dynamic_system_dataset_test(system_name).unwrap();
        let rust_dataset = from_series::<B>(series, &lengths, ExpFlag::Test, &device);

        let py_tensor_stamp =
            TensorData::new(py_dataset_result.1, rust_dataset.data_stamp.shape());
        let rust_tensor_stamp = rust_dataset.data_stamp.to_data();
        assert_tensor_shape_and_val(py_tensor_stamp, rust_tensor_stamp);

        let py_tensor_x = TensorData::new(py_dataset_result.0, rust_dataset.data_x.shape());
        let rust_tensor_x = rust_dataset.data_x.to_data();
        assert_tensor_shape_and_val(py_tensor_x, rust_tensor_x);

        let py_tensor_y = TensorData::new(py_dataset_result.2, rust_dataset.data_y.shape());
        let rust_tensor_y = rust_dataset.data_y.to_data();
        assert_tensor_shape_and_val(py_tensor_y, rust_tensor_y);
    }

    #[test]
    fn test_lorenz96_dataset_against_python() {
        let n_timesteps = 400;
        let lorenz96_series = lorenz96(n_timesteps, 0, 8, 8.0, 0.01, 0.01, None).unwrap();
        assert_dataset_matches_python("lorenz96", lorenz96_series);
    }
}
