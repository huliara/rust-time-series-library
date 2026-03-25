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
use burn::prelude::Backend;

#[derive(Args, Debug, Clone, Deserialize, Serialize)]
pub struct DoubleScrollConfig {
    #[arg(long, default_value_t = 10000)]
    pub n_timesteps: usize,
    #[arg(long, default_value_t = 1.2)]
    pub r1: f64,
    #[arg(long, default_value_t = 3.44)]
    pub r2: f64,
    #[arg(long, default_value_t = 0.193)]
    pub r4: f64,
    #[arg(long, default_value_t = 2.25)]
    pub ir: f64,
    #[arg(long, default_value_t = 11.6)]
    pub beta: f64,
    #[arg(long, default_value_t = 0.01)]
    pub h: f64,
    #[arg(long, num_args = 3, default_values_t = [0.1, 0.0, 0.0])]
    pub initial_value: Vec<f64>,
}

impl std::fmt::Display for DoubleScrollConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "doublescroll_nt{}_r1{:.2}_r2{:.2}",
            self.n_timesteps, self.r1, self.r2
        )
    }
}

impl InitTimeSeries for DoubleScrollConfig {
    fn split_borders(
        lengths: &TimeLengths,
        total_rows: usize,
    ) -> ((usize, usize, usize), (usize, usize, usize)) {
        split_borders(lengths, total_rows)
    }
}

impl InitDynamicSystem for DoubleScrollConfig {
    fn init<B: Backend>(
        &self,
        lengths: &TimeLengths,
        flag: ExpFlag,
        device: &B::Device,
    ) -> TimeSeriesDataset<B> {
        if self.initial_value.len() != 3 {
            panic!("doublescroll initial_value must contain exactly 3 elements");
        }
        let series = doublescroll(
            self.n_timesteps,
            self.r1,
            self.r2,
            self.r4,
            self.ir,
            self.beta,
            [
                self.initial_value[0],
                self.initial_value[1],
                self.initial_value[2],
            ],
            self.h,
        )
        .into_iter()
        .map(|v| v.to_vec())
        .collect::<Vec<_>>();
        from_series(series, lengths, flag, device)
    }
}

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

    use super::doublescroll;

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
    fn test_doublescroll_dataset_against_python() {
        let n_timesteps = 400;
        let doublescroll_series = doublescroll(
            n_timesteps,
            1.2,
            3.44,
            0.193,
            2.25,
            11.6,
            [0.1, 0.0, 0.0],
            0.01,
        )
        .into_iter()
        .map(|v| v.to_vec())
        .collect::<Vec<_>>();
        assert_dataset_matches_python("doublescroll", doublescroll_series);
    }
}
