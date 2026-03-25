use clap::Args;
use serde::{Deserialize, Serialize};

use crate::{
    args::time_lengths::TimeLengths,
    data::dataset::{
        dynamic_system::config::{from_series, split_borders},
        init_dynamic_system::InitDynamicSystem as InitDynamicSystem,
        init_time_series::InitTimeSeries,
        time_series_dataset::{ExpFlag, TimeSeriesDataset},
    },
};
use burn::prelude::Backend;

#[derive(Args, Debug, Clone, Deserialize, Serialize)]
pub struct RosslerConfig {
    #[arg(long, default_value_t = 10000)]
    pub n_timesteps: usize,
    #[arg(long, default_value_t = 0.2)]
    pub a: f64,
    #[arg(long, default_value_t = 0.2)]
    pub b: f64,
    #[arg(long, default_value_t = 5.7)]
    pub c: f64,
    #[arg(long, default_value_t = 0.01)]
    pub h: f64,
    #[arg(long, num_args = 3, default_values_t = [1.0, 1.0, 1.0])]
    pub initial_value: Vec<f64>,
}

impl std::fmt::Display for RosslerConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "rossler_nt{}_a{:.2}_b{:.2}_c{:.2}",
            self.n_timesteps, self.a, self.b, self.c
        )
    }
}

impl InitTimeSeries for RosslerConfig {
    fn split_borders(
        lengths: &TimeLengths,
        total_rows: usize,
    ) -> ((usize, usize, usize), (usize, usize, usize)) {
        split_borders(lengths, total_rows)
    }
}

impl InitDynamicSystem for RosslerConfig {
    fn init<B: Backend>(
        &self,
        lengths: &TimeLengths,
        flag: ExpFlag,
        device: &B::Device,
    ) -> TimeSeriesDataset<B> {
        if self.initial_value.len() != 3 {
            panic!("rossler initial_value must contain exactly 3 elements");
        }
        let series = rossler(
            self.n_timesteps,
            self.a,
            self.b,
            self.c,
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

fn rossler_diff(state: [f64; 3], a: f64, b: f64, c: f64) -> [f64; 3] {
    let x = state[0];
    let y = state[1];
    let z = state[2];
    [-y - z, x + a * y, b + z * (x - c)]
}

fn rk4_step(state: [f64; 3], dt: f64, a: f64, b: f64, c: f64) -> [f64; 3] {
    let k1 = rossler_diff(state, a, b, c);
    let s2 = [
        state[0] + 0.5 * dt * k1[0],
        state[1] + 0.5 * dt * k1[1],
        state[2] + 0.5 * dt * k1[2],
    ];
    let k2 = rossler_diff(s2, a, b, c);
    let s3 = [
        state[0] + 0.5 * dt * k2[0],
        state[1] + 0.5 * dt * k2[1],
        state[2] + 0.5 * dt * k2[2],
    ];
    let k3 = rossler_diff(s3, a, b, c);
    let s4 = [
        state[0] + dt * k3[0],
        state[1] + dt * k3[1],
        state[2] + dt * k3[2],
    ];
    let k4 = rossler_diff(s4, a, b, c);

    [
        state[0] + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
        state[1] + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
        state[2] + dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
    ]
}

pub fn rossler(n_timesteps: usize, a: f64, b: f64, c: f64, x0: [f64; 3], h: f64) -> Vec<[f64; 3]> {
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

    use super::rossler;

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
    fn test_rossler_dataset_against_python() {
        let n_timesteps = 400;
        let rossler_series = rossler(n_timesteps, 0.2, 0.2, 5.7, [1.0, 1.0, 1.0], 0.01)
            .into_iter()
            .map(|v| v.to_vec())
            .collect::<Vec<_>>();
        assert_dataset_matches_python("rossler", rossler_series);
    }
}
