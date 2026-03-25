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
pub struct HenonMapConfig {
    #[arg(long, default_value_t = 10000)]
    pub n_timesteps: usize,
    #[arg(long, default_value_t = 1.4)]
    pub a: f64,
    #[arg(long, default_value_t = 0.3)]
    pub b: f64,
    #[arg(long, num_args = 2, default_values_t = [0.0, 0.0])]
    pub initial_value: Vec<f64>,
}

impl std::fmt::Display for HenonMapConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "henon_nt{}_a{:.2}_b{:.2}",
            self.n_timesteps, self.a, self.b
        )
    }
}

impl InitTimeSeries for HenonMapConfig {
    fn split_borders(
        lengths: &TimeLengths,
        total_rows: usize,
    ) -> ((usize, usize, usize), (usize, usize, usize)) {
        split_borders(lengths, total_rows)
    }
}

impl InitDynamicSystem for HenonMapConfig {
    fn init<B: Backend>(
        &self,
        lengths: &TimeLengths,
        flag: ExpFlag,
        device: &B::Device,
    ) -> TimeSeriesDataset<B> {
        if self.initial_value.len() != 2 {
            panic!("henon_map initial_value must contain exactly 2 elements");
        }
        let series = henon_map(
            self.n_timesteps,
            self.a,
            self.b,
            [self.initial_value[0], self.initial_value[1]],
        )
        .into_iter()
        .map(|v| v.to_vec())
        .collect::<Vec<_>>();
        from_series(series, lengths, flag, device)
    }
}

pub fn henon_map(n_timesteps: usize, a: f64, b: f64, x0: [f64; 2]) -> Vec<[f64; 2]> {
    if n_timesteps == 0 {
        return Vec::new();
    }

    let mut states = vec![[0.0, 0.0]; n_timesteps];
    states[0] = x0;

    for i in 1..n_timesteps {
        states[i][0] = 1.0 - a * states[i - 1][0] * states[i - 1][0] + states[i - 1][1];
        states[i][1] = b * states[i - 1][0];
    }

    states
}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;

    use crate::{
        args::time_lengths::TimeLengths,
        data::dataset::{dynamic_system::config::from_series, time_series_dataset::ExpFlag},
        test_utils::{
            assert_tensor_shape_value::assert_tensor_shape_and_val,
            test_py::execute_dynamic_system_dataset_test,
        },
    };

    use super::henon_map;

    type B = burn::backend::wgpu::Wgpu;

    fn assert_dataset_matches_python(system_name: &str, series: Vec<Vec<f64>>) {
        let lengths = TimeLengths::default();
        let device = Default::default();

        let py_dataset_result = execute_dynamic_system_dataset_test(system_name).unwrap();
        let rust_dataset = from_series::<B>(series, &lengths, ExpFlag::Test, &device);

        let py_tensor_stamp = TensorData::new(py_dataset_result.1, rust_dataset.data_stamp.shape());
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
    fn test_henon_map_dataset_against_python() {
        let n_timesteps = 400;
        let henon_series = henon_map(n_timesteps, 1.4, 0.3, [0.0, 0.0])
            .into_iter()
            .map(|v| v.to_vec())
            .collect::<Vec<_>>();
        assert_dataset_matches_python("henon_map", henon_series);
    }
}
