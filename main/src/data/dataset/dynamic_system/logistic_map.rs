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
pub struct LogisticMapConfig {
    #[arg(long, default_value_t = 10000)]
    pub n_timesteps: usize,
    #[arg(long, default_value_t = 3.9)]
    pub r: f64,
    #[arg(long, default_value_t = 0.1)]
    pub x0: f64,
}

impl std::fmt::Display for LogisticMapConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "logistic_nt{}_r{:.3}", self.n_timesteps, self.r)
    }
}

impl InitTimeSeries for LogisticMapConfig {
    fn split_borders(
        lengths: &TimeLengths,
        total_rows: usize,
    ) -> ((usize, usize, usize), (usize, usize, usize)) {
        split_borders(lengths, total_rows)
    }
}

impl InitDynamicSystem for LogisticMapConfig {
    fn init<B: Backend>(
        &self,
        lengths: &TimeLengths,
        flag: ExpFlag,
        device: &B::Device,
    ) -> TimeSeriesDataset<B> {
        let series = logistic_map(self.n_timesteps, self.r, self.x0)
            .expect("Failed to generate logistic_map series")
            .into_iter()
            .map(|v| v.to_vec())
            .collect::<Vec<_>>();
        from_series(series, lengths, flag, device)
    }
}

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

    use super::logistic_map;

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
    fn test_logistic_map_dataset_against_python() {
        let n_timesteps = 400;
        let logistic_series = logistic_map(n_timesteps, 3.9, 0.1)
            .unwrap()
            .into_iter()
            .map(|v| v.to_vec())
            .collect::<Vec<_>>();
        assert_dataset_matches_python("logistic_map", logistic_series);
    }
}
