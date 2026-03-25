use burn::prelude::Backend;
use clap::Args;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

use crate::{
    args::time_lengths::TimeLengths,
    data::dataset::{
        dynamic_system::config::{from_series, split_borders},
        init_dynamic_system::InitDynamicSystem as InitDynamicSystem,
        init_time_series::InitTimeSeries,
        time_series_dataset::{ExpFlag, TimeSeriesDataset},
    },
};

use super::_kuramoto_sivashinsky::_kuramoto_sivashinsky;

#[derive(Args, Debug, Clone, Deserialize, Serialize)]
pub struct KuramotoSivashinskyConfig {
    #[arg(long, default_value_t = 10000)]
    pub n_timesteps: usize,
    #[arg(long, default_value_t = 0)]
    pub warmup: usize,
    #[arg(long, default_value_t = 64)]
    pub n: usize,
    #[arg(long, default_value_t = 16.0)]
    pub m: f64,
    #[arg(long, default_value_t = 0.25)]
    pub h: f64,
}

impl std::fmt::Display for KuramotoSivashinskyConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ks_nt{}_n{}_m{:.1}", self.n_timesteps, self.n, self.m)
    }
}

impl InitTimeSeries for KuramotoSivashinskyConfig {
    fn split_borders(
        lengths: &TimeLengths,
        total_rows: usize,
    ) -> ((usize, usize, usize), (usize, usize, usize)) {
        split_borders(lengths, total_rows)
    }
}

impl InitDynamicSystem for KuramotoSivashinskyConfig {
    fn init<B: Backend>(
        &self,
        lengths: &TimeLengths,
        flag: ExpFlag,
        device: &B::Device,
    ) -> TimeSeriesDataset<B> {
        let series =
            kuramoto_sivashinsky(self.n_timesteps, self.warmup, self.n, self.m, None, self.h)
                .expect("Failed to generate kuramoto_sivashinsky series");
        from_series(series, lengths, flag, device)
    }
}

pub fn kuramoto_sivashinsky(
    n_timesteps: usize,
    warmup: usize,
    n: usize,
    m: f64,
    x0: Option<Vec<f64>>,
    h: f64,
) -> Result<Vec<Vec<f64>>, String> {
    let initial = if let Some(x0) = x0 {
        if x0.len() != n {
            return Err(format!(
                "Initial condition x0 should be of shape {n} (= N) but has length {}",
                x0.len()
            ));
        }
        x0
    } else {
        (1..=n)
            .map(|idx| {
                let x = 2.0 * m * PI * (idx as f64) / (n as f64);
                (x / m).cos() * (1.0 + (x / m).sin())
            })
            .collect::<Vec<_>>()
    };

    _kuramoto_sivashinsky(n_timesteps, warmup, n, m, initial, h)
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

    use super::kuramoto_sivashinsky;

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
    fn test_kuramoto_sivashinsky_dataset_against_python() {
        let n_timesteps = 120;
        let ks_series = kuramoto_sivashinsky(n_timesteps, 0, 16, 8.0, None, 0.25).unwrap();
        assert_dataset_matches_python("kuramoto_sivashinsky", ks_series);
    }
}
