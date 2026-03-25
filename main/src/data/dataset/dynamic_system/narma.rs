use burn::prelude::Backend;
use clap::Args;
use rand::{rngs::StdRng, Rng, SeedableRng};
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
pub struct NarmaConfig {
    #[arg(long, default_value_t = 10000)]
    pub n_timesteps: usize,
    #[arg(long, default_value_t = 10)]
    pub order: usize,
    #[arg(long, default_value_t = 0.3)]
    pub a1: f64,
    #[arg(long, default_value_t = 0.05)]
    pub a2: f64,
    #[arg(long, default_value_t = 1.5)]
    pub b: f64,
    #[arg(long, default_value_t = 0.1)]
    pub c: f64,
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
}

impl std::fmt::Display for NarmaConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "narma_nt{}_ord{}", self.n_timesteps, self.order)
    }
}

impl InitTimeSeries for NarmaConfig {
    fn split_borders(
        lengths: &TimeLengths,
        total_rows: usize,
    ) -> ((usize, usize, usize), (usize, usize, usize)) {
        split_borders(lengths, total_rows)
    }
}

impl InitDynamicSystem for NarmaConfig {
    fn init<B: Backend>(
        &self,
        lengths: &TimeLengths,
        flag: ExpFlag,
        device: &B::Device,
    ) -> TimeSeriesDataset<B> {
        let x0 = vec![0.0_f64; self.order.max(1)];
        let (_u, y) = narma(
            self.n_timesteps,
            self.order,
            self.a1,
            self.a2,
            self.b,
            self.c,
            x0,
            Some(self.seed),
            None,
        );
        let series = y.into_iter().map(|v| v.to_vec()).collect::<Vec<_>>();
        from_series(series, lengths, flag, device)
    }
}

pub fn narma(
    n_timesteps: usize,
    order: usize,
    a1: f64,
    a2: f64,
    b: f64,
    c: f64,
    x0: Vec<f64>,
    seed: Option<u64>,
    u: Option<Vec<f64>>,
) -> (Vec<[f64; 1]>, Vec<[f64; 1]>) {
    let mut y = vec![[0.0_f64; 1]; n_timesteps + order];

    for (i, v) in x0.iter().enumerate().take(y.len()) {
        y[i][0] = *v;
    }

    let u_series = if let Some(input) = u {
        input
    } else {
        let mut rng = StdRng::seed_from_u64(seed.unwrap_or(42));
        (0..(n_timesteps + order))
            .map(|_| rng.gen_range(0.0..0.5))
            .collect::<Vec<_>>()
    };

    for t in order..(n_timesteps + order - 1) {
        let sum_hist = y[t - order..t].iter().map(|v| v[0]).sum::<f64>();
        y[t + 1][0] =
            a1 * y[t][0] + a2 * y[t][0] * sum_hist + b * u_series[t - order] * u_series[t] + c;
    }

    let u_out = u_series.into_iter().map(|v| [v]).collect::<Vec<_>>();
    let y_out = y[order..].to_vec();
    (u_out, y_out)
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

    use super::narma;

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
    fn test_narma_dataset_against_python() {
        let n_timesteps = 400;
        let order = 10;
        let x0 = vec![0.0_f64; order];
        let u = (0..(n_timesteps + order))
            .map(|idx| (idx % 7) as f64 * 0.05)
            .collect::<Vec<_>>();
        let (_u, narma_y) = narma(n_timesteps, order, 0.3, 0.05, 1.5, 0.1, x0, None, Some(u));
        let narma_series = narma_y.into_iter().map(|v| v.to_vec()).collect::<Vec<_>>();
        assert_dataset_matches_python("narma", narma_series);
    }
}
