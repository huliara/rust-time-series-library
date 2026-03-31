use burn::prelude::Backend;
use clap::Args;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

use crate::{
    args::time_lengths::TimeLengths,
    data::dataset::{
        init_dynamic_system::InitDynamicSystem,
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

impl InitTimeSeries for KuramotoSivashinskyConfig {}

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
        Self::from_series(series, lengths, flag, device)
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
    use crate::data::dataset::dynamic_system::test::assert_dynamic_system_series;

    use super::kuramoto_sivashinsky;

    #[test]
    fn test_kuramoto_sivashinsky_dataset_against_python() {
        let n_timesteps = 120;
        let series = kuramoto_sivashinsky(n_timesteps, 0, 16, 8.0, None, 0.25).unwrap();

        let system_name = "kuramoto_sivashinsky";
        assert_dynamic_system_series(system_name, series);
    }
}
