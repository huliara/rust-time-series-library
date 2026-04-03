use burn::prelude::Backend;
use clap::Args;
use serde::{Deserialize, Serialize};

use crate::{
    args::time_lengths::TimeLengths,
    data::dataset::{
        init_dynamic_system::InitDynamicSystem,
        init_time_series::InitTimeSeries,
        time_series_dataset::{ExpFlag, TimeSeriesDataset},
    },
};

#[derive(Args, Debug, Clone, Deserialize, Serialize)]
pub struct BoolTransformConfig {
    #[arg(long, default_value_t = 10000)]
    pub n_timesteps: usize,
    #[arg(long, default_value_t = 0.4999999999)]
    pub a: f64,
    #[arg(long, default_value_t = 0.5)]
    pub x0: f64,
}

impl std::fmt::Display for BoolTransformConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "bool_transform_nt{}_a{:.2}_x0{:.2}",
            self.n_timesteps, self.a, self.x0
        )
    }
}

impl InitTimeSeries for BoolTransformConfig {}

impl InitDynamicSystem for BoolTransformConfig {
    fn init<B: Backend>(
        &self,
        lengths: &TimeLengths,
        flag: ExpFlag,
        device: &B::Device,
    ) -> TimeSeriesDataset<B> {
        let series = bool_transform(self.n_timesteps, self.a, self.x0)
            .into_iter()
            .map(|v| v.to_vec())
            .collect::<Vec<_>>();
        Self::from_series(series, lengths, flag, device)
    }
}

pub fn bool_transform(n_timesteps: usize, a: f64, x0: f64) -> Vec<[f64; 1]> {
    if n_timesteps == 0 {
        return Vec::new();
    }
    let mut x = vec![[x0; 1]; n_timesteps];
    for i in 1..n_timesteps {
        x[i][0] = a * (x[i - 1][0] - 1.0 / x[i - 1][0]);
    }
    x
}
