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

impl InitTimeSeries for HenonMapConfig {}

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
        Self::from_series(series, lengths, flag, device)
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
    use crate::data::dataset::dynamic_system::test::{
        assert_dynamic_system_series, TEST_STEP_SIZE,
    };

    use super::henon_map;

    #[test]
    fn test_henon_map_dataset_against_python() {
        let n_timesteps = TEST_STEP_SIZE;
        let series = henon_map(n_timesteps, 1.4, 0.3, [0.0, 0.0])
            .into_iter()
            .map(|v| v.to_vec())
            .collect::<Vec<_>>();

        let system_name = "henon_map";
        assert_dynamic_system_series(system_name, series);
    }
}
