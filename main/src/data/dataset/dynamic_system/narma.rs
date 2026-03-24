use burn::prelude::Backend;
use chrono::NaiveDateTime;
use clap::Args;
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::{
    args::time_lengths::TimeLengths,
    data::dataset::{
        dynamic_system::config::{
            default_columns, default_embed, default_parse_dates, default_path, from_series,
            split_borders, DynamicColumnName,
        },
        init_real_time_series::InitRealTimeSeries,
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

impl InitRealTimeSeries<DynamicColumnName> for NarmaConfig {
    fn parse_dates(
        _df: &polars::prelude::DataFrame,
        start_idx: usize,
        slice_len: usize,
    ) -> Vec<NaiveDateTime> {
        default_parse_dates(start_idx, slice_len)
    }

    fn path(&self) -> String {
        default_path()
    }

    fn train_columns(&self) -> Vec<DynamicColumnName> {
        default_columns()
    }

    fn target_columns(&self) -> Vec<DynamicColumnName> {
        default_columns()
    }

    fn embed(&self) -> crate::args::time_embed::TimeEmbed {
        default_embed()
    }

    fn split_borders(
        lengths: &TimeLengths,
        total_rows: usize,
    ) -> ((usize, usize, usize), (usize, usize, usize)) {
        split_borders(lengths, total_rows)
    }

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
