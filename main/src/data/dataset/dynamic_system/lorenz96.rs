use burn::prelude::Backend;
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
use ode_solvers::{DVector, Dop853, System};

struct Lorenz96System {
    forcing: f64,
}

impl System<f64, DVector<f64>> for Lorenz96System {
    fn system(&self, _t: f64, y: &DVector<f64>, dy: &mut DVector<f64>) {
        let n = y.len();
        for i in 0..n {
            let ip1 = (i + 1) % n;
            let im1 = (i + n - 1) % n;
            let im2 = (i + n - 2) % n;
            dy[i] = (y[ip1] - y[im2]) * y[im1] - y[i] + self.forcing;
        }
    }
}
#[derive(Args, Debug, Clone, Deserialize, Serialize)]
pub struct Lorenz96Config {
    #[arg(long, default_value_t = 10000)]
    pub total_steps: usize,
    #[arg(long, default_value_t = 36)]
    pub dimention: usize,
    #[arg(long, default_value_t = 8.0)]
    pub f: f64,
    #[arg(long, default_value_t = 0.01)]
    pub dt: f64,
    #[arg(long, default_value_t = 0.01)]
    pub h: f64,
    #[arg(long, num_args = 1..)]
    pub initial_value: Vec<f64>,
}

impl Default for Lorenz96Config {
    fn default() -> Self {
        Self {
            total_steps: 10000,
            dimention: 36,
            f: 8.0,
            dt: 0.01,
            h: 0.01,
            initial_value: Vec::new(),
        }
    }
}

impl std::fmt::Display for Lorenz96Config {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "lorenz96_nt{}_dim{}_f{:.2}",
            self.total_steps, self.dimention, self.f
        )
    }
}

impl InitTimeSeries for Lorenz96Config {
    fn split_borders(
        lengths: &TimeLengths,
        total_rows: usize,
    ) -> ((usize, usize, usize), (usize, usize, usize)) {
        split_borders(lengths, total_rows)
    }
}

impl InitDynamicSystem for Lorenz96Config {
    fn init<B: Backend>(
        &self,
        lengths: &TimeLengths,
        flag: ExpFlag,
        device: &B::Device,
    ) -> TimeSeriesDataset<B> {
        let x0 = if self.initial_value.is_empty() {
            None
        } else {
            Some(self.initial_value.clone())
        };
        let series = lorenz96(
            self.total_steps,
            0,
            self.dimention,
            self.f,
            self.dt,
            self.h,
            x0,
        )
        .expect("Failed to generate lorenz96 series");
        from_series(series, lengths, flag, device)
    }
}

pub fn lorenz96(
    n_timesteps: usize,
    warmup: usize,
    n: usize,
    f: f64,
    df: f64,
    h: f64,
    x0: Option<Vec<f64>>,
) -> Result<Vec<Vec<f64>>, String> {
    if n < 4 {
        return Err("N must be >= 4.".to_string());
    }
    if n_timesteps == 0 {
        return Ok(Vec::new());
    }

    let state = if let Some(initial) = x0 {
        if initial.len() != n {
            return Err(format!(
                "x0 should have shape ({n},), but has length {}",
                initial.len()
            ));
        }
        initial
    } else {
        let mut init = vec![f; n];
        init[0] = f + df;
        init
    };

    if n_timesteps == 1 {
        return Ok(vec![state]);
    }

    let t_max = (warmup + n_timesteps) as f64 * h;
    let span_end = t_max * h;
    let dt_eval = span_end / (n_timesteps as f64 - 1.0);
    let system = Lorenz96System { forcing: f };
    let y0 = DVector::from_vec(state);
    let mut stepper = Dop853::new(system, 0.0, span_end, dt_eval, y0, 1e-8, 1e-10);
    stepper
        .integrate()
        .map_err(|e| format!("Failed to solve lorenz96 IVP: {e}"))?;

    let values = stepper
        .y_out()
        .iter()
        .map(|v| v.iter().copied().collect::<Vec<_>>())
        .collect::<Vec<_>>();

    Ok(values.into_iter().skip(warmup).collect::<Vec<_>>())
}

#[cfg(test)]
mod tests {
    use crate::data::dataset::dynamic_system::test::assert_dynamic_system_series;

    use super::lorenz96;

    #[test]
    fn test_lorenz96_dataset_against_python() {
        let n_timesteps = 400;
        let series = lorenz96(n_timesteps, 0, 8, 8.0, 0.01, 0.01, None).unwrap();

        let system_name = "lorenz96";
        assert_dynamic_system_series(system_name, series);
    }
}
