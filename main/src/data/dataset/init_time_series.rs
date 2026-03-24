use std::{
    fmt::{Debug, Display},
    path::PathBuf,
};

use crate::{
    args::{time_embed::TimeEmbed, time_lengths::TimeLengths},
    data::dataset::{
        standard_scaler::StandardScaler,
        time_series_dataset::{ExpFlag, TimeSeriesDataset},
        util::time_features,
    },
};
use burn::{prelude::Backend, tensor::TensorData, Tensor};
use chrono::{Datelike, NaiveDateTime, Timelike};
use clap::ValueEnum;
use lib::env_path::get_dataset_path;
use ndarray::{s, Array2};
use polars::prelude::*;

pub trait InitTimeSeries {
    fn embed(&self) -> TimeEmbed;

    fn split_borders(
        lengths: &TimeLengths,
        total_rows: usize,
    ) -> ((usize, usize, usize), (usize, usize, usize));
}
