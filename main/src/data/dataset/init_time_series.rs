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
use burn::prelude::Backend;

pub trait InitTimeSeries {
    fn split_borders(
        lengths: &TimeLengths,
        total_rows: usize,
    ) -> ((usize, usize, usize), (usize, usize, usize));
}
