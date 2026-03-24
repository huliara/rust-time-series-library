use std::fmt::{Debug, Display};

use crate::{
    args::time_lengths::TimeLengths,
    data::dataset::{
        init_time_series::InitTimeSeries,
        time_series_dataset::{ExpFlag, TimeSeriesDataset},
    },
};

use burn::prelude::Backend;
use clap::ValueEnum;

pub trait InitRealTimeSeries<
    C: Clone
        + std::marker::Send
        + std::marker::Sync
        + 'static
        + ValueEnum
        + Display
        + Debug
        + PartialEq,
>: InitTimeSeries
{
    fn init<B: Backend>(
        &self,
        lengths: &TimeLengths,
        flag: ExpFlag,
        device: &B::Device,
    ) -> TimeSeriesDataset<B>;
}
