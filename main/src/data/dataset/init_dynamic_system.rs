use crate::{
    args::time_lengths::TimeLengths,
    data::dataset::{
        init_time_series::InitTimeSeries,
        time_series_dataset::{ExpFlag, TimeSeriesDataset},
    },
};

use burn::prelude::Backend;
pub trait InitDynamicSystem<C = ()>: InitTimeSeries {
    fn init<B: Backend>(
        &self,
        lengths: &TimeLengths,
        flag: ExpFlag,
        device: &B::Device,
    ) -> TimeSeriesDataset<B>;
}
