use burn::prelude::Backend;

use crate::{
    args::{data_config::DataConfig, time_lengths::TimeLengths},
    data::dataset::time_series_dataset::{ExpFlag, TimeSeriesDataset},
};

pub fn get_dataset<B: Backend>(
    data_config: &DataConfig,
    lengths: &TimeLengths,
    flag: ExpFlag,
    device: &B::Device,
) -> TimeSeriesDataset<B> {
    match data_config {
        DataConfig::ETTh1(ref data_command) => {
            TimeSeriesDataset::<B>::new(&data_config, data_command, &lengths, flag, &device)
        }
        DataConfig::Exchange(ref data_command) => {
            TimeSeriesDataset::<B>::new(&data_config, data_command, &lengths, flag, &device)
        }
    }
}
