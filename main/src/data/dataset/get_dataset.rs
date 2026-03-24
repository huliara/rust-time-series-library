use burn::{data, prelude::Backend};

use crate::{
    args::time_lengths::TimeLengths,
    data::{
        data_config::{init_dataset::InitDataset, DataConfig},
        dataset::time_series_dataset::{ExpFlag, TimeSeriesDataset},
    },
};

pub fn get_dataset<B: Backend>(
    data_config: &DataConfig,
    lengths: &TimeLengths,
    flag: ExpFlag,
    device: &B::Device,
) -> TimeSeriesDataset<B> {
    match data_config {
        DataConfig::ETTh1(ref data_command) => data_command.init_dataset(lengths, flag, device),
        DataConfig::Exchange(ref data_command) => data_command.init_dataset(lengths, flag, device),
    }
}
