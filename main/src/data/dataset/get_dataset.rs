use burn::prelude::Backend;

use crate::{
    args::time_lengths::TimeLengths,
    data::{
        data_config::DataConfig,
        dataset::{
            init_dataset::InitDataset,
            real_time_series::{etth1::Etth1Args, exchange::ExchangeArgs},
            time_series_dataset::{ExpFlag, TimeSeriesDataset},
        },
    },
};

pub fn get_dataset<B: Backend>(
    data_config: &DataConfig,
    lengths: &TimeLengths,
    flag: ExpFlag,
    device: &B::Device,
) -> TimeSeriesDataset<B> {
    match data_config {
        DataConfig::ETTh1(data_command) => Etth1Args::init::<B>(
            data_command.path.clone(),
            data_command.train_features.clone(),
            data_command.targets.clone(),
            data_command.embed.clone(),
            lengths,
            flag,
            device,
        ),
        DataConfig::Exchange(data_command) => ExchangeArgs::init::<B>(
            data_command.path.clone(),
            data_command.train_features.clone(),
            data_command.targets.clone(),
            data_command.embed.clone(),
            lengths,
            flag,
            device,
        ),
    }
}
