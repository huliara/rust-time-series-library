use burn::prelude::Backend;

use crate::{
    args::{data::DataCommand, time_lengths::TimeLengths},
    data::dataset::{
        init_dataset::InitDataset,
        real_time_series::{etth1::Etth1Config, exchange::ExchangeConfig},
        time_series_dataset::{ExpFlag, TimeSeriesDataset},
    },
};

pub fn get_dataset<B: Backend>(
    data_config: &DataCommand,
    lengths: &TimeLengths,
    flag: ExpFlag,
    device: &B::Device,
) -> TimeSeriesDataset<B> {
    match data_config {
        DataCommand::ETTh1(data_command) => Etth1Config::init::<B>(
            data_command.path.clone(),
            data_command.train_features.clone(),
            data_command.targets.clone(),
            data_command.embed.clone(),
            lengths,
            flag,
            device,
        ),
        DataCommand::Exchange(data_command) => ExchangeConfig::init::<B>(
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
