use burn::prelude::Backend;

use crate::{
    args::{data::DataCommand, time_lengths::TimeLengths},
    data::dataset::{
        init_dynamic_system::InitDynamicSystem,
        init_real_time_series::InitRealTimeSeries,
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
        DataCommand::ETTh1(data_command) => data_command.init::<B>(lengths, flag, device),
        DataCommand::Exchange(data_command) => data_command.init::<B>(lengths, flag, device),
        DataCommand::LogisticMap(data_command) => data_command.init::<B>(lengths, flag, device),
        DataCommand::HenonMap(data_command) => data_command.init::<B>(lengths, flag, device),
        DataCommand::Lorenz(data_command) => data_command.init::<B>(lengths, flag, device),
        DataCommand::Lorenz96(data_command) => data_command.init::<B>(lengths, flag, device),
        DataCommand::Rossler(data_command) => data_command.init::<B>(lengths, flag, device),
        DataCommand::DoubleScroll(data_command) => data_command.init::<B>(lengths, flag, device),
        DataCommand::MultiScroll(data_command) => data_command.init::<B>(lengths, flag, device),
        DataCommand::RabinovichFabrikant(data_command) => {
            data_command.init::<B>(lengths, flag, device)
        }
        DataCommand::MackeyGlass(data_command) => data_command.init::<B>(lengths, flag, device),
        DataCommand::Narma(data_command) => data_command.init::<B>(lengths, flag, device),
        DataCommand::KuramotoSivashinsky(data_command) => {
            data_command.init::<B>(lengths, flag, device)
        }
        DataCommand::BoolTransform(data_command) => data_command.init::<B>(lengths, flag, device),
    }
}
