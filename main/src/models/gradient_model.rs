pub mod dlinear;
pub mod patch_tst;
pub mod time_xer;
use crate::{
    args::{exp::TaskName, model::gradient_model::GradientModelCommand, time_lengths::TimeLengths},
    models::gradient_model::{
        dlinear::{DLinear, DLinearConfig},
        patch_tst::{PatchTST, PatchTSTConfig},
        time_xer::{TimeXer, TimeXerConfig},
    },
};
use burn::prelude::*;

impl GradientModelCommand {
    pub fn init<B: Backend>(
        &self,
        task_name: TaskName,
        lengths: TimeLengths,
        device: &B::Device,
    ) -> GradientModel<B> {
        match self {
            GradientModelCommand::PatchTST(cmd) => GradientModel::PatchTST(
                PatchTSTConfig::new(cmd.model_args.clone()).init(task_name, lengths, device),
            ),
            GradientModelCommand::DLinear(cmd) => GradientModel::DLinear(
                DLinearConfig::new(cmd.model_args.clone()).init(task_name, lengths, device),
            ),
            GradientModelCommand::TimeXer(cmd) => {
                let input_dim = cmd.data_command.input_dim();
                GradientModel::TimeXer(
                    TimeXerConfig::new(cmd.model_args.clone())
                        .init(task_name, lengths, input_dim, device),
                )
            }
        }
    }
}

#[derive(Module, Debug)]
pub enum GradientModel<B: Backend> {
    PatchTST(PatchTST<B>),
    DLinear(DLinear<B>),
    TimeXer(TimeXer<B>),
}
