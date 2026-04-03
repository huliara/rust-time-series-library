use burn::prelude::Backend;

use crate::args::model::rc_model::RCModelCommand;
pub mod ngrc;
pub mod rc;

impl RCModelCommand {
    pub fn init<B: Backend>(&self, device: &B::Device) -> RCModel<B> {
        match self {
            RCModelCommand::NGRC(cmd) => RCModel::NGRC(cmd.model_args.clone().init(device)),
        }
    }
}
pub enum RCModel<B: Backend> {
    NGRC(ngrc::NGRC<B>),
}
