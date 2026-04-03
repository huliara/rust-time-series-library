use burn::prelude::Backend;

use crate::args::model::rc_model::RCModelConfig;
pub mod ngrc;
pub mod rc;

pub enum RCModel<B: Backend> {
    NGRC(ngrc::NGRC<B>),
}

impl RCModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> RCModel<B> {
        match self {
            RCModelConfig::NGRC(cmd) => RCModel::NGRC(cmd.model_args.clone().init(device)),
        }
    }
}
