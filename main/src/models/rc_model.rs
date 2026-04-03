pub mod ngrc;
pub mod rc;
use self::ngrc::{NGRCState, Ngrc};
use crate::args::model::rc_model::RCModelCommand;
use burn::prelude::Backend;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub enum RCModelState {
    NGRC(NGRCState),
}

impl RCModelCommand {
    pub fn init<B: Backend>(&self, device: &B::Device) -> RCModel<B> {
        match self {
            RCModelCommand::NGRC(cmd) => RCModel::NGRC(cmd.model_args.clone().init(device)),
        }
    }
}

pub enum RCModel<B: Backend> {
    NGRC(Ngrc<B>),
}

impl<B: Backend> RCModel<B> {
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let state = match self {
            RCModel::NGRC(ngrc) => RCModelState::NGRC(ngrc.get_state()),
        };

        let yaml = serde_yaml::to_string(&state)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

        std::fs::write(path, yaml)
    }

    pub fn load(device: &B::Device, path: &str) -> std::io::Result<Self> {
        let yaml = std::fs::read_to_string(path)?;
        let state: RCModelState = serde_yaml::from_str(&yaml)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

        match state {
            RCModelState::NGRC(ngrc_state) => {
                Ok(RCModel::NGRC(Ngrc::from_state(device, ngrc_state)))
            }
        }
    }
}
