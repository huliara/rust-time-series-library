use burn::{
    config::Config,
    module::Module,
    nn::{Dropout, DropoutConfig, Initializer, Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
};

#[derive(Config, Debug)]
pub struct FlattenHeadConfig {
    pub nf: usize,
    pub target_window: usize,
    pub head_dropout: f64,
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer: Initializer,
}

impl FlattenHeadConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> FlattenHead<B> {
        let linear = LinearConfig::new(self.nf, self.target_window)
            .with_initializer(self.initializer.clone())
            .init(device);
        let dropout = DropoutConfig::new(self.head_dropout).init();
        FlattenHead { linear, dropout }
    }
}

#[derive(Module, Debug)]
pub struct FlattenHead<B: Backend> {
    linear: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> FlattenHead<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        let x = x.flatten(-2, -1);
        let x = self.linear.forward(x);
        self.dropout.forward(x)
    }
}
