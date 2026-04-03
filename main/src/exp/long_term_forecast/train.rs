use crate::{
    args::model::ModelCommand,
    data::{
        batcher::TimeSeriesBatch,
        data_loader::create_data_loader,
        dataset::{get_dataset::get_dataset, time_series_dataset::ExpFlag},
    },
    exp::{
        long_term_forecast::{
            save_results::plot_samples::plot_samples, GradientForecastModel, LongTermForecastExp,
        },
        loss::barron_loss::BarronLoss,
        Train,
    },
    models::traits::Forecast,
};
use burn::{
    module::AutodiffModule,
    nn::loss::MseLoss,
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
};
use clap::Args;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::{BufWriter, Write},
};
#[derive(Debug, Args, Clone, Deserialize, Serialize)]
pub struct ExpConfig {
    #[arg(long, default_value_t = 10)]
    pub num_epochs: usize,
    #[arg(long, default_value_t = 32)]
    pub batch_size: usize,
    #[arg(long, default_value_t = 4)]
    pub num_workers: usize,
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
    #[arg(long, default_value_t = 1.0e-4)]
    pub learning_rate: f64,
    #[arg(long, default_value_t = 2.0)]
    pub loss_alpha: f64,
    #[arg(long, default_value_t = 1.0)]
    pub loss_scale: f64,
}

impl std::fmt::Display for ExpConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ep{}-bs{}-lr{:.0e}-ls{:.0e}-la{:.1}",
            self.num_epochs, self.batch_size, self.learning_rate, self.loss_scale, self.loss_alpha
        )
    }
}

impl<B: AutodiffBackend> Train<B> for LongTermForecastExp<B> {
    fn train(&self, model_config: ModelCommand)
    where
        B: AutodiffBackend,
    {
        B::seed(&self.device, self.exp_config.seed);
        match model_config {
            ModelCommand::GradientModel(arg) => {
                let mut model = GradientForecastModel::<B>::new(
                    arg.model_command.clone(),
                    self.lengths.clone(),
                    &self.device,
                );
                let dataloader_train = create_data_loader::<B>(
                    &self.data_config,
                    &self.lengths,
                    self.exp_config.batch_size,
                    self.exp_config.num_workers,
                    self.exp_config.seed,
                    ExpFlag::Train,
                );

                let dataloader_valid = create_data_loader::<B::InnerBackend>(
                    &self.data_config,
                    &self.lengths,
                    self.exp_config.batch_size,
                    self.exp_config.num_workers,
                    self.exp_config.seed,
                    ExpFlag::Val,
                );

                let mut optim = AdamConfig::new().init();
                let train_log_root = format!("{0}/train", self.result_path);
                let valid_log_root = format!("{0}/valid", self.result_path);
                fs::create_dir_all(&train_log_root).unwrap();
                fs::create_dir_all(&valid_log_root).unwrap();

                for epoch in 1..=self.exp_config.num_epochs {
                    let mut train_loss_sum = 0.0f64;
                    let mut train_steps = 0usize;
                    let epoch_dir = format!("{train_log_root}/epoch-{epoch}");
                    fs::create_dir_all(&epoch_dir).unwrap();
                    let epoch_loss_log = format!("{epoch_dir}/Loss.log");
                    let epoch_loss_file = fs::File::create(&epoch_loss_log)
                        .expect("Failed to create per-epoch Loss.log");
                    let mut epoch_loss_writer = BufWriter::new(epoch_loss_file);

                    for batch in dataloader_train.iter() {
                        let TimeSeriesBatch {
                            x,
                            x_mark,
                            y,
                            y_mark,
                        } = batch;

                        let mut dec_input = Tensor::zeros_like(&y);
                        dec_input = Tensor::cat(vec![y.clone(), dec_input], 1);

                        let output = model.forecast(x, x_mark, dec_input, y_mark);
                        let loss =
                            BarronLoss::new(self.exp_config.loss_alpha, self.exp_config.loss_scale)
                                .forward(output.clone(), y.clone(), nn::loss::Reduction::Mean);

                        let loss_scalar =
                            loss.clone().into_data().into_vec::<f32>().unwrap()[0] as f64;
                        train_loss_sum += loss_scalar;
                        train_steps += 1;
                        // Keep the same format as Burn trainer logs: `<loss>,1` per iteration.
                        writeln!(&mut epoch_loss_writer, "{loss_scalar},1")
                            .expect("Failed to write training loss log line");

                        let grads = loss.backward();
                        let grads = GradientsParams::from_grads(grads, &model);
                        model = optim.step(self.exp_config.learning_rate, model, grads);
                    }

                    let mut valid_loss_sum = 0.0f64;
                    let mut valid_steps = 0usize;

                    let valid_epoch_dir = format!("{valid_log_root}/epoch-{epoch}");

                    fs::create_dir_all(&valid_epoch_dir).unwrap();
                    let valid_epoch_loss_log = format!("{valid_epoch_dir}/Loss.log");
                    let valid_epoch_loss_file = fs::File::create(&valid_epoch_loss_log)
                        .expect("Failed to create per-epoch validation Loss.log");
                    let mut valid_epoch_loss_writer = BufWriter::new(valid_epoch_loss_file);
                    let model_valid = model.valid();

                    for batch in dataloader_valid.iter() {
                        let TimeSeriesBatch {
                            x,
                            x_mark,
                            y,
                            y_mark,
                        } = batch;

                        let mut dec_input = Tensor::zeros_like(&y);
                        dec_input = Tensor::cat(vec![y.clone(), dec_input], 1);

                        let output = model_valid.forecast(x, x_mark, dec_input, y_mark);
                        let loss = MseLoss::new().forward(output, y, nn::loss::Reduction::Mean);
                        let loss_scalar =
                            loss.clone().into_data().into_vec::<f32>().unwrap()[0] as f64;

                        valid_loss_sum += loss_scalar;
                        valid_steps += 1;
                        // Keep the same format as Burn trainer logs: `<loss>,1` per iteration.
                        writeln!(&mut valid_epoch_loss_writer, "{loss_scalar},1")
                            .expect("Failed to write validation loss log line");
                    }

                    let train_avg = if train_steps > 0 {
                        train_loss_sum / train_steps as f64
                    } else {
                        0.0
                    };
                    let valid_avg = if valid_steps > 0 {
                        valid_loss_sum / valid_steps as f64
                    } else {
                        0.0
                    };

                    println!(
                        "[Epoch {} Summary] train_loss={:.6} valid_loss={:.6}",
                        epoch, train_avg, valid_avg
                    );
                    epoch_loss_writer
                        .flush()
                        .expect("Failed to flush per-epoch Loss.log");
                    valid_epoch_loss_writer
                        .flush()
                        .expect("Failed to flush per-epoch validation Loss.log");
                }

                // Plot a few training samples right after training for quick sanity checks.
                let train_plot_dir = train_log_root;
                let dataloader_train_plot = create_data_loader::<B>(
                    &self.data_config,
                    &self.lengths,
                    self.exp_config.batch_size,
                    self.exp_config.num_workers,
                    self.exp_config.seed,
                    ExpFlag::Train,
                );

                if let Some(batch) = dataloader_train_plot.iter().next() {
                    let contexts = batch.x.clone();
                    let futures = batch.y.clone();
                    let predicts = model.forecast(batch.x, batch.x_mark, batch.y, batch.y_mark);

                    let num_plots = usize::min(5, contexts.dims()[0]);
                    plot_samples(contexts, predicts, futures, num_plots, &train_plot_dir);
                }

                model
                    .save_file(
                        format!("{0}/model", self.result_path),
                        &CompactRecorder::new(),
                    )
                    .expect("Trained model should be saved successfully");
            }
            ModelCommand::RCModel(args) => {
                let model = args.model_config.init::<B>(&self.device);
                let dataset = get_dataset::<B>(
                    &self.data_config,
                    &self.lengths,
                    ExpFlag::Train,
                    &self.device,
                );
                let dataloader = create_data_loader::<B>(
                    &self.data_config,
                    &self.lengths,
                    self.exp_config.batch_size,
                    self.exp_config.num_workers,
                    self.exp_config.seed,
                    ExpFlag::Val,
                );
            }
        }
    }
}
