use std::fs;

use crate::{
    args::model::ModelCommand,
    data::{
        batcher::TimeSeriesBatch, data_loader::create_data_loader,
        dataset::time_series_dataset::ExpFlag,
    },
    exp::{
        long_term_forecast::{
            save_results::{plot_samples::plot_samples, save_metric::save_results},
            GradientForecastModel, LongTermForecastExp,
        },
        Infer,
    },
    models::{rc_model::RCModel, traits::Forecast},
};
use burn::{
    prelude::*,
    record::{CompactRecorder, Recorder},
    tensor::backend::AutodiffBackend,
};

impl<B: AutodiffBackend> Infer<B> for LongTermForecastExp<B> {
    fn infer(&self, model_config: ModelCommand) {
        match model_config {
            ModelCommand::GradientModel(config) => {
                let record = CompactRecorder::new()
                    .load(format!("{0}/model", self.result_path).into(), &self.device)
                    .expect("Trained model should exist; run train first");

                let model: GradientForecastModel<B> = GradientForecastModel::<B>::new(
                    config.model_command.clone(),
                    self.lengths.clone(),
                    &self.device,
                )
                .load_record(record);

                let dataloader_test = create_data_loader(
                    &self.data_config,
                    &self.lengths,
                    self.exp_config.batch_size,
                    self.exp_config.num_workers,
                    self.exp_config.seed,
                    ExpFlag::Test,
                );
                let mut _contexts = Vec::with_capacity(3);
                let mut _predicts = Vec::with_capacity(3);
                let mut _futures = Vec::with_capacity(3);
                let test_dir = format!("{0}/test", self.result_path);
                fs::create_dir_all(&test_dir).unwrap();

                for batch in dataloader_test.iter() {
                    let output = model.forecast(
                        batch.x.clone(),
                        batch.x_mark,
                        batch.y.clone(),
                        batch.y_mark,
                    );
                    _contexts.push(batch.x);
                    _predicts.push(output);
                    _futures.push(batch.y);
                }

                let contexts = Tensor::cat(_contexts, 0);
                let predicts = Tensor::cat(_predicts, 0);
                let futures = Tensor::cat(_futures, 0);
                let error = predicts.clone() - futures.clone();
                plot_samples(contexts, predicts, futures.clone(), 12, &test_dir);
                save_results(&test_dir, error, futures);
            }
            ModelCommand::RCModel(_) => {
                let model =
                    RCModel::<B>::load(&self.device, &format!("{0}/model.yaml", self.result_path))
                        .expect("Trained RC model should exist; run train first");

                let dataloader_test = create_data_loader::<B>(
                    &self.data_config,
                    &self.lengths,
                    self.exp_config.batch_size,
                    self.exp_config.num_workers,
                    self.exp_config.seed,
                    ExpFlag::Test,
                );
                let mut _contexts = Vec::with_capacity(3);
                let mut _predicts = Vec::with_capacity(3);
                let mut _futures = Vec::with_capacity(3);
                let test_dir = format!("{0}/test", self.result_path);
                fs::create_dir_all(&test_dir).unwrap();

                match model {
                    RCModel::NGRC(ngrc) => {
                        for batch in dataloader_test.iter() {
                            let TimeSeriesBatch { x, y, .. } = batch;
                            let dims = x.dims();
                            let batch_size = dims[0];
                            let seq_len = dims[1];
                            let n_dim = dims[2];
                            let pred_len = y.dims()[1];

                            let mut outputs = Vec::with_capacity(batch_size);
                            for b in 0..batch_size {
                                let x_b = x
                                    .clone()
                                    .slice([b..b + 1, 0..seq_len, 0..n_dim])
                                    .squeeze_dim::<2>(0);
                                let pred =
                                    ngrc.forecast(&x_b, pred_len).expect("Failed to forecast");
                                outputs.push(pred.unsqueeze_dim::<3>(0));
                            }
                            let output = Tensor::cat(outputs, 0);

                            _contexts.push(x.clone());
                            _predicts.push(output);
                            _futures.push(y);
                        }
                    }
                }

                let contexts = Tensor::cat(_contexts, 0);
                let predicts = Tensor::cat(_predicts, 0);
                let futures = Tensor::cat(_futures, 0);
                let error = predicts.clone() - futures.clone();
                plot_samples(contexts, predicts, futures.clone(), 12, &test_dir);
                save_results(&test_dir, error, futures);
            }
        }
    }
}
