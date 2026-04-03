use std::fs;

use crate::{
    args::{
        data::DataCommand,
        model::{gradient_model::GradientModelConfig, ModelConfig},
        time_lengths::TimeLengths,
    },
    data::{data_loader::create_data_loader, dataset::time_series_dataset::ExpFlag},
    exp::{
        long_term_forecast::{
            save_results::{plot_samples::plot_samples, save_metric::save_results},
            train::ExpConfig,
            GradientForecastModel, LongTermForecastExp,
        },
        Infer,
    },
    models::traits::Forecast,
};
use burn::{
    prelude::*,
    record::{CompactRecorder, Recorder},
    tensor::backend::AutodiffBackend,
};

impl<B: AutodiffBackend> Infer<B> for LongTermForecastExp<B> {
    fn infer(&self, model_config: GradientModelConfig) {
        let record = CompactRecorder::new()
            .load(format!("{0}/model", self.result_path).into(), &self.device)
            .expect("Trained model should exist; run train first");

        let model: GradientForecastModel<B> =
            GradientForecastModel::<B>::new(model_config, self.lengths.clone(), &self.device)
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
            let output =
                model.forecast(batch.x.clone(), batch.x_mark, batch.y.clone(), batch.y_mark);
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
}
