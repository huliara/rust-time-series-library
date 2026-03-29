use std::{fs, io::Write};

use crate::{
    args::{data::DataCommand, model::ModelConfig, time_lengths::TimeLengths},
    data::{data_loader::create_data_loader, dataset::time_series_dataset::ExpFlag},
    exp::{
        long_term_forecast::{
            save_results::sample_plots::sample_plots, train::ExpConfig, ForecastModel,
        },
        Infer,
    },
    models::traits::Forecast,
};
use burn::{
    prelude::{s, *},
    record::{CompactRecorder, Recorder},
    tensor::backend::AutodiffBackend,
};

fn write_infer_metrics(
    exp_root_path: &str,
    mse_t: &[f64],
    mae_t: &[f64],
    metrics: AggregateMetrics,
) {
    let mut mse_writer = csv::Writer::from_path(format!("{exp_root_path}/test/mse.csv")).unwrap();
    let mut mae_writer = csv::Writer::from_path(format!("{exp_root_path}/test/mae.csv")).unwrap();

    for val in mse_t {
        mse_writer.write_record(&[val.to_string()]).unwrap();
    }
    for val in mae_t {
        mae_writer.write_record(&[val.to_string()]).unwrap();
    }
    mse_writer.flush().unwrap();
    mae_writer.flush().unwrap();

    let mut file = std::fs::File::create(format!("{exp_root_path}/test/results.txt")).unwrap();
    file.write_all(
        format!(
            "MSE: {}\nMAE: {}\nRMSE: {}\nMAPE: {}\nMSPE: {}",
            metrics.all_mse, metrics.all_mae, metrics.all_rmse, metrics.all_mape, metrics.all_mspe
        )
        .as_bytes(),
    )
    .unwrap();
}

#[derive(Clone, Copy)]
struct AggregateMetrics {
    all_mse: f64,
    all_mae: f64,
    all_rmse: f64,
    all_mape: f64,
    all_mspe: f64,
}

impl<B: AutodiffBackend> Infer<B> for ForecastModel<B> {
    #[allow(clippy::too_many_arguments)]
    fn infer(
        &self,
        exp_root_path: &str,
        exp_config: ExpConfig,
        model_config: ModelConfig,
        lengths: TimeLengths,
        data_config: DataCommand,
        device: B::Device,
    ) {
        let record = CompactRecorder::new()
            .load(format!("{exp_root_path}/model").into(), &device)
            .expect("Trained model should exist; run train first");

        let model: ForecastModel<B> =
            ForecastModel::<B>::new(model_config, lengths.clone(), &device).load_record(record);

        let dataloader_test = create_data_loader(
            &data_config,
            &lengths,
            exp_config.batch_size,
            exp_config.num_workers,
            exp_config.seed,
            ExpFlag::Test,
        );
        let plot_num = 16usize;
        let mut contexts_for_plot: Vec<Tensor<B, 3>> = Vec::with_capacity(plot_num);
        let mut predicts_for_plot: Vec<Tensor<B, 3>> = Vec::with_capacity(plot_num);
        let mut futures_for_plot: Vec<Tensor<B, 3>> = Vec::with_capacity(plot_num);

        let mut mse_t_sum: Vec<f64> = Vec::new();
        let mut mae_t_sum: Vec<f64> = Vec::new();
        let mut time_weight_sum = 0.0f64;

        let mut total_samples = 0.0f64;
        let mut all_mse_weighted_sum = 0.0f64;
        let mut all_mae_weighted_sum = 0.0f64;
        let mut all_rmse_weighted_sum = 0.0f64;
        let mut all_mape_weighted_sum = 0.0f64;
        let mut all_mspe_weighted_sum = 0.0f64;

        let test_dir = format!("{exp_root_path}/test");
        fs::create_dir_all(&test_dir).unwrap();

        for batch in dataloader_test.iter() {
            let output =
                model.forecast(batch.x.clone(), batch.x_mark, batch.y.clone(), batch.y_mark);

            let error = output.clone() - batch.y.clone();
            let [batch_size, _, feature_size] = error.dims();
            let sample_weight = batch_size as f64;
            let time_weight = (batch_size * feature_size) as f64;

            let batch_mse_t = error
                .clone()
                .powf_scalar(2.0)
                .mean_dims(&[0, 2])
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            let batch_mae_t = error
                .clone()
                .abs()
                .mean_dims(&[0, 2])
                .into_data()
                .to_vec::<f32>()
                .unwrap();

            if mse_t_sum.is_empty() {
                mse_t_sum = vec![0.0; batch_mse_t.len()];
                mae_t_sum = vec![0.0; batch_mae_t.len()];
            }

            for (i, v) in batch_mse_t.iter().enumerate() {
                mse_t_sum[i] += (*v as f64) * time_weight;
            }
            for (i, v) in batch_mae_t.iter().enumerate() {
                mae_t_sum[i] += (*v as f64) * time_weight;
            }
            time_weight_sum += time_weight;

            all_mse_weighted_sum += error
                .clone()
                .powf_scalar(2.0)
                .mean()
                .into_data()
                .into_vec::<f32>()
                .unwrap()[0] as f64
                * sample_weight;
            all_mae_weighted_sum += error
                .clone()
                .abs()
                .mean()
                .into_data()
                .into_vec::<f32>()
                .unwrap()[0] as f64
                * sample_weight;
            all_rmse_weighted_sum += error
                .clone()
                .powf_scalar(2.0)
                .sqrt()
                .mean()
                .into_data()
                .into_vec::<f32>()
                .unwrap()[0] as f64
                * sample_weight;
            all_mape_weighted_sum += (error.clone() / batch.y.clone())
                .abs()
                .mean()
                .into_data()
                .into_vec::<f32>()
                .unwrap()[0] as f64
                * sample_weight;
            all_mspe_weighted_sum += (error.clone() / batch.y.clone())
                .powf_scalar(2.0)
                .mean()
                .into_data()
                .into_vec::<f32>()
                .unwrap()[0] as f64
                * sample_weight;
            total_samples += sample_weight;

            let remain = plot_num.saturating_sub(contexts_for_plot.len());
            if remain > 0 {
                let take = remain.min(batch_size);
                for i in 0..take {
                    contexts_for_plot.push(batch.x.clone().slice(s![i..i + 1, .., ..]));
                    predicts_for_plot.push(output.clone().slice(s![i..i + 1, .., ..]));
                    futures_for_plot.push(batch.y.clone().slice(s![i..i + 1, .., ..]));
                }
            }
        }

        if !contexts_for_plot.is_empty() {
            let contexts = Tensor::cat(contexts_for_plot, 0);
            let predicts = Tensor::cat(predicts_for_plot, 0);
            let futures = Tensor::cat(futures_for_plot, 0);
            sample_plots(contexts, predicts, futures, plot_num, &test_dir);
        }

        if total_samples > 0.0 && time_weight_sum > 0.0 {
            let mse_t: Vec<f64> = mse_t_sum.iter().map(|v| *v / time_weight_sum).collect();
            let mae_t: Vec<f64> = mae_t_sum.iter().map(|v| *v / time_weight_sum).collect();

            write_infer_metrics(
                exp_root_path,
                &mse_t,
                &mae_t,
                AggregateMetrics {
                    all_mse: all_mse_weighted_sum / total_samples,
                    all_mae: all_mae_weighted_sum / total_samples,
                    all_rmse: all_rmse_weighted_sum / total_samples,
                    all_mape: all_mape_weighted_sum / total_samples,
                    all_mspe: all_mspe_weighted_sum / total_samples,
                },
            );
        }
    }
}
