use burn::{prelude::Backend, Tensor};
use plotters::prelude::*;
use std::io::Write;

pub fn plot_single_prediction_in_dir(
    output_dir: &str,
    sample_idx: usize,
    context_series_values: &[f32],
    pred_series_values: &[f32],
    true_series_values: &[f32],
) {
    let file_name = format!("{output_dir}/prediction_{}.png", sample_idx);
    let root = BitMapBackend::new(&file_name, (1024, 768)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut context_series = Vec::new();
    let mut pred_series = Vec::new();
    let mut true_series = Vec::new();

    let mut min_y = f32::MAX;
    let mut max_y = f32::MIN;
    let context_len = context_series_values.len();
    let time_steps = pred_series_values.len();

    for (t, &c_val) in context_series_values.iter().enumerate() {
        context_series.push((t as f32, c_val));
        if c_val < min_y {
            min_y = c_val;
        }
        if c_val > max_y {
            max_y = c_val;
        }
    }

    for (t, (&p_val, &t_val)) in pred_series_values
        .iter()
        .zip(true_series_values.iter())
        .enumerate()
    {
        let x = (context_len + t) as f32;
        pred_series.push((x, p_val));
        true_series.push((x, t_val));

        if p_val < min_y {
            min_y = p_val;
        }
        if p_val > max_y {
            max_y = p_val;
        }
        if t_val < min_y {
            min_y = t_val;
        }
        if t_val > max_y {
            max_y = t_val;
        }
    }

    let y_margin = (max_y - min_y) * 0.1;
    let y_range = (min_y - y_margin)..(max_y + y_margin);
    let total_steps = (context_len + time_steps) as f32;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("Sample {} Prediction vs Ground Truth", sample_idx),
            ("sans-serif", 20).into_font(),
        )
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0f32..total_steps, y_range)
        .unwrap();

    chart.configure_mesh().draw().unwrap();

    chart
        .draw_series(LineSeries::new(context_series, &RED))
        .unwrap()
        .label("Context")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], GREEN));

    chart
        .draw_series(LineSeries::new(pred_series, &BLUE))
        .unwrap()
        .label("Prediction")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

    chart
        .draw_series(LineSeries::new(true_series, &RED))
        .unwrap()
        .label("Ground Truth")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()
        .unwrap();

    root.present().unwrap();
    println!("Saved plot to {}", file_name);
}

pub fn plot_single_prediction(
    exp_root_path: &str,
    sample_idx: usize,
    context_series_values: &[f32],
    pred_series_values: &[f32],
    true_series_values: &[f32],
) {
    plot_single_prediction_in_dir(
        &format!("{exp_root_path}/test"),
        sample_idx,
        context_series_values,
        pred_series_values,
        true_series_values,
    );
}

pub fn save_results<B: Backend>(exp_root_path: &str, error: Tensor<B, 3>, futures: Tensor<B, 3>) {
    // Calculate MSE and MAE per time step (average over Batch and Features)
    // error shape: [Batch, Time, Features]
    // mean_dim(0) -> [1, Time, Features]
    // mean_dim(2) -> [1, Time, 1]
    // squeeze -> [Time]
    let mse_t = error
        .clone()
        .powf_scalar(2.0)
        .mean_dims(&[0, 2])
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let mae_t = error
        .clone()
        .abs()
        .mean_dims(&[0, 2])
        .into_data()
        .to_vec::<f32>()
        .unwrap();

    let mut mse_writer = csv::Writer::from_path(format!("{exp_root_path}/test/mse.csv")).unwrap();
    let mut mae_writer = csv::Writer::from_path(format!("{exp_root_path}/test/mae.csv")).unwrap();

    for val in mse_t.iter() {
        mse_writer.write_record(&[val.to_string()]).unwrap();
    }
    for val in mae_t.iter() {
        mae_writer.write_record(&[val.to_string()]).unwrap();
    }
    mse_writer.flush().unwrap();
    mae_writer.flush().unwrap();

    let all_mse = error
        .clone()
        .powf_scalar(2.0)
        .mean()
        .into_data()
        .into_vec::<f32>()
        .unwrap()[0];
    let all_mae = error
        .clone()
        .abs()
        .mean()
        .into_data()
        .into_vec::<f32>()
        .unwrap()[0];
    let all_rmse = error
        .clone()
        .powf_scalar(2.0)
        .sqrt()
        .mean()
        .into_data()
        .into_vec::<f32>()
        .unwrap()[0];
    let all_mape = (error.clone() / futures.clone())
        .abs()
        .mean()
        .into_data()
        .into_vec::<f32>()
        .unwrap()[0];
    let all_mspe = (error.clone() / futures.clone())
        .powf_scalar(2.0)
        .mean()
        .into_data()
        .into_vec::<f32>()
        .unwrap()[0];
    let mut file = std::fs::File::create(format!("{exp_root_path}/test/results.txt")).unwrap();
    file.write_all(
        format!(
            "MSE: {all_mse}\nMAE: {all_mae}\nRMSE: {all_rmse}\nMAPE: {all_mape}\nMSPE: {all_mspe}"
        )
        .as_bytes(),
    )
    .unwrap();
}

#[cfg(test)]
mod tests {
    use burn::tensor::Distribution;
    use burn_ndarray::NdArray;

    use lib::env_path::get_result_root_path;
    use std::fs;

    use super::*;

    #[test]
    fn test_save_results() {
        type B = NdArray;

        // Create a persistent directory for output
        let exp_root_path = &get_result_root_path();
        let test_path = format!("{exp_root_path}/test_save_results");
        fs::create_dir_all(format!("{test_path}/test")).unwrap();

        // Create dummy data
        // Batch=20, Time=96, Features=2
        let batch_size = 20;
        let time_steps = 96;
        let features = 2;

        let device = Default::default();

        // Generate random data for predicts and futures
        let predicts: Tensor<B, 3> = Tensor::random(
            [batch_size, time_steps, features],
            Distribution::Uniform(0.0, 10.0),
            &device,
        );
        let futures: Tensor<B, 3> = Tensor::random(
            [batch_size, time_steps, features],
            Distribution::Uniform(0.0, 10.0),
            &device,
        );

        let error = predicts.clone() - futures.clone();

        save_results(&test_path, error, futures);

        // Verify files are created
        let test_dir = std::path::Path::new(&test_path).join("test");
        assert!(test_dir.exists());
        assert!(test_dir.join("mse.csv").exists());
        assert!(test_dir.join("mae.csv").exists());
        assert!(test_dir.join("results.txt").exists());

        // Optional: Verify content of results.txt
        let results_content = std::fs::read_to_string(test_dir.join("results.txt")).unwrap();
        assert!(results_content.contains("MSE:"));
        assert!(results_content.contains("MAE:"));
        assert!(results_content.contains("RMSE:"));
        assert!(results_content.contains("MAPE:"));
        assert!(results_content.contains("MSPE:"));

        println!("Test output saved to: {}", test_dir.display());
    }

    #[test]
    fn test_plot_single_prediction() {
        let exp_root_path = &get_result_root_path();
        let test_dir = std::path::Path::new(exp_root_path).join("test_save_results/test");
        fs::create_dir_all(&test_dir).unwrap();

        let sample_idx = 9999;
        let plot_path = test_dir.join(format!("prediction_{}.png", sample_idx));
        if plot_path.exists() {
            fs::remove_file(&plot_path).unwrap();
        }

        let context_series_values = vec![1.0_f32, 1.5, 2.0, 2.5, 3.0];
        let pred_series_values = vec![3.2_f32, 3.4, 3.1, 3.6];
        let true_series_values = vec![3.0_f32, 3.5, 3.0, 3.7];

        plot_single_prediction(
            &format!("{exp_root_path}/test_save_results"),
            sample_idx,
            &context_series_values,
            &pred_series_values,
            &true_series_values,
        );

        assert!(plot_path.exists());
        let metadata = fs::metadata(&plot_path).unwrap();
        assert!(metadata.len() > 0);
    }
}
