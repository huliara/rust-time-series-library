use burn::{prelude::Backend, Tensor};
use plotters::element::DashedPathElement;
use plotters::prelude::*;
use std::io::Write;

fn feature_color(index: usize) -> RGBColor {
    const COLORS: [RGBColor; 10] = [
        RGBColor(31, 119, 180),
        RGBColor(255, 127, 14),
        RGBColor(44, 160, 44),
        RGBColor(214, 39, 40),
        RGBColor(148, 103, 189),
        RGBColor(140, 86, 75),
        RGBColor(227, 119, 194),
        RGBColor(127, 127, 127),
        RGBColor(188, 189, 34),
        RGBColor(23, 190, 207),
    ];
    COLORS[index % COLORS.len()]
}

pub fn plot_multi_feature_prediction_in_dir(
    output_dir: &str,
    sample_name: &str,
    context_series_values: &[Vec<f32>],
    pred_series_values: &[Vec<f32>],
    true_series_values: &[Vec<f32>],
) {
    let feature_count = context_series_values.len();
    if feature_count == 0
        || pred_series_values.len() != feature_count
        || true_series_values.len() != feature_count
    {
        return;
    }

    let file_name = format!("{output_dir}/{}.png", sample_name);
    let root = BitMapBackend::new(&file_name, (1280, 900)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let context_len = context_series_values[0].len();
    let pred_len = pred_series_values[0].len();
    let total_steps = (context_len + pred_len) as f32;

    let mut min_y = f32::MAX;
    let mut max_y = f32::MIN;
    for feature_idx in 0..feature_count {
        for &v in &context_series_values[feature_idx] {
            min_y = min_y.min(v);
            max_y = max_y.max(v);
        }
        for &v in &pred_series_values[feature_idx] {
            min_y = min_y.min(v);
            max_y = max_y.max(v);
        }
        for &v in &true_series_values[feature_idx] {
            min_y = min_y.min(v);
            max_y = max_y.max(v);
        }
    }

    let y_margin = (max_y - min_y) * 0.1;
    let y_range = (min_y - y_margin)..(max_y + y_margin);
    let y_min = min_y - y_margin;
    let y_max = max_y + y_margin;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("Sample {} Multi-Feature Prediction", sample_name),
            ("sans-serif", 20).into_font(),
        )
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(45)
        .build_cartesian_2d(0f32..total_steps, y_range)
        .unwrap();

    chart.configure_mesh().draw().unwrap();

    let boundary_x = context_len as f32;
    chart
        .draw_series(DashedLineSeries::new(
            vec![(boundary_x, y_min), (boundary_x, y_max)],
            8,
            6,
            ShapeStyle::from(&BLACK.mix(0.5)).stroke_width(2),
        ))
        .unwrap()
        .label("context/predict boundary")
        .legend(|(x, y)| {
            DashedPathElement::new(
                vec![(x, y), (x + 20, y)],
                8,
                6,
                ShapeStyle::from(&BLACK.mix(0.5)).stroke_width(2),
            )
        });

    for feature_idx in 0..feature_count {
        let color = feature_color(feature_idx);

        let context_series = context_series_values[feature_idx]
            .iter()
            .enumerate()
            .map(|(t, v)| (t as f32, *v))
            .collect::<Vec<_>>();

        let pred_series = pred_series_values[feature_idx]
            .iter()
            .enumerate()
            .map(|(t, v)| ((context_len + t) as f32, *v))
            .collect::<Vec<_>>();

        let true_series = true_series_values[feature_idx]
            .iter()
            .enumerate()
            .map(|(t, v)| ((context_len + t) as f32, *v))
            .collect::<Vec<_>>();

        chart
            .draw_series(LineSeries::new(
                context_series,
                ShapeStyle::from(&color).stroke_width(1),
            ))
            .unwrap()
            .label(format!("f{} context", feature_idx))
            .legend(move |(x, y)| {
                PathElement::new(
                    vec![(x, y), (x + 20, y)],
                    ShapeStyle::from(&color).stroke_width(1),
                )
            });

        chart
            .draw_series(LineSeries::new(
                pred_series,
                ShapeStyle::from(&color).stroke_width(3),
            ))
            .unwrap()
            .label(format!("f{} pred", feature_idx))
            .legend(move |(x, y)| {
                PathElement::new(
                    vec![(x, y), (x + 20, y)],
                    ShapeStyle::from(&color).stroke_width(3),
                )
            });

        chart
            .draw_series(DashedLineSeries::new(
                true_series,
                8,
                6,
                ShapeStyle::from(&color).stroke_width(2),
            ))
            .unwrap()
            .label(format!("f{} true", feature_idx))
            .legend(move |(x, y)| {
                DashedPathElement::new(
                    vec![(x, y), (x + 20, y)],
                    8,
                    6,
                    ShapeStyle::from(&color).stroke_width(2),
                )
            });
    }

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()
        .unwrap();

    root.present().unwrap();
    println!("Saved plot to {}", file_name);
}

pub fn plot_multi_feature_prediction(
    dir_path: &str,
    sample_idx: usize,
    context_series_values: &[Vec<f32>],
    pred_series_values: &[Vec<f32>],
    true_series_values: &[Vec<f32>],
) {
    plot_multi_feature_prediction_in_dir(
        dir_path,
        &format!("multi_{}", sample_idx),
        context_series_values,
        pred_series_values,
        true_series_values,
    );
}

pub fn plot_single_feature_prediction(
    dir_path: &str,
    sample_idx: usize,
    context_series: &[f32],
    pred_series: &[f32],
    true_series: &[f32],
) {
    plot_multi_feature_prediction_in_dir(
        dir_path,
        &format!("single_{}", sample_idx),
        &[context_series.to_vec()],
        &[pred_series.to_vec()],
        &[true_series.to_vec()],
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
    fn test_plot_multi_feature_prediction() {
        let exp_root_path = &get_result_root_path();
        let test_dir = std::path::Path::new(exp_root_path).join("test_save_results/test");
        fs::create_dir_all(&test_dir).unwrap();

        let sample_idx = 10000;
        let plot_path = test_dir.join(format!("prediction_multi_{}.png", sample_idx));
        if plot_path.exists() {
            fs::remove_file(&plot_path).unwrap();
        }

        let context_series_values = vec![
            vec![1.0_f32, 1.5, 2.0, 2.5, 3.0],
            vec![2.0_f32, 2.2, 2.4, 2.6, 2.8],
        ];
        let pred_series_values = vec![vec![3.2_f32, 3.4, 3.1, 3.6], vec![3.0_f32, 3.1, 3.2, 3.3]];
        let true_series_values = vec![vec![3.0_f32, 3.5, 3.0, 3.7], vec![2.9_f32, 3.0, 3.1, 3.4]];

        plot_multi_feature_prediction(
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
