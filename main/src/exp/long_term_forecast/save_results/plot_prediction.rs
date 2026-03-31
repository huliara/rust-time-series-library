use super::plot_color::feature_color;
use plotters::element::DashedPathElement;
use plotters::prelude::*;

pub fn plot_multi_feature_prediction_in_dir(
    output_dir: &str,
    sample_name: &str,
    context_series_values: &[Vec<f32>],
    pred_series_values: &[Vec<f32>],
    true_series_values: &[Vec<f32>],
) {
    let feature_count = context_series_values.len();
    let target_feature_count = pred_series_values.len();
    if feature_count == 0 || pred_series_values.len() != true_series_values.len() {
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
    for series in context_series_values.iter() {
        for &v in series.iter() {
            min_y = min_y.min(v);
            max_y = max_y.max(v);
        }
    }

    for feature_idx in 0..target_feature_count {
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

    for (feature_idx, item) in context_series_values.iter().enumerate().take(feature_count) {
        let color = feature_color(feature_idx);

        let context_series = item
            .iter()
            .enumerate()
            .map(|(t, v)| (t as f32, *v))
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
    }

    for feature_idx in 0..target_feature_count {
        let color = feature_color(feature_idx);

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
                pred_series,
                ShapeStyle::from(&color).stroke_width(2),
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

#[cfg(test)]
mod tests {
    use lib::env_path::get_result_root_path;
    use std::{fs, path::PathBuf};

    use super::*;

    fn prepare_test_dir(name: &str) -> PathBuf {
        let root = PathBuf::from(get_result_root_path());
        let dir = root.join("test_save_results").join(name);
        if dir.exists() {
            fs::remove_dir_all(&dir).unwrap();
        }
        fs::create_dir_all(&dir).unwrap();
        dir
    }
    #[test]
    fn test_plot_single_feature_prediction_creates_png() {
        let base_dir = prepare_test_dir("single_plot");
        let sample_idx = 7;
        let plot_path = base_dir.join(format!("single_{}.png", sample_idx));

        let context = vec![1.0_f32, 1.5, 2.0, 2.5, 3.0];
        let pred = vec![3.2_f32, 3.4, 3.1, 3.6];
        let truth = vec![3.0_f32, 3.5, 3.0, 3.7];

        plot_single_feature_prediction(
            &base_dir.to_string_lossy(),
            sample_idx,
            &context,
            &pred,
            &truth,
        );

        assert!(plot_path.exists());
        let metadata = fs::metadata(&plot_path).unwrap();
        assert!(metadata.len() > 0);
    }

    #[test]
    fn test_plot_multi_feature_prediction_creates_png() {
        let base_dir = prepare_test_dir("multi_plot");
        let sample_idx = 11;
        let plot_path = base_dir.join(format!("multi_{}.png", sample_idx));

        let context_series_values = vec![
            vec![1.0_f32, 1.5, 2.0, 2.5, 3.0],
            vec![2.0_f32, 2.2, 2.4, 2.6, 2.8],
        ];
        let pred_series_values = vec![vec![3.2_f32, 3.4, 3.1, 3.6], vec![3.0_f32, 3.1, 3.2, 3.3]];
        let true_series_values = vec![vec![3.0_f32, 3.5, 3.0, 3.7], vec![2.9_f32, 3.0, 3.1, 3.4]];

        plot_multi_feature_prediction(
            &base_dir.to_string_lossy(),
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
