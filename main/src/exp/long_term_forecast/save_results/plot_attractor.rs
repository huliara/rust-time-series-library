use super::plot_color::feature_color;
use plotters::prelude::*;
pub fn plot_multi_feature_attractor_in_dir(
    output_dir: &str,
    sample_name: &str,
    pred_series_values: &[Vec<f32>],
) {
    if pred_series_values.is_empty() {
        return;
    }

    let file_name = format!("{output_dir}/{}.png", sample_name);
    let root = BitMapBackend::new(&file_name, (900, 900)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut min_x = f32::MAX;
    let mut max_x = f32::MIN;
    let mut min_y = f32::MAX;
    let mut max_y = f32::MIN;

    let mut series_points: Vec<Vec<(f32, f32)>> = Vec::new();

    if pred_series_values.len() >= 2 {
        let point_count = pred_series_values[0].len().min(pred_series_values[1].len());
        let x = &pred_series_values[0];
        let y = &pred_series_values[1];

        let mut points = Vec::with_capacity(point_count);
        for i in 0..point_count {
            points.push((x[i], y[i]));
            min_x = min_x.min(x[i]);
            max_x = max_x.max(x[i]);
            min_y = min_y.min(y[i]);
            max_y = max_y.max(y[i]);
        }
        series_points.push(points);
    } else {
        for series in pred_series_values {
            if series.len() < 3 {
                continue;
            }
            let lag = 2usize;
            let mut points = Vec::with_capacity(series.len() - lag);
            for i in 0..(series.len() - lag) {
                let x = series[i];
                let y = series[i + lag];
                points.push((x, y));
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
            }
            series_points.push(points);
        }
    }

    if series_points.is_empty() {
        return;
    }

    let x_margin = (max_x - min_x).max(1e-6) * 0.1;
    let y_margin = (max_y - min_y).max(1e-6) * 0.1;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("Sample {} Predicted Attractor", sample_name),
            ("sans-serif", 20).into_font(),
        )
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(
            (min_x - x_margin)..(max_x + x_margin),
            (min_y - y_margin)..(max_y + y_margin),
        )
        .unwrap();

    if pred_series_values.len() >= 2 {
        chart
            .configure_mesh()
            .x_desc("x(t) [feature 0]")
            .y_desc("y(t) [feature 1]")
            .draw()
            .unwrap();
    } else {
        chart
            .configure_mesh()
            .x_desc("x(t)")
            .y_desc("x(t+2)")
            .draw()
            .unwrap();
    }

    for (idx, points) in series_points.iter().enumerate() {
        let color = feature_color(idx);
        chart
            .draw_series(
                points
                    .iter()
                    .map(|&(x, y)| Circle::new((x, y), 2, color.mix(0.8).filled())),
            )
            .unwrap()
            .label(if pred_series_values.len() >= 2 {
                "f0-f1 projection".to_string()
            } else {
                format!("f{} delay-embed", idx)
            })
            .legend(move |(x, y)| Circle::new((x, y), 4, color.filled()));
    }

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()
        .unwrap();

    root.present().unwrap();
    println!("Saved attractor plot to {}", file_name);
}

pub fn plot_multi_feature_attractor(
    dir_path: &str,
    sample_idx: usize,
    pred_series_values: &[Vec<f32>],
) {
    plot_multi_feature_attractor_in_dir(
        dir_path,
        &format!("attractor_multi_{}", sample_idx),
        pred_series_values,
    );
}

pub fn plot_single_feature_attractor(dir_path: &str, sample_idx: usize, pred_series: &[f32]) {
    plot_multi_feature_attractor_in_dir(
        dir_path,
        &format!("attractor_single_{}", sample_idx),
        &[pred_series.to_vec()],
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
    fn test_plot_single_feature_attractor_creates_png() {
        let base_dir = prepare_test_dir("single_attractor");
        let sample_idx = 5;
        let plot_path = base_dir.join(format!("attractor_single_{}.png", sample_idx));

        let pred = vec![1.0_f32, 1.2, 0.8, 1.4, 1.1, 0.9, 1.3];
        plot_single_feature_attractor(&base_dir.to_string_lossy(), sample_idx, &pred);

        assert!(plot_path.exists());
        let metadata = fs::metadata(&plot_path).unwrap();
        assert!(metadata.len() > 0);
    }

    #[test]
    fn test_plot_multi_feature_attractor_creates_png() {
        let base_dir = prepare_test_dir("multi_attractor");
        let sample_idx = 6;
        let plot_path = base_dir.join(format!("attractor_multi_{}.png", sample_idx));

        let pred_series_values = vec![
            vec![1.0_f32, 1.2, 0.8, 1.4, 1.1, 0.9, 1.3],
            vec![2.0_f32, 1.8, 2.2, 2.1, 1.9, 2.3, 2.0],
        ];
        plot_multi_feature_attractor(&base_dir.to_string_lossy(), sample_idx, &pred_series_values);

        assert!(plot_path.exists());
        let metadata = fs::metadata(&plot_path).unwrap();
        assert!(metadata.len() > 0);
    }
}
