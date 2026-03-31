use super::plot_color::feature_color;
use plotters::prelude::*;

pub fn plot_multi_feature_return_map_in_dir(
    output_dir: &str,
    sample_name: &str,
    pred_series_values: &[Vec<f32>],
) {
    if pred_series_values.is_empty() || pred_series_values[0].len() < 2 {
        return;
    }

    let file_name = format!("{output_dir}/{}.png", sample_name);
    let root = BitMapBackend::new(&file_name, (900, 900)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut min_v = f32::MAX;
    let mut max_v = f32::MIN;

    for series in pred_series_values {
        if series.len() < 2 {
            continue;
        }
        for &v in series {
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }
    }

    if min_v == f32::MAX || max_v == f32::MIN {
        return;
    }

    let margin = (max_v - min_v).max(1e-6) * 0.1;
    let axis_min = min_v - margin;
    let axis_max = max_v + margin;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("Sample {} Predicted Return Map", sample_name),
            ("sans-serif", 20).into_font(),
        )
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(axis_min..axis_max, axis_min..axis_max)
        .unwrap();

    chart
        .configure_mesh()
        .x_desc("x(t)")
        .y_desc("x(t+1)")
        .draw()
        .unwrap();

    for (feature_idx, series) in pred_series_values.iter().enumerate() {
        if series.len() < 2 {
            continue;
        }
        let color = feature_color(feature_idx);
        let points = series
            .windows(2)
            .map(|w| (w[0], w[1]))
            .collect::<Vec<(f32, f32)>>();

        chart
            .draw_series(
                points
                    .iter()
                    .map(|&(x, y)| Circle::new((x, y), 2, color.mix(0.8).filled())),
            )
            .unwrap()
            .label(format!("f{}", feature_idx))
            .legend(move |(x, y)| Circle::new((x, y), 4, color.filled()));
    }

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()
        .unwrap();

    root.present().unwrap();
    println!("Saved return-map plot to {}", file_name);
}

pub fn plot_multi_feature_return_map(
    dir_path: &str,
    sample_idx: usize,
    pred_series_values: &[Vec<f32>],
) {
    plot_multi_feature_return_map_in_dir(
        dir_path,
        &format!("return_multi_{}", sample_idx),
        pred_series_values,
    );
}

pub fn plot_single_feature_return_map(dir_path: &str, sample_idx: usize, pred_series: &[f32]) {
    plot_multi_feature_return_map_in_dir(
        dir_path,
        &format!("return_single_{}", sample_idx),
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
    fn test_plot_single_feature_return_map_creates_png() {
        let base_dir = prepare_test_dir("single_return_map");
        let sample_idx = 3;
        let plot_path = base_dir.join(format!("return_single_{}.png", sample_idx));

        let pred = vec![1.0_f32, 1.2, 0.8, 1.4, 1.1, 0.9];
        plot_single_feature_return_map(&base_dir.to_string_lossy(), sample_idx, &pred);

        assert!(plot_path.exists());
        let metadata = fs::metadata(&plot_path).unwrap();
        assert!(metadata.len() > 0);
    }

    #[test]
    fn test_plot_multi_feature_return_map_creates_png() {
        let base_dir = prepare_test_dir("multi_return_map");
        let sample_idx = 4;
        let plot_path = base_dir.join(format!("return_multi_{}.png", sample_idx));

        let pred_series_values = vec![
            vec![1.0_f32, 1.2, 0.8, 1.4, 1.1, 0.9],
            vec![2.0_f32, 1.8, 2.2, 2.1, 1.9, 2.3],
        ];
        plot_multi_feature_return_map(&base_dir.to_string_lossy(), sample_idx, &pred_series_values);

        assert!(plot_path.exists());
        let metadata = fs::metadata(&plot_path).unwrap();
        assert!(metadata.len() > 0);
    }
}
