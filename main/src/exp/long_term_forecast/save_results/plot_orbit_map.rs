#![allow(clippy::too_many_arguments)]

use plotters::prelude::*;

use crate::exp::long_term_forecast::save_results::plot_color::feature_color;

pub fn plot_2d_orbit_in_dir(
    output_dir: &str,
    sample_name: &str,
    context_x_series: &[f32],
    context_y_series: &[f32],
    pred_x_series: &[f32],
    pred_y_series: &[f32],
) {
    let context_steps = context_x_series.len().min(context_y_series.len());
    let pred_steps = pred_x_series.len().min(pred_y_series.len());
    if context_steps < 2 && pred_steps < 2 {
        return;
    }

    let file_name = format!("{output_dir}/{}.png", sample_name);
    let root = BitMapBackend::new(&file_name, (900, 900)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut min_x = f32::MAX;
    let mut max_x = f32::MIN;
    let mut min_y = f32::MAX;
    let mut max_y = f32::MIN;

    let mut context_points = Vec::with_capacity(context_steps);
    for i in 0..context_steps {
        let x = context_x_series[i];
        let y = context_y_series[i];
        context_points.push((x, y));
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }

    let mut pred_points = Vec::with_capacity(pred_steps);
    for i in 0..pred_steps {
        let x = pred_x_series[i];
        let y = pred_y_series[i];
        pred_points.push((x, y));
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }

    let x_margin = (max_x - min_x).max(1e-6) * 0.1;
    let y_margin = (max_y - min_y).max(1e-6) * 0.1;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("Sample {} 2D Orbit", sample_name),
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

    chart
        .configure_mesh()
        .x_desc("feature 0")
        .y_desc("feature 1")
        .draw()
        .unwrap();

    let context_color = feature_color(0);
    if context_steps >= 2 {
        chart
            .draw_series(LineSeries::new(
                context_points.iter().copied(),
                ShapeStyle::from(&context_color).stroke_width(2),
            ))
            .unwrap()
            .label("context")
            .legend(move |(x, y)| {
                PathElement::new(
                    vec![(x, y), (x + 16, y)],
                    ShapeStyle::from(&context_color).stroke_width(2),
                )
            });

        chart
            .draw_series(std::iter::once(Circle::new(
                context_points[0],
                5,
                GREEN.filled(),
            )))
            .unwrap();
    }

    let pred_color = feature_color(1);
    if pred_steps >= 2 {
        chart
            .draw_series(LineSeries::new(
                pred_points.iter().copied(),
                ShapeStyle::from(&pred_color).stroke_width(2),
            ))
            .unwrap()
            .label("prediction")
            .legend(move |(x, y)| {
                PathElement::new(
                    vec![(x, y), (x + 16, y)],
                    ShapeStyle::from(&pred_color).stroke_width(2),
                )
            });

        chart
            .draw_series(std::iter::once(Circle::new(
                pred_points[0],
                5,
                RED.filled(),
            )))
            .unwrap();
    }

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()
        .unwrap();

    root.present().unwrap();
}

#[allow(clippy::too_many_arguments)]
pub fn plot_3d_orbit_in_dir(
    output_dir: &str,
    sample_name: &str,
    context_x_series: &[f32],
    context_y_series: &[f32],
    context_z_series: &[f32],
    pred_x_series: &[f32],
    pred_y_series: &[f32],
    pred_z_series: &[f32],
) {
    let context_steps = context_x_series
        .len()
        .min(context_y_series.len())
        .min(context_z_series.len());
    let pred_steps = pred_x_series
        .len()
        .min(pred_y_series.len())
        .min(pred_z_series.len());
    if context_steps < 2 && pred_steps < 2 {
        return;
    }

    let file_name = format!("{output_dir}/{}.png", sample_name);
    let root = BitMapBackend::new(&file_name, (900, 900)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut min_x = f32::MAX;
    let mut max_x = f32::MIN;
    let mut min_y = f32::MAX;
    let mut max_y = f32::MIN;
    let mut min_z = f32::MAX;
    let mut max_z = f32::MIN;

    for i in 0..context_steps {
        min_x = min_x.min(context_x_series[i]);
        max_x = max_x.max(context_x_series[i]);
        min_y = min_y.min(context_y_series[i]);
        max_y = max_y.max(context_y_series[i]);
        min_z = min_z.min(context_z_series[i]);
        max_z = max_z.max(context_z_series[i]);
    }
    for i in 0..pred_steps {
        min_x = min_x.min(pred_x_series[i]);
        max_x = max_x.max(pred_x_series[i]);
        min_y = min_y.min(pred_y_series[i]);
        max_y = max_y.max(pred_y_series[i]);
        min_z = min_z.min(pred_z_series[i]);
        max_z = max_z.max(pred_z_series[i]);
    }

    let x_margin = (max_x - min_x).max(1e-6) * 0.1;
    let y_margin = (max_y - min_y).max(1e-6) * 0.1;
    let z_margin = (max_z - min_z).max(1e-6) * 0.1;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("Sample {} 3D Orbit", sample_name),
            ("sans-serif", 20).into_font(),
        )
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_3d(
            (min_x - x_margin)..(max_x + x_margin),
            (min_y - y_margin)..(max_y + y_margin),
            (min_z - z_margin)..(max_z + z_margin),
        )
        .unwrap();

    chart.with_projection(|mut p| {
        p.yaw = 0.55;
        p.pitch = 0.45;
        p.scale = 0.9;
        p.into_matrix()
    });

    chart.configure_axes().draw().unwrap();

    let context_color = feature_color(0);
    if context_steps >= 2 {
        chart
            .draw_series(LineSeries::new(
                (0..context_steps).map(|i| {
                    (
                        context_x_series[i],
                        context_y_series[i],
                        context_z_series[i],
                    )
                }),
                ShapeStyle::from(&context_color).stroke_width(2),
            ))
            .unwrap();
    }

    let pred_color = feature_color(1);
    if pred_steps >= 2 {
        chart
            .draw_series(LineSeries::new(
                (0..pred_steps).map(|i| (pred_x_series[i], pred_y_series[i], pred_z_series[i])),
                ShapeStyle::from(&pred_color).stroke_width(2),
            ))
            .unwrap();
    }

    root.present().unwrap();
}

pub fn plot_orbit_maps(
    dir_path: &str,
    sample_idx: usize,
    context_series_values: &[Vec<f32>],
    pred_series_values: &[Vec<f32>],
) {
    if context_series_values.len() >= 2 && pred_series_values.len() >= 2 {
        plot_2d_orbit_in_dir(
            dir_path,
            &format!("orbit2d_{}", sample_idx),
            &context_series_values[0],
            &context_series_values[1],
            &pred_series_values[0],
            &pred_series_values[1],
        );
    }

    if context_series_values.len() >= 3 && pred_series_values.len() >= 3 {
        plot_3d_orbit_in_dir(
            dir_path,
            &format!("orbit3d_{}", sample_idx),
            &context_series_values[0],
            &context_series_values[1],
            &context_series_values[2],
            &pred_series_values[0],
            &pred_series_values[1],
            &pred_series_values[2],
        );
    }
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
    fn test_plot_2d_orbit_creates_png() {
        let base_dir = prepare_test_dir("orbit2d_plot");
        let plot_path = base_dir.join("orbit2d_2.png");

        let cx = vec![0.1_f32, 0.3, 0.2, 0.4, 0.7, 0.6, 0.8];
        let cy = vec![1.0_f32, 0.8, 0.9, 0.7, 0.5, 0.6, 0.4];
        let px = vec![0.2_f32, 0.35, 0.25, 0.5, 0.75, 0.65, 0.85];
        let py = vec![0.9_f32, 0.75, 0.85, 0.65, 0.45, 0.55, 0.35];
        plot_2d_orbit_in_dir(&base_dir.to_string_lossy(), "orbit2d_2", &cx, &cy, &px, &py);

        assert!(plot_path.exists());
        let metadata = fs::metadata(&plot_path).unwrap();
        assert!(metadata.len() > 0);
    }

    #[test]
    fn test_plot_3d_orbit_creates_png() {
        let base_dir = prepare_test_dir("orbit3d_plot");
        let plot_path = base_dir.join("orbit3d_3.png");

        let cx = vec![0.1_f32, 0.3, 0.2, 0.4, 0.7, 0.6, 0.8];
        let cy = vec![1.0_f32, 0.8, 0.9, 0.7, 0.5, 0.6, 0.4];
        let cz = vec![0.2_f32, 0.4, 0.5, 0.3, 0.6, 0.9, 0.7];
        let px = vec![0.2_f32, 0.35, 0.25, 0.45, 0.75, 0.7, 0.9];
        let py = vec![0.95_f32, 0.75, 0.85, 0.68, 0.48, 0.58, 0.38];
        let pz = vec![0.3_f32, 0.45, 0.55, 0.35, 0.7, 0.95, 0.75];
        plot_3d_orbit_in_dir(
            &base_dir.to_string_lossy(),
            "orbit3d_3",
            &cx,
            &cy,
            &cz,
            &px,
            &py,
            &pz,
        );

        assert!(plot_path.exists());
        let metadata = fs::metadata(&plot_path).unwrap();
        assert!(metadata.len() > 0);
    }
}
