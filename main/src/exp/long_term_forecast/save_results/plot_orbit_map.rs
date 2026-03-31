use plotters::prelude::*;

use crate::exp::long_term_forecast::save_results::plot_color::feature_color;

pub fn plot_2d_orbit_in_dir(
    output_dir: &str,
    sample_name: &str,
    x_series: &[f32],
    y_series: &[f32],
) {
    let steps = x_series.len().min(y_series.len());
    if steps < 2 {
        return;
    }

    let file_name = format!("{output_dir}/{}.png", sample_name);
    let root = BitMapBackend::new(&file_name, (900, 900)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut min_x = f32::MAX;
    let mut max_x = f32::MIN;
    let mut min_y = f32::MAX;
    let mut max_y = f32::MIN;

    let mut points = Vec::with_capacity(steps);
    for i in 0..steps {
        let x = x_series[i];
        let y = y_series[i];
        points.push((x, y));
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

    let color = feature_color(0);
    chart
        .draw_series(LineSeries::new(
            points.iter().copied(),
            ShapeStyle::from(&color).stroke_width(2),
        ))
        .unwrap();

    chart
        .draw_series(points.iter().map(|&(x, y)| Circle::new((x, y), 2, color.mix(0.7).filled())))
        .unwrap();

    chart
        .draw_series(std::iter::once(Circle::new(points[0], 5, GREEN.filled())))
        .unwrap()
        .label("start")
        .legend(|(x, y)| Circle::new((x, y), 4, GREEN.filled()));

    chart
        .draw_series(std::iter::once(Circle::new(points[steps - 1], 5, RED.filled())))
        .unwrap()
        .label("end")
        .legend(|(x, y)| Circle::new((x, y), 4, RED.filled()));

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()
        .unwrap();

    root.present().unwrap();
    println!("Saved 2D orbit plot to {}", file_name);
}

fn project_3d_to_2d(x: f32, y: f32, z: f32) -> (f32, f32) {
    // Simple oblique projection for visualizing 3D trajectory on 2D canvas.
    let px = x - 0.45 * y;
    let py = z + 0.35 * y;
    (px, py)
}

pub fn plot_3d_orbit_in_dir(
    output_dir: &str,
    sample_name: &str,
    x_series: &[f32],
    y_series: &[f32],
    z_series: &[f32],
) {
    let steps = x_series.len().min(y_series.len()).min(z_series.len());
    if steps < 2 {
        return;
    }

    let file_name = format!("{output_dir}/{}.png", sample_name);
    let root = BitMapBackend::new(&file_name, (900, 900)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut proj_points = Vec::with_capacity(steps);
    let mut min_x = f32::MAX;
    let mut max_x = f32::MIN;
    let mut min_y = f32::MAX;
    let mut max_y = f32::MIN;

    for i in 0..steps {
        let p = project_3d_to_2d(x_series[i], y_series[i], z_series[i]);
        min_x = min_x.min(p.0);
        max_x = max_x.max(p.0);
        min_y = min_y.min(p.1);
        max_y = max_y.max(p.1);
        proj_points.push(p);
    }

    let x_margin = (max_x - min_x).max(1e-6) * 0.1;
    let y_margin = (max_y - min_y).max(1e-6) * 0.1;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("Sample {} 3D Orbit (projected)", sample_name),
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
        .x_desc("projected x")
        .y_desc("projected y")
        .draw()
        .unwrap();

    let line_color = feature_color(1);
    chart
        .draw_series(LineSeries::new(
            proj_points.iter().copied(),
            ShapeStyle::from(&line_color).stroke_width(2),
        ))
        .unwrap();

    chart
        .draw_series(
            proj_points
                .iter()
                .map(|&(x, y)| Circle::new((x, y), 2, line_color.mix(0.7).filled())),
        )
        .unwrap();

    chart
        .draw_series(std::iter::once(Circle::new(proj_points[0], 5, GREEN.filled())))
        .unwrap()
        .label("start")
        .legend(|(x, y)| Circle::new((x, y), 4, GREEN.filled()));

    chart
        .draw_series(std::iter::once(Circle::new(proj_points[steps - 1], 5, RED.filled())))
        .unwrap()
        .label("end")
        .legend(|(x, y)| Circle::new((x, y), 4, RED.filled()));

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()
        .unwrap();

    root.present().unwrap();
    println!("Saved 3D orbit plot to {}", file_name);
}

pub fn plot_orbit_maps(dir_path: &str, sample_idx: usize, pred_series_values: &[Vec<f32>]) {
    if pred_series_values.len() >= 2 {
        plot_2d_orbit_in_dir(
            dir_path,
            &format!("orbit2d_{}", sample_idx),
            &pred_series_values[0],
            &pred_series_values[1],
        );
    }

    if pred_series_values.len() >= 3 {
        plot_3d_orbit_in_dir(
            dir_path,
            &format!("orbit3d_{}", sample_idx),
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

        let x = vec![0.1_f32, 0.3, 0.2, 0.4, 0.7, 0.6, 0.8];
        let y = vec![1.0_f32, 0.8, 0.9, 0.7, 0.5, 0.6, 0.4];
        plot_2d_orbit_in_dir(&base_dir.to_string_lossy(), "orbit2d_2", &x, &y);

        assert!(plot_path.exists());
        let metadata = fs::metadata(&plot_path).unwrap();
        assert!(metadata.len() > 0);
    }

    #[test]
    fn test_plot_3d_orbit_creates_png() {
        let base_dir = prepare_test_dir("orbit3d_plot");
        let plot_path = base_dir.join("orbit3d_3.png");

        let x = vec![0.1_f32, 0.3, 0.2, 0.4, 0.7, 0.6, 0.8];
        let y = vec![1.0_f32, 0.8, 0.9, 0.7, 0.5, 0.6, 0.4];
        let z = vec![0.2_f32, 0.4, 0.5, 0.3, 0.6, 0.9, 0.7];
        plot_3d_orbit_in_dir(&base_dir.to_string_lossy(), "orbit3d_3", &x, &y, &z);

        assert!(plot_path.exists());
        let metadata = fs::metadata(&plot_path).unwrap();
        assert!(metadata.len() > 0);
    }
}
