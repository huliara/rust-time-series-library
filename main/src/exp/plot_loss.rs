use plotters::prelude::*;
use std::fs;
use std::path::Path;

/// Read loss values from Loss.log files in a directory
fn read_loss_from_file(loss_file_path: &Path) -> Option<f32> {
    match fs::read_to_string(loss_file_path) {
        Ok(content) => {
            let mut total_loss = 0.0f32;
            let mut count = 0i32;

            for line in content.lines().skip(1) {
                if let Ok(loss) = line.split(',').next().unwrap_or("").parse::<f32>() {
                    total_loss += loss;
                    count += 1;
                }
            }

            if count > 0 {
                Some(total_loss / count as f32)
            } else {
                None
            }
        }
        Err(_) => None,
    }
}

fn read_losses_for_phase(phase_dir: &Path) -> Option<Vec<f32>> {
    if !phase_dir.exists() {
        return None;
    }

    let mut losses = Vec::new();
    let mut epoch_dirs: Vec<_> = fs::read_dir(phase_dir)
        .ok()?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_name().to_string_lossy().starts_with("epoch-"))
        .collect();

    // Sort by epoch number
    epoch_dirs.sort_by_key(|entry| {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        name_str
            .strip_prefix("epoch-")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0)
    });

    for entry in epoch_dirs {
        let loss_file = entry.path().join("Loss.log");
        if let Some(loss) = read_loss_from_file(&loss_file) {
            losses.push(loss);
        }
    }

    if losses.is_empty() {
        None
    } else {
        Some(losses)
    }
}

pub fn plot_loss_for_experiment(exp_dir: &str) -> Result<(), Box<dyn std::error::Error>> {
    let exp_path = Path::new(exp_dir);
    let train_dir = exp_path.join("train");
    let valid_dir = exp_path.join("valid");

    let train_losses = read_losses_for_phase(&train_dir);
    let valid_losses = read_losses_for_phase(&valid_dir);

    match (train_losses, valid_losses) {
        (Some(train), Some(valid)) => {
            let output_path = exp_path.join("loss_curve.png");

            create_loss_plot(&train, &valid, output_path.to_str().unwrap())?;
            println!("Plot saved to {}", output_path.display());
            Ok(())
        }
        _ => Err("Could not read train or valid loss files".into()),
    }
}

fn create_loss_plot(
    train_losses: &[f32],
    valid_losses: &[f32],
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let _file = std::fs::File::create(output_path)?;
    let root = BitMapBackend::new(output_path, (1280, 720)).into_drawing_area();
    root.fill(&WHITE)?;

    // Calculate min and max for y-axis
    let min_loss = train_losses
        .iter()
        .chain(valid_losses.iter())
        .copied()
        .fold(f32::INFINITY, f32::min);
    let max_loss = train_losses
        .iter()
        .chain(valid_losses.iter())
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);

    let y_margin = (max_loss - min_loss) * 0.1;
    let y_min = (min_loss - y_margin).max(0.0);
    let y_max = max_loss + y_margin;

    let num_epochs = train_losses.len().max(valid_losses.len()) as f32;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Training and Validation Loss",
            ("sans-serif", 30).into_font(),
        )
        .margin(15)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0f32..num_epochs, y_min..y_max)?;

    chart
        .configure_mesh()
        .x_desc("Epoch")
        .y_desc("Loss")
        .draw()?;

    // Plot training loss
    let train_series: Vec<(f32, f32)> = train_losses
        .iter()
        .enumerate()
        .map(|(idx, &loss)| ((idx as f32) + 1.0, loss))
        .collect();

    chart
        .draw_series(LineSeries::new(
            train_series.iter().copied(),
            ShapeStyle::from(&BLUE).stroke_width(2),
        ))?
        .label("Training Loss")
        .legend(|(x, y)| Circle::new((x, y), 3, BLUE.filled()));

    // Plot validation loss
    let valid_series: Vec<(f32, f32)> = valid_losses
        .iter()
        .enumerate()
        .map(|(idx, &loss)| ((idx as f32) + 1.0, loss))
        .collect();

    chart
        .draw_series(LineSeries::new(
            valid_series.iter().copied(),
            ShapeStyle::from(&RED).stroke_width(2),
        ))?
        .label("Validation Loss")
        .legend(|(x, y)| Circle::new((x, y), 3, RED.filled()));

    // Add markers for training loss
    chart.draw_series(
        train_series
            .iter()
            .map(|&point| Circle::new(point, 3, BLUE.filled())),
    )?;

    // Add markers for validation loss
    chart.draw_series(
        valid_series
            .iter()
            .map(|&point| Circle::new(point, 3, RED.filled())),
    )?;

    // Add legend
    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}
