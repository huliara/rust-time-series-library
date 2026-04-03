use burn::{prelude::Backend, Tensor};
use std::io::Write;
pub fn save_results<B: Backend>(test_dir: &str, error: Tensor<B, 3>, futures: Tensor<B, 3>) {
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

    let mut mse_writer = csv::Writer::from_path(format!("{test_dir}/mse.csv")).unwrap();
    let mut mae_writer = csv::Writer::from_path(format!("{test_dir}/mae.csv")).unwrap();

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
    let mut file = std::fs::File::create(format!("{test_dir}/results.txt")).unwrap();
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
    fn test_save_results_writes_metric_files() {
        type B = NdArray;
        let base_dir = prepare_test_dir("metrics");
        let test_path = base_dir.to_string_lossy().to_string();
        fs::create_dir_all(base_dir.join("test")).unwrap();

        let batch_size = 20;
        let time_steps = 96;
        let features = 2;
        let device = Default::default();

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

        let test_dir = base_dir.join("test");
        assert!(test_dir.exists());
        assert!(test_dir.join("mse.csv").exists());
        assert!(test_dir.join("mae.csv").exists());
        assert!(test_dir.join("results.txt").exists());

        let results_content = std::fs::read_to_string(test_dir.join("results.txt")).unwrap();
        assert!(results_content.contains("MSE:"));
        assert!(results_content.contains("MAE:"));
        assert!(results_content.contains("RMSE:"));
        assert!(results_content.contains("MAPE:"));
        assert!(results_content.contains("MSPE:"));
    }
}
