use burn::{prelude::Backend, tensor::TensorData, Tensor};
use clap::ValueEnum;
use ndarray::{s, Array2};
use strum::Display;

use crate::{
    args::time_lengths::TimeLengths,
    data::dataset::{
        standard_scaler::StandardScaler,
        time_series_dataset::{ExpFlag, TimeSeriesDataset},
    },
};

#[derive(Clone, Debug, PartialEq, Eq, ValueEnum, Display)]
pub enum DynamicColumnName {
    Value,
}

pub fn split_borders(
    lengths: &TimeLengths,
    total_rows: usize,
) -> ((usize, usize, usize), (usize, usize, usize)) {
    let num_train = (total_rows as f64 * 0.7) as usize;
    let num_test = (total_rows as f64 * 0.2) as usize;
    let num_val = total_rows.saturating_sub(num_train + num_test);

    let raw_border1s = (
        0,
        num_train.saturating_sub(lengths.seq_len),
        total_rows.saturating_sub(num_test.saturating_add(lengths.seq_len)),
    );
    let raw_border2s: (usize, usize, usize) = (num_train, num_train + num_val, total_rows);

    (raw_border1s, raw_border2s)
}

pub fn from_series<B: Backend>(
    series: Vec<Vec<f64>>,
    lengths: &TimeLengths,
    flag: ExpFlag,
    device: &B::Device,
) -> TimeSeriesDataset<B> {
    if series.is_empty() {
        panic!("Generated dynamic series cannot be empty");
    }

    let total_rows = series.len();
    let cols = series[0].len();
    if cols == 0 {
        panic!("Generated dynamic series must have at least one feature column");
    }
    if series.iter().any(|row| row.len() != cols) {
        panic!("Generated dynamic series has inconsistent row lengths");
    }

    let flat = series.into_iter().flatten().collect::<Vec<_>>();
    let data_array = Array2::from_shape_vec((total_rows, cols), flat)
        .expect("Failed to convert generated dynamic series into ndarray");

    let (raw_border1s, raw_border2s) = split_borders(lengths, total_rows);
    let clamp_idx = |idx: usize| idx.min(total_rows);
    let border1s = (
        clamp_idx(raw_border1s.0),
        clamp_idx(raw_border1s.1),
        clamp_idx(raw_border1s.2),
    );
    let border2s = (
        clamp_idx(raw_border2s.0),
        clamp_idx(raw_border2s.1),
        clamp_idx(raw_border2s.2),
    );

    if border2s.0 <= border1s.0 {
        panic!(
            "Invalid train split range: start={}, end={}, total_rows={}",
            border1s.0, border2s.0, total_rows
        );
    }

    let (start_idx, end_idx) = match flag {
        ExpFlag::Train => (border1s.0, border2s.0),
        ExpFlag::Val => (border1s.1, border2s.1),
        ExpFlag::Test => (border1s.2, border2s.2),
    };

    let mut scaler = StandardScaler::new();
    let train_slice = data_array.slice(s![border1s.0..border2s.0, ..]).to_owned();
    scaler.fit(&train_slice);
    let scaled_data = scaler.transform(&data_array);

    let data_x_array = scaled_data.slice(s![start_idx..end_idx, ..]).to_owned();
    let data_y_array = data_x_array.clone();
    let data_stamp_array = Array2::<f64>::zeros((end_idx.saturating_sub(start_idx), 1));

    let shape_x = data_x_array.shape().to_vec();
    let data_x = Tensor::from_data(
        TensorData::new(data_x_array.into_raw_vec_and_offset().0, shape_x),
        device,
    );

    let shape_y = data_y_array.shape().to_vec();
    let data_y = Tensor::from_data(
        TensorData::new(data_y_array.into_raw_vec_and_offset().0, shape_y),
        device,
    );

    let shape_stamp = data_stamp_array.shape().to_vec();
    let data_stamp = Tensor::from_data(
        TensorData::new(data_stamp_array.into_raw_vec_and_offset().0, shape_stamp),
        device,
    );

    TimeSeriesDataset {
        data_x,
        data_y,
        data_stamp,
        seq_len: lengths.seq_len,
        label_len: lengths.label_len,
        pred_len: lengths.pred_len,
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::TensorData;

    use crate::{
        args::time_lengths::TimeLengths,
        data::dataset::{
            dynamic_system::{
                doublescroll::doublescroll, henon_map::henon_map,
                kuramoto_sivashinsky::kuramoto_sivashinsky, logistic_map::logistic_map,
                lorenz::lorenz, lorenz96::lorenz96, mackey_glass::mackey_glass,
                multiscroll::multiscroll, narma::narma, rabinovich_fabrikant::rabinovich_fabrikant,
                rossler::rossler,
            },
            time_series_dataset::ExpFlag,
        },
        test_utils::{
            assert_tensor_shape_value::assert_tensor_shape_and_val,
            test_py::execute_dynamic_system_dataset_test,
        },
    };

    use super::from_series;

    type B = burn::backend::wgpu::Wgpu;

    fn assert_dataset_matches_python(system_name: &str, series: Vec<Vec<f64>>) {
        let lengths = TimeLengths::default();
        let device = Default::default();

        let py_dataset_result = execute_dynamic_system_dataset_test(system_name).unwrap();
        let rust_dataset = from_series::<B>(series, &lengths, ExpFlag::Test, &device);

        let py_tensor_stamp = TensorData::new(py_dataset_result.1, rust_dataset.data_stamp.shape());
        let rust_tensor_stamp = rust_dataset.data_stamp.to_data();
        assert_tensor_shape_and_val(py_tensor_stamp, rust_tensor_stamp);

        let py_tensor_x = TensorData::new(py_dataset_result.0, rust_dataset.data_x.shape());
        let rust_tensor_x = rust_dataset.data_x.to_data();
        assert_tensor_shape_and_val(py_tensor_x, rust_tensor_x);

        let py_tensor_y = TensorData::new(py_dataset_result.2, rust_dataset.data_y.shape());
        let rust_tensor_y = rust_dataset.data_y.to_data();
        assert_tensor_shape_and_val(py_tensor_y, rust_tensor_y);
    }

    #[test]
    fn test_dynamic_system_datasets_against_python() {
        let n_timesteps = 400;

        let logistic_series = logistic_map(n_timesteps, 3.9, 0.1)
            .unwrap()
            .into_iter()
            .map(|v| v.to_vec())
            .collect::<Vec<_>>();
        assert_dataset_matches_python("logistic_map", logistic_series);

        let henon_series = henon_map(n_timesteps, 1.4, 0.3, [0.0, 0.0])
            .into_iter()
            .map(|v| v.to_vec())
            .collect::<Vec<_>>();
        assert_dataset_matches_python("henon_map", henon_series);

        let lorenz_series = lorenz(
            n_timesteps,
            28.0,
            10.0,
            2.6666666666666665,
            [1.0, 1.0, 1.0],
            0.01,
        )
        .into_iter()
        .map(|v| v.to_vec())
        .collect::<Vec<_>>();
        assert_dataset_matches_python("lorenz", lorenz_series);

        let lorenz96_series = lorenz96(n_timesteps, 0, 8, 8.0, 0.01, 0.01, None).unwrap();
        assert_dataset_matches_python("lorenz96", lorenz96_series);

        let rossler_series = rossler(n_timesteps, 0.2, 0.2, 5.7, [1.0, 1.0, 1.0], 0.01)
            .into_iter()
            .map(|v| v.to_vec())
            .collect::<Vec<_>>();
        assert_dataset_matches_python("rossler", rossler_series);

        let doublescroll_series = doublescroll(
            n_timesteps,
            1.2,
            3.44,
            0.193,
            2.25,
            11.6,
            [0.1, 0.0, 0.0],
            0.01,
        )
        .into_iter()
        .map(|v| v.to_vec())
        .collect::<Vec<_>>();
        assert_dataset_matches_python("doublescroll", doublescroll_series);

        let multiscroll_series = multiscroll(n_timesteps, 36.0, 3.0, 20.0, [0.1, 0.0, 0.0], 0.01)
            .into_iter()
            .map(|v| v.to_vec())
            .collect::<Vec<_>>();
        assert_dataset_matches_python("multiscroll", multiscroll_series);

        let rf_series = rabinovich_fabrikant(n_timesteps, 0.14, 0.1, [0.1, 0.1, 0.1], 0.005)
            .into_iter()
            .map(|v| v.to_vec())
            .collect::<Vec<_>>();
        assert_dataset_matches_python("rabinovich_fabrikant", rf_series);

        let mackey_glass_series = mackey_glass(n_timesteps, 0, 0.2, 0.1, 10, 1.2, 0.1, None, None)
            .unwrap()
            .into_iter()
            .map(|v| v.to_vec())
            .collect::<Vec<_>>();
        assert_dataset_matches_python("mackey_glass", mackey_glass_series);

        let order = 10;
        let x0 = vec![0.0_f64; order];
        let u = (0..(n_timesteps + order))
            .map(|idx| (idx % 7) as f64 * 0.05)
            .collect::<Vec<_>>();
        let (_u, narma_y) = narma(n_timesteps, order, 0.3, 0.05, 1.5, 0.1, x0, None, Some(u));
        let narma_series = narma_y.into_iter().map(|v| v.to_vec()).collect::<Vec<_>>();
        assert_dataset_matches_python("narma", narma_series);

        let ks_series = kuramoto_sivashinsky(120, 0, 16, 8.0, None, 0.25).unwrap();
        assert_dataset_matches_python("kuramoto_sivashinsky", ks_series);
    }
}
