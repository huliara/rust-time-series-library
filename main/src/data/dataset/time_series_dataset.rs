use super::util::*;
use crate::args::{
    data_config::{DataCommand, DataConfig},
    time_embed::TimeEmbed,
    time_lengths::TimeLengths,
};
use burn::{
    data::dataset::Dataset,
    tensor::{backend::Backend, Tensor, TensorData},
};
use chrono::{DateTime, Datelike, NaiveDateTime, Timelike};
use clap::ValueEnum;
use lib::env_path::get_dataset_path;
use ndarray::{s, Array1, Array2, Axis};
use polars::prelude::*;
use std::{fmt::Display, path::PathBuf};

#[derive(Clone, Debug)]
pub struct TimeSeriesItem<B: Backend> {
    pub seq_x: Tensor<B, 2>,
    pub seq_y: Tensor<B, 2>,
    pub seq_x_mark: Tensor<B, 2>,
    pub seq_y_mark: Tensor<B, 2>,
}

#[derive(Clone, Debug)]
pub struct StandardScaler {
    pub mean: Array1<f64>,
    pub scale: Array1<f64>,
}

impl StandardScaler {
    pub fn new() -> Self {
        Self {
            mean: Array1::zeros(0),
            scale: Array1::zeros(0),
        }
    }

    pub fn fit(&mut self, data: &Array2<f64>) {
        self.mean = data.mean_axis(Axis(0)).expect("Mean axis 0 failed");
        // Using ddof=0 for consistency with sklearn's StandardScaler which uses biased estimator by default
        self.scale = data.std_axis(Axis(0), 0.0);
        // Avoid division by zero
        self.scale.mapv_inplace(|x| if x == 0.0 { 1.0 } else { x });
    }

    pub fn transform(&self, data: &Array2<f64>) -> Array2<f64> {
        (data - &self.mean) / &self.scale
    }

    pub fn _inverse_transform(&self, data: &Array2<f64>) -> Array2<f64> {
        (data * &self.scale) + &self.mean
    }
}

pub struct TimeSeriesDataset<B: Backend> {
    pub data_x: Tensor<B, 2>,
    pub data_y: Tensor<B, 2>,
    pub data_stamp: Tensor<B, 2>,
    pub seq_len: usize,
    pub label_len: usize,
    pub pred_len: usize,
}

#[derive(Clone, Copy, Debug)]
pub enum ExpFlag {
    Train,
    Val,
    Test,
}

impl<B: Backend> TimeSeriesDataset<B> {
    fn parse_dates(
        df: &DataFrame,
        data_config: &DataConfig,
        start_idx: usize,
        slice_len: usize,
    ) -> Vec<NaiveDateTime> {
        match data_config {
            DataConfig::ETTh1(_) => df
                .slice(start_idx as i64, slice_len)
                .column("date")
                .unwrap()
                .str()
                .unwrap()
                .into_no_null_iter()
                .map(|s| NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S").expect("Parse date"))
                .collect(),
            DataConfig::Exchange(_) => df
                .slice(start_idx as i64, slice_len)
                .column("time")
                .unwrap()
                .i64()
                .unwrap()
                .into_no_null_iter()
                .map(|s| {
                    DateTime::from_timestamp(s, 0)
                        .expect("Parse date")
                        .naive_utc()
                })
                .collect(),
        }
    }

    pub fn new(
        data_config: &DataConfig,
        data_command: &DataCommand<
            impl Clone + std::marker::Send + std::marker::Sync + 'static + ValueEnum + Display,
        >,
        lengths: &TimeLengths,
        flag: ExpFlag,
        device: &B::Device,
    ) -> Self {
        // Default size

        let path = PathBuf::from(get_dataset_path(data_command.path.clone()));
        let df = CsvReadOptions::default()
            .with_has_header(true)
            .try_into_reader_with_file_path(Some(path))
            .expect("Failed to read CSV file")
            .finish();

        match df {
            Ok(df) => {
                let train_features = data_command
                    .train_features
                    .clone()
                    .iter()
                    .map(|f| col(f.to_string()))
                    .collect::<Vec<_>>();
                let target_features = data_command
                    .targets
                    .clone()
                    .iter()
                    .map(|t| col(t.to_string()))
                    .collect::<Vec<_>>();

                let data_x_array: Array2<f64> = df
                    .clone()
                    .lazy()
                    .select(train_features)
                    .collect()
                    .unwrap()
                    .to_ndarray::<Float64Type>(IndexOrder::C)
                    .unwrap()
                    .into_dimensionality::<ndarray::Ix2>()
                    .unwrap();
                let data_y_array = df
                    .clone()
                    .lazy()
                    .select(target_features)
                    .collect()
                    .unwrap()
                    .to_ndarray::<Float64Type>(IndexOrder::C)
                    .unwrap()
                    .into_dimensionality::<ndarray::Ix2>()
                    .unwrap();

                if data_x_array.is_empty() || data_y_array.is_empty() {
                    panic!(
                        "Data arrays cannot be empty. Please check the CSV file and column names."
                    );
                }

                let total_rows = data_x_array.nrows();
                let num_train = (total_rows as f64 * 0.7) as usize;
                let num_test = (total_rows as f64 * 0.2) as usize;
                let num_val = total_rows - num_train - num_test;

                let raw_border1s = match data_config {
                    DataConfig::ETTh1(_) => (
                        0,
                        (12usize * 30 * 24).saturating_sub(lengths.seq_len),
                        (12usize * 30 * 24 + 4 * 30 * 24).saturating_sub(lengths.seq_len),
                    ),
                    DataConfig::Exchange(_) => (
                        0,
                        num_train.saturating_sub(lengths.seq_len),
                        total_rows.saturating_sub(num_test.saturating_add(lengths.seq_len)),
                    ),
                };
                let raw_border2s: (usize, usize, usize) = match data_config {
                    DataConfig::ETTh1(_) => (
                        12 * 30 * 24,
                        12 * 30 * 24 + 4 * 30 * 24,
                        12 * 30 * 24 + 8 * 30 * 24,
                    ),
                    DataConfig::Exchange(_) => (num_train, num_train + num_val, total_rows),
                };

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
                let train_data_sliced = data_x_array
                    .slice(s![border1s.0..border2s.0, ..])
                    .to_owned();
                scaler.fit(&train_data_sliced);

                let data_x_array = scaler.transform(&data_x_array);

                let mut target_scaler = StandardScaler::new();
                let target_data_sliced = data_y_array
                    .slice(s![border1s.0..border2s.0, ..])
                    .to_owned();
                target_scaler.fit(&target_data_sliced);

                let data_y_array = target_scaler.transform(&data_y_array);

                let slice_len = end_idx.saturating_sub(start_idx);

                let data_stamp_array: Array2<f64> = match data_command.embed {
                    TimeEmbed::Fixed => {
                        let dates: Vec<NaiveDateTime> =
                            Self::parse_dates(&df, &data_config, start_idx, slice_len);
                        let month: Vec<f64> = dates.iter().map(|d| d.month() as f64).collect();
                        let day: Vec<f64> = dates.iter().map(|d| d.day() as f64).collect();
                        let weekday: Vec<f64> = dates
                            .iter()
                            .map(|d| d.weekday().number_from_monday() as f64 - 1.0)
                            .collect();
                        let hour: Vec<f64> = dates.iter().map(|d| d.hour() as f64).collect();

                        let month_series = Column::new("month".into(), month);
                        let day_series = Column::new("day".into(), day);
                        let weekday_series = Column::new("weekday".into(), weekday);
                        let hour_series = Column::new("hour".into(), hour);

                        DataFrame::new(vec![month_series, day_series, weekday_series, hour_series])
                            .unwrap()
                            .to_ndarray::<Float64Type>(IndexOrder::C)
                            .unwrap()
                            .into_dimensionality::<ndarray::Ix2>()
                            .unwrap()
                    }
                    TimeEmbed::TimeF => {
                        let dates: Vec<NaiveDateTime> =
                            Self::parse_dates(&df, &data_config, start_idx, slice_len);
                        time_features(&dates, "h")
                    }
                };

                let data_x_array = data_x_array.slice(s![start_idx..end_idx, ..]).to_owned();
                let data_y_array = data_y_array.slice(s![start_idx..end_idx, ..]).to_owned();

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

                Self {
                    data_x,
                    data_y,
                    data_stamp,
                    seq_len: lengths.seq_len,
                    label_len: lengths.label_len,
                    pred_len: lengths.pred_len,
                }
            }
            Err(e) => {
                panic!("Error reading CSV file: {:?}", e);
            }
        }
    }
}

impl<B: Backend> Dataset<TimeSeriesItem<B>> for TimeSeriesDataset<B> {
    fn get(&self, index: usize) -> Option<TimeSeriesItem<B>> {
        if index >= self.len() {
            return None;
        }
        let s_begin = index;
        let s_end = s_begin + self.seq_len;
        let r_begin = s_end;
        let r_end = r_begin + self.pred_len;

        let dim_x = self.data_x.dims()[1];
        let dim_mark = self.data_stamp.dims()[1];

        // Slicing in Burn: ranges for each dimension
        let seq_x = self.data_x.clone().slice([s_begin..s_end, 0..dim_x]);
        let seq_y = self.data_y.clone().slice([r_begin..r_end, 0..dim_x]);
        let seq_x_mark = self.data_stamp.clone().slice([s_begin..s_end, 0..dim_mark]);
        let seq_y_mark = self.data_stamp.clone().slice([r_begin..r_end, 0..dim_mark]);

        Some(TimeSeriesItem {
            seq_x,
            seq_y,
            seq_x_mark,
            seq_y_mark,
        })
    }

    fn len(&self) -> usize {
        let len_x = self.data_x.dims()[0];
        let required = self.seq_len + self.pred_len;
        if len_x >= required {
            len_x - required + 1
        } else {
            0
        }
    }
}
#[cfg(test)]
mod tests {
    use crate::args::time_lengths::TimeLengths;
    use crate::data::dataset::get_dataset::get_dataset;
    use crate::data::dataset::time_series_dataset::ExpFlag;
    use crate::test_utils::test_py::execute_dataset_test;
    use crate::{
        args::data_config::DataConfig,
        test_utils::assert_tensor_shape_value::assert_tensor_shape_and_val,
    };
    use burn::tensor::TensorData;
    #[test]
    fn test_time_series_dataset() {
        type B = burn::backend::wgpu::Wgpu;
        let py_dataset_result = execute_dataset_test().unwrap();
        let device = Default::default();
        let data_config = DataConfig::default();
        let lengths = TimeLengths::default();

        let rust_dataset = get_dataset::<B>(&data_config, &lengths, ExpFlag::Test, &device);

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
}
