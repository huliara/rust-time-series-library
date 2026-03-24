use std::{
    fmt::{Debug, Display},
    path::PathBuf,
};

use crate::{
    args::{time_embed::TimeEmbed, time_lengths::TimeLengths},
    data::dataset::{
        init_time_series::InitTimeSeries,
        standard_scaler::StandardScaler,
        time_series_dataset::{ExpFlag, TimeSeriesDataset},
        util::time_features,
    },
};
use burn::{prelude::Backend, tensor::TensorData, Tensor};
use chrono::{Datelike, NaiveDateTime, Timelike};
use clap::ValueEnum;
use lib::env_path::get_dataset_path;
use ndarray::{s, Array2};
use polars::prelude::*;

pub trait InitRealTimeSeries<
    C: Clone
        + std::marker::Send
        + std::marker::Sync
        + 'static
        + ValueEnum
        + Display
        + Debug
        + PartialEq,
>: InitTimeSeries
{
    fn parse_dates(df: &DataFrame, start_idx: usize, slice_len: usize) -> Vec<NaiveDateTime>;
    fn path(&self) -> String;
    fn train_columns(&self) -> Vec<C>;
    fn target_columns(&self) -> Vec<C>;
    fn embed(&self) -> TimeEmbed;

    fn read_data(path: String) -> Result<DataFrame, PolarsError> {
        let path = PathBuf::from(get_dataset_path(path.clone()));
        CsvReadOptions::default()
            .with_has_header(true)
            .try_into_reader_with_file_path(Some(path))
            .expect("Failed to read CSV file")
            .finish()
    }

    fn init<B: Backend>(
        &self,
        lengths: &TimeLengths,
        flag: ExpFlag,
        device: &B::Device,
    ) -> TimeSeriesDataset<B> {
        let path = self.path();
        let train_columns = self.train_columns();
        let target_columns = self.target_columns();
        let embed = self.embed();
        let df = Self::read_data(path.clone());
        match df {
            Ok(df) => {
                let train_features = train_columns
                    .clone()
                    .iter()
                    .map(|f| col(f.to_string()))
                    .collect::<Vec<_>>();
                let target_features = target_columns
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

                let (raw_border1s, raw_border2s) = Self::split_borders(lengths, total_rows);

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

                let data_stamp_array: Array2<f64> = match embed {
                    TimeEmbed::Fixed => {
                        let dates: Vec<NaiveDateTime> =
                            Self::parse_dates(&df, start_idx, slice_len);
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
                            Self::parse_dates(&df, start_idx, slice_len);
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

                TimeSeriesDataset {
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
