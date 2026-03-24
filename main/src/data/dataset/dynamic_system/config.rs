use burn::{prelude::Backend, tensor::TensorData, Tensor};
use clap::ValueEnum;
use ndarray::{s, Array2};
use serde::{Deserialize, Serialize};

use crate::{
    args::{time_embed::TimeEmbed, time_lengths::TimeLengths},
    data::dataset::time_series_dataset::{ExpFlag, TimeSeriesDataset},
};

#[derive(Debug, Clone, ValueEnum, PartialEq, Eq, Deserialize, Serialize, strum::Display)]
pub enum DynamicColumnName {
    #[strum(serialize = "value")]
    Value,
}

pub fn split_borders(
    lengths: &TimeLengths,
    total_rows: usize,
) -> ((usize, usize, usize), (usize, usize, usize)) {
    let num_train = (total_rows as f64 * 0.7) as usize;
    let num_test = (total_rows as f64 * 0.2) as usize;
    let num_val = total_rows.saturating_sub(num_train).saturating_sub(num_test);

    let raw_border1s = (
        0,
        num_train.saturating_sub(lengths.seq_len),
        total_rows.saturating_sub(num_test.saturating_add(lengths.seq_len)),
    );
    let raw_border2s = (num_train, num_train + num_val, total_rows);
    (raw_border1s, raw_border2s)
}

pub fn default_parse_dates(_start_idx: usize, _slice_len: usize) -> Vec<chrono::NaiveDateTime> {
    Vec::new()
}

pub fn default_path() -> String {
    String::new()
}

pub fn default_columns() -> Vec<DynamicColumnName> {
    vec![DynamicColumnName::Value]
}

pub fn default_embed() -> TimeEmbed {
    TimeEmbed::TimeF
}

pub fn from_series<B: Backend>(
    series: Vec<Vec<f64>>,
    lengths: &TimeLengths,
    flag: ExpFlag,
    device: &B::Device,
) -> TimeSeriesDataset<B> {
    if series.is_empty() {
        panic!("Generated dynamic-system series is empty");
    }

    let feature_dim = series[0].len();
    if feature_dim == 0 {
        panic!("Generated dynamic-system feature dimension is zero");
    }
    if series.iter().any(|row| row.len() != feature_dim) {
        panic!("All dynamic-system rows must have the same feature dimension");
    }

    let total_rows = series.len();
    let flat = series.into_iter().flatten().collect::<Vec<_>>();
    let data_array = Array2::from_shape_vec((total_rows, feature_dim), flat)
        .expect("Failed to reshape dynamic-system data into Array2");

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

    let (start_idx, end_idx) = match flag {
        ExpFlag::Train => (border1s.0, border2s.0),
        ExpFlag::Val => (border1s.1, border2s.1),
        ExpFlag::Test => (border1s.2, border2s.2),
    };

    if end_idx <= start_idx {
        panic!(
            "Invalid dynamic-system split range: start={}, end={}, total_rows={}",
            start_idx, end_idx, total_rows
        );
    }

    let data_x_array = data_array.slice(s![start_idx..end_idx, ..]).to_owned();
    let data_y_array = data_x_array.clone();
    let data_stamp_array = Array2::from_shape_fn((end_idx - start_idx, 1), |(i, _)| {
        (start_idx + i) as f64
    });

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
