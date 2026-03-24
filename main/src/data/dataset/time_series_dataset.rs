use burn::{
    data::dataset::Dataset,
    tensor::{backend::Backend, Tensor},
};

#[derive(Clone, Debug)]
pub struct TimeSeriesItem<B: Backend> {
    pub seq_x: Tensor<B, 2>,
    pub seq_y: Tensor<B, 2>,
    pub seq_x_mark: Tensor<B, 2>,
    pub seq_y_mark: Tensor<B, 2>,
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
    use crate::args::data::DataCommand;
    use crate::args::time_lengths::TimeLengths;
    use crate::data::dataset::get_dataset::get_dataset;
    use crate::data::dataset::time_series_dataset::ExpFlag;
    use crate::test_utils::assert_tensor_shape_value::assert_tensor_shape_and_val;
    use crate::test_utils::test_py::execute_dataset_test;
    use burn::tensor::TensorData;
    #[test]
    fn test_time_series_dataset() {
        type B = burn::backend::wgpu::Wgpu;
        let py_dataset_result = execute_dataset_test().unwrap();
        let device = Default::default();
        let data_config = DataCommand::default();
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
