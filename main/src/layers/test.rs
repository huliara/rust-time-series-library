use crate::args::{column_name::ColumnName, data_config::DataConfig};
use crate::data::test_utils::setup_test_dataloader;
use crate::layers::Layer;
use crate::test_utils::{
    dim::Dim,
    test_py::{execute_python_forward_multidim, execute_python_forward_onedim},
};
use burn::{
    tensor::{backend::Backend, TensorData, Tolerance},
    Tensor,
};
use std::vec;

pub fn assert_layer_forward<B: Backend>(dim: Dim, layer: Layer<B>) {
    let data_config = match dim {
        Dim::Multidim => DataConfig::default(),
        Dim::Onedim => DataConfig {
            train_features: vec![ColumnName::Ot],
            targets: vec![ColumnName::Ot],
            ..DataConfig::default()
        },
    };
    let data_loader = setup_test_dataloader(data_config);
    let mut rust_vec = Vec::with_capacity(3);
    for batch in data_loader.iter() {
        let output = layer.forward(batch.x);
        rust_vec.push(output);
    }
    let rust_tensor = Tensor::cat(rust_vec, 0).to_data();
    let layer_name = layer.to_string();
    let py_forward_results: Vec<f64> = match dim {
        Dim::Multidim => execute_python_forward_multidim(&layer_name).unwrap(),
        Dim::Onedim => execute_python_forward_onedim(&layer_name).unwrap(),
    };

    let py_tensor = TensorData::new(py_forward_results, rust_tensor.clone().shape);

    assert_eq!(py_tensor.shape, rust_tensor.shape);

    py_tensor.assert_approx_eq::<f64>(&rust_tensor, Tolerance::default());
}
