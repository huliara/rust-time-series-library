use burn::{tensor::TensorData, tensor::Tolerance};

pub fn assert_tensor_shape_and_val(py_tensor: TensorData, rust_tensor: TensorData) {
    assert_eq!(py_tensor.shape, rust_tensor.shape);
    py_tensor.assert_approx_eq::<f32>(&rust_tensor, Tolerance::default());
}
