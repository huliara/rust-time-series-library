use burn::tensor::TensorData;

use crate::test_utils::{
    assert_tensor_shape_value::assert_tensor_shape_and_val,
    test_py::execute_dynamic_system_dataset_test,
};

pub fn assert_dynamic_system_series(system_name: &str, series: Vec<Vec<f64>>) {
    let py_dataset_result = execute_dynamic_system_dataset_test(system_name).unwrap();
    let rust_vec = series.into_iter().flatten().collect::<Vec<f64>>();
    let rust_tensor = TensorData::new(rust_vec, [py_dataset_result.clone().len()]);
    let py_tensor_x = TensorData::new(py_dataset_result.clone(), [py_dataset_result.len()]);

    assert_tensor_shape_and_val(py_tensor_x, rust_tensor);
}
