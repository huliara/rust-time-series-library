use crate::args::column_name::EtthColumnName;
use crate::args::data_config::DataArgs;
use crate::args::data_config::DataConfig;
use crate::data::test_utils::setup_test_dataloader;
use crate::models::traits::Forecast;
use crate::test_utils::{
    dim::Dim,
    test_py::{execute_python_forward_multidim, execute_python_forward_onedim},
};
use burn::{
    tensor::{backend::Backend, TensorData, Tolerance},
    Tensor,
};
use std::any::type_name;
use std::vec;

pub fn assert_module_forecast<B: Backend, M: Forecast<B>>(dim: Dim, module: M) {
    let data_config = match dim {
        Dim::Multidim => DataConfig::default(),
        Dim::Onedim => DataConfig::ETTh1(DataArgs {
            train_features: vec![EtthColumnName::Ot],
            targets: vec![EtthColumnName::Ot],
            ..DataArgs::default()
        }),
    };
    let data_loader = setup_test_dataloader(data_config);
    let mut rust_vec = Vec::with_capacity(3);
    for batch in data_loader.iter() {
        let output = module.forecast(batch.x, batch.x_mark, batch.y, batch.y_mark);
        rust_vec.push(output);
    }
    let rust_tensor = Tensor::cat(rust_vec, 0).to_data();
    let type_name_vec: Vec<&str> = type_name::<M>().split('<').collect();
    let type_name = type_name_vec[0].split("::").last().unwrap();
    let py_forward_results: Vec<f64> = match dim {
        Dim::Multidim => execute_python_forward_multidim(type_name).unwrap(),
        Dim::Onedim => execute_python_forward_onedim(type_name).unwrap(),
    };

    let py_tensor = TensorData::new(py_forward_results, rust_tensor.clone().shape);

    assert_eq!(py_tensor.shape, rust_tensor.shape);

    py_tensor.assert_approx_eq::<f64>(&rust_tensor, Tolerance::default());
}
