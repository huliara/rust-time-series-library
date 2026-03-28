use burn::tensor::TensorData;
use std::{env, process::Command};

use crate::test_utils::assert_tensor_shape_value::assert_tensor_shape_and_val;

pub const TEST_STEP_SIZE: usize = 500;

fn execute_dynamic_system_dataset_test(system_name: &str) -> Vec<f64> {
    let current_dir = env::current_dir().expect("failed to get current directory");
    let project_root = current_dir
        .parent()
        .expect("failed to resolve project root");
    let python_exec = project_root.join(".venv/bin/python");
    let python_dir = project_root.join("python");

    let script = r#"
import sys
from _dynamic_system_dataset_test import dynamic_system_dataset_test

for x in dynamic_system_dataset_test(sys.argv[1]):
    print(x)
"#;

    let output = Command::new(python_exec)
        .current_dir(project_root)
        .env("PYTHONPATH", python_dir)
        .arg("-c")
        .arg(script)
        .arg(system_name)
        .output()
        .expect("failed to execute python for dynamic system comparison");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!(
            "Python dynamic system test failed for {system_name}: {}",
            stderr.trim()
        );
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(|line| {
            line.parse::<f64>()
                .unwrap_or_else(|e| panic!("failed to parse python output '{line}': {e}"))
        })
        .collect()
}

pub fn assert_dynamic_system_series(system_name: &str, series: Vec<Vec<f64>>) {
    let py_dataset_result = execute_dynamic_system_dataset_test(system_name);
    let rust_vec = series.into_iter().flatten().collect::<Vec<f64>>();
    let rust_tensor = TensorData::new(rust_vec, [py_dataset_result.clone().len()]);
    let py_tensor_x = TensorData::new(py_dataset_result.clone(), [py_dataset_result.len()]);

    assert_tensor_shape_and_val(py_tensor_x, rust_tensor);
}
