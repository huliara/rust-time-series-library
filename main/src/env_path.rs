use serde::Deserialize;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

#[derive(Debug, Deserialize, Clone)]
pub struct EnvPath {
    pub result_root_path: String,
    pub data_root_path: String,
    pub python_path: String,
}

fn get_env_paths() -> EnvPath {
    let file = File::open("../env.yml").expect("env.yml not found in the root directory");
    let reader = BufReader::new(file);
    serde_yaml::from_reader(reader).expect("Failed to parse env.yml")
}

pub fn get_result_root_path() -> String {
    get_env_paths().result_root_path
}

pub fn get_dataset_path(path: String) -> String {
    let paths = get_env_paths();

    Path::new(&paths.data_root_path)
        .join(path)
        .to_str()
        .unwrap()
        .to_string()
}

pub fn get_python_path() -> String {
    get_env_paths().python_path
}
