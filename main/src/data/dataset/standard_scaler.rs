use ndarray::{Array1, Array2, Axis};

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
