use nalgebra::DMatrix;

fn main() {
    let a = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
    let inv = a.try_inverse().unwrap();
    let dat: Vec<f32> = inv.transpose().iter().copied().collect();
    println!("{:?}", dat);
}
