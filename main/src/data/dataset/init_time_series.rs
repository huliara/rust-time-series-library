use crate::args::time_lengths::TimeLengths;

pub trait InitTimeSeries {
    fn split_borders(
        lengths: &TimeLengths,
        total_rows: usize,
    ) -> ((usize, usize, usize), (usize, usize, usize));
}
