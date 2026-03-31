use crate::args::time_lengths::TimeLengths;

pub trait InitTimeSeries {
    fn split_borders(
        lengths: &TimeLengths,
        total_rows: usize,
    ) -> ((usize, usize, usize), (usize, usize, usize)) {
        let num_train = (total_rows as f64 * 0.7) as usize;
        let num_test = (total_rows as f64 * 0.2) as usize;
        let num_val = total_rows.saturating_sub(num_train + num_test);

        let raw_border1s = (
            0,
            num_train.saturating_sub(lengths.seq_len),
            total_rows.saturating_sub(num_test.saturating_add(lengths.seq_len)),
        );
        let raw_border2s: (usize, usize, usize) = (num_train, num_train + num_val, total_rows);

        (raw_border1s, raw_border2s)
    }
}
