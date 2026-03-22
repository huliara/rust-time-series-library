use burn::{
    prelude::{s, Backend},
    Tensor,
};

use crate::exp::long_term_forecast::save_results::plot_prediction::{
    plot_multi_feature_prediction, plot_single_feature_prediction,
};

pub fn sample_plots<B: Backend>(
    contexts: Tensor<B, 3>,
    predicts: Tensor<B, 3>,
    futures: Tensor<B, 3>,
    plot_num: usize,
    test_dir: &str,
) {
    let sample_count = contexts.dims()[0];
    let context_feature_count = contexts.dims()[2];
    let target_feature_count = futures.dims()[2];
    let num_plots = usize::min(plot_num, sample_count);
    let plot_step = usize::max(1, sample_count / num_plots);

    for sample_id in 0..num_plots {
        let sample_idx = usize::min(sample_id * plot_step, sample_count - 1);

        let mut context_multi = Vec::with_capacity(context_feature_count);

        for feature_id in 0..context_feature_count {
            let context_vec = contexts
                .clone()
                .slice(s![sample_idx, .., feature_id])
                .into_data()
                .to_vec::<f32>()
                .unwrap();

            context_multi.push(context_vec);
        }
        let mut pred_multi = Vec::with_capacity(target_feature_count);
        let mut future_multi = Vec::with_capacity(target_feature_count);

        for feature_id in 0..target_feature_count {
            let pred_vec = predicts
                .clone()
                .slice(s![sample_idx, .., feature_id])
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            let future_vec = futures
                .clone()
                .slice(s![sample_idx, .., feature_id])
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            pred_multi.push(pred_vec);
            future_multi.push(future_vec);
        }

        plot_multi_feature_prediction(
            test_dir,
            sample_id,
            &context_multi,
            &pred_multi,
            &future_multi,
        );
        plot_single_feature_prediction(
            test_dir,
            sample_id,
            &context_multi[0],
            &pred_multi[0],
            &future_multi[0],
        );
    }
}
