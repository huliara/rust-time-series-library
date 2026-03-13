pub fn build_forward_str(forward_args: Vec<String>, forward_stmts: Vec<String>) -> String {
    let mut out = String::new();

    out.push_str("impl<B: Backend> Model<B> {\n");

    let forward_args_str = if forward_args.is_empty() {
        "x: Tensor<B, 3>".to_string()
    } else {
        forward_args
            .iter()
            .enumerate()
            .map(|(i, arg)| {
                // 推測: 最初の引数は Tensor<B, 2>、追加は任意
                if i == 0 {
                    format!("{}: Tensor<B, 3>", arg)
                } else {
                    format!("{}: Tensor<B, 3>", arg)
                }
            })
            .collect::<Vec<_>>()
            .join(", ")
    };

    out.push_str(&format!(
        "    pub fn forward(&self, {}) -> Tensor<B, 3> {{\n",
        forward_args_str
    ));

    for stmt in &forward_stmts {
        out.push_str(&format!("        {}\n", stmt));
    }

    out.push_str("    }\n");
    out
}
