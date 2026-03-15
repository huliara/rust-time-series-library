#[derive(Config, Debug)]
pub struct TransposeConfig {
#[config(
 default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}")]
    pub initializer: Initializer,
}

impl TransposeConfig {
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Transpose<B> {
        /* TODO: super().__init__() */
        let (self.dims, self.contiguous) = (dims, contiguous);
Transpose {
    }
}
}
struct Transpose<B: Backend> {
}
impl<B: Backend> Model<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        /* TODO: If(StmtIf { range: 399..531, test: Attribute(ExprAttribute { range: 402..417, value: Name(ExprName { range: 402..406, id: Identifier("self"), ctx: Load }), attr: Identifier("contiguous"), ctx: Load }), body: [Return(StmtReturn { range: 431..474, value: Some(Call(ExprCall { range: 438..474, func: Attribute(ExprAttribute { range: 438..472, value: Call(ExprCall { range: 438..461, func: Attribute(ExprAttribute { range: 438..449, value: Name(ExprName { range: 438..439, id: Identifier("x"), ctx: Load }), attr: Identifier("transpose"), ctx: Load }), args: [Starred(ExprStarred { range: 450..460, value: Attribute(ExprAttribute { range: 451..460, value: Name(ExprName { range: 451..455, id: Identifier("self"), ctx: Load }), attr: Identifier("dims"), ctx: Load }), ctx: Load })], keywords: [] }), attr: Identifier("contiguous"), ctx: Load }), args: [], keywords: [] })) })], orelse: [Return(StmtReturn { range: 501..531, value: Some(Call(ExprCall { range: 508..531, func: Attribute(ExprAttribute { range: 508..519, value: Name(ExprName { range: 508..509, id: Identifier("x"), ctx: Load }), attr: Identifier("transpose"), ctx: Load }), args: [Starred(ExprStarred { range: 520..530, value: Attribute(ExprAttribute { range: 521..530, value: Name(ExprName { range: 521..525, id: Identifier("self"), ctx: Load }), attr: Identifier("dims"), ctx: Load }), ctx: Load })], keywords: [] })) })] }) */
    }
}
#[derive(Config, Debug)]
pub struct FlattenHeadConfig {
    pub n_vars:usize ,
    pub nf:usize ,
    pub target_window:usize ,
    pub head_dropout:usize ,
#[config(
 default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}")]
    pub initializer: Initializer,
}

impl FlattenHeadConfig {
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> FlattenHead<B> {
        /* TODO: super().__init__() */
        let n_vars = n_vars;
        let flatten = nn.Flatten(-2);
        let linear = nn.Linear(nf, target_window);
        let dropout = nn.Dropout(head_dropout);
FlattenHead {
            n_vars,
            flatten,
            linear,
            dropout,
    }
}
}
struct FlattenHead<B: Backend> {
    n_vars:,
    flatten:,
    linear:,
    dropout:,
}
impl<B: Backend> Model<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.flatten.forward(x);
        let x = self.linear.forward(x);
        let x = self.dropout.forward(x);
        x
    }
}
#[derive(Debug, Clone, Deserialize, Serialize, Args)]
pub struct ModelArgs {
    #[arg(long, default_value_t = )]
    pub patch_len:usize ,
    #[arg(long, default_value_t = )]
    pub stride:usize ,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    model_args: ModelArgs,
#[config(
 default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}")]
    pub initializer: Initializer,
}

impl ModelConfig {
    pub fn init<B: Backend>(
        &self,
        task_name: TaskName,
        lengths: TimeLengths,
        device: &B::Device,
    ) -> Model<B> {
        /* TODO: "
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        " */
        /* TODO: super().__init__() */
        let task_name = configs.task_name;
        let seq_len = configs.seq_len;
        let pred_len = configs.pred_len;
        let padding = stride;
        let patch_embedding = PatchEmbedding(configs.d_model, patch_len, stride, padding, configs.dropout);
        let encoder = Encoder(/* TODO: ... */);
        let head_nf = configs.d_model * int(configs.seq_len - patch_len / stride + 2);
        /* TODO: If(StmtIf { range: 2634..3502, test: BoolOp(ExprBoolOp { range: 2651..2744, op: Or, values: [Compare(ExprCompare { range: 2651..2689, left: Attribute(ExprAttribute { range: 2651..2665, value: Name(ExprName { range: 2651..2655, id: Identifier("self"), ctx: Load }), attr: Identifier("task_name"), ctx: Load }), ops: [Eq], comparators: [Constant(ExprConstant { range: 2669..2689, value: Str("long_term_forecast"), kind: None })] }), Compare(ExprCompare { range: 2705..2744, left: Attribute(ExprAttribute { range: 2705..2719, value: Name(ExprName { range: 2705..2709, id: Identifier("self"), ctx: Load }), attr: Identifier("task_name"), ctx: Load }), ops: [Eq], comparators: [Constant(ExprConstant { range: 2723..2744, value: Str("short_term_forecast"), kind: None })] })] }), body: [Assign(StmtAssign { range: 2768..2948, targets: [Attribute(ExprAttribute { range: 2768..2777, value: Name(ExprName { range: 2768..2772, id: Identifier("self"), ctx: Load }), attr: Identifier("head"), ctx: Store })], value: Call(ExprCall { range: 2780..2948, func: Name(ExprName { range: 2780..2791, id: Identifier("FlattenHead"), ctx: Load }), args: [Attribute(ExprAttribute { range: 2809..2823, value: Name(ExprName { range: 2809..2816, id: Identifier("configs"), ctx: Load }), attr: Identifier("enc_in"), ctx: Load }), Attribute(ExprAttribute { range: 2841..2853, value: Name(ExprName { range: 2841..2845, id: Identifier("self"), ctx: Load }), attr: Identifier("head_nf"), ctx: Load }), Attribute(ExprAttribute { range: 2871..2887, value: Name(ExprName { range: 2871..2878, id: Identifier("configs"), ctx: Load }), attr: Identifier("pred_len"), ctx: Load })], keywords: [Keyword { range: 2905..2933, arg: Some(Identifier("head_dropout")), value: Attribute(ExprAttribute { range: 2918..2933, value: Name(ExprName { range: 2918..2925, id: Identifier("configs"), ctx: Load }), attr: Identifier("dropout"), ctx: Load }) }] }), type_comment: None })], orelse: [If(StmtIf { range: 2957..3502, test: BoolOp(ExprBoolOp { range: 2962..3033, op: Or, values: [Compare(ExprCompare { range: 2962..2992, left: Attribute(ExprAttribute { range: 2962..2976, value: Name(ExprName { range: 2962..2966, id: Identifier("self"), ctx: Load }), attr: Identifier("task_name"), ctx: Load }), ops: [Eq], comparators: [Constant(ExprConstant { range: 2980..2992, value: Str("imputation"), kind: None })] }), Compare(ExprCompare { range: 2996..3033, left: Attribute(ExprAttribute { range: 2996..3010, value: Name(ExprName { range: 2996..3000, id: Identifier("self"), ctx: Load }), attr: Identifier("task_name"), ctx: Load }), ops: [Eq], comparators: [Constant(ExprConstant { range: 3014..3033, value: Str("anomaly_detection"), kind: None })] })] }), body: [Assign(StmtAssign { range: 3047..3226, targets: [Attribute(ExprAttribute { range: 3047..3056, value: Name(ExprName { range: 3047..3051, id: Identifier("self"), ctx: Load }), attr: Identifier("head"), ctx: Store })], value: Call(ExprCall { range: 3059..3226, func: Name(ExprName { range: 3059..3070, id: Identifier("FlattenHead"), ctx: Load }), args: [Attribute(ExprAttribute { range: 3088..3102, value: Name(ExprName { range: 3088..3095, id: Identifier("configs"), ctx: Load }), attr: Identifier("enc_in"), ctx: Load }), Attribute(ExprAttribute { range: 3120..3132, value: Name(ExprName { range: 3120..3124, id: Identifier("self"), ctx: Load }), attr: Identifier("head_nf"), ctx: Load }), Attribute(ExprAttribute { range: 3150..3165, value: Name(ExprName { range: 3150..3157, id: Identifier("configs"), ctx: Load }), attr: Identifier("seq_len"), ctx: Load })], keywords: [Keyword { range: 3183..3211, arg: Some(Identifier("head_dropout")), value: Attribute(ExprAttribute { range: 3196..3211, value: Name(ExprName { range: 3196..3203, id: Identifier("configs"), ctx: Load }), attr: Identifier("dropout"), ctx: Load }) }] }), type_comment: None })], orelse: [If(StmtIf { range: 3235..3502, test: Compare(ExprCompare { range: 3240..3274, left: Attribute(ExprAttribute { range: 3240..3254, value: Name(ExprName { range: 3240..3244, id: Identifier("self"), ctx: Load }), attr: Identifier("task_name"), ctx: Load }), ops: [Eq], comparators: [Constant(ExprConstant { range: 3258..3274, value: Str("classification"), kind: None })] }), body: [Assign(StmtAssign { range: 3288..3327, targets: [Attribute(ExprAttribute { range: 3288..3300, value: Name(ExprName { range: 3288..3292, id: Identifier("self"), ctx: Load }), attr: Identifier("flatten"), ctx: Store })], value: Call(ExprCall { range: 3303..3327, func: Attribute(ExprAttribute { range: 3303..3313, value: Name(ExprName { range: 3303..3305, id: Identifier("nn"), ctx: Load }), attr: Identifier("Flatten"), ctx: Load }), args: [], keywords: [Keyword { range: 3314..3326, arg: Some(Identifier("start_dim")), value: UnaryOp(ExprUnaryOp { range: 3324..3326, op: USub, operand: Constant(ExprConstant { range: 3325..3326, value: Int(2), kind: None }) }) }] }), type_comment: None }), Assign(StmtAssign { range: 3340..3382, targets: [Attribute(ExprAttribute { range: 3340..3352, value: Name(ExprName { range: 3340..3344, id: Identifier("self"), ctx: Load }), attr: Identifier("dropout"), ctx: Store })], value: Call(ExprCall { range: 3355..3382, func: Attribute(ExprAttribute { range: 3355..3365, value: Name(ExprName { range: 3355..3357, id: Identifier("nn"), ctx: Load }), attr: Identifier("Dropout"), ctx: Load }), args: [Attribute(ExprAttribute { range: 3366..3381, value: Name(ExprName { range: 3366..3373, id: Identifier("configs"), ctx: Load }), attr: Identifier("dropout"), ctx: Load })], keywords: [] }), type_comment: None }), Assign(StmtAssign { range: 3395..3502, targets: [Attribute(ExprAttribute { range: 3395..3410, value: Name(ExprName { range: 3395..3399, id: Identifier("self"), ctx: Load }), attr: Identifier("projection"), ctx: Store })], value: Call(ExprCall { range: 3413..3502, func: Attribute(ExprAttribute { range: 3413..3422, value: Name(ExprName { range: 3413..3415, id: Identifier("nn"), ctx: Load }), attr: Identifier("Linear"), ctx: Load }), args: [BinOp(ExprBinOp { range: 3440..3469, left: Attribute(ExprAttribute { range: 3440..3452, value: Name(ExprName { range: 3440..3444, id: Identifier("self"), ctx: Load }), attr: Identifier("head_nf"), ctx: Load }), op: Mult, right: Attribute(ExprAttribute { range: 3455..3469, value: Name(ExprName { range: 3455..3462, id: Identifier("configs"), ctx: Load }), attr: Identifier("enc_in"), ctx: Load }) }), Attribute(ExprAttribute { range: 3471..3488, value: Name(ExprName { range: 3471..3478, id: Identifier("configs"), ctx: Load }), attr: Identifier("num_class"), ctx: Load })], keywords: [] }), type_comment: None })], orelse: [] })] })] }) */
        Model {
            task_name,
            seq_len,
            pred_len,
            patch_embedding,
            encoder,
            head_nf,
    }
}
}
impl<B: Backend> Model<B> {
    pub fn forecast(&self, x_enc: Tensor<B, 3>, x_mark_enc: Tensor<B, 3>, x_dec: Tensor<B, 3>, x_mark_dec: Tensor<B, 3>) -> Tensor<B, 3> {
        let means = x_enc.mean_dim(1);
        let x_enc = x_enc - means;
        let stdev = torch.sqrt(torch.var(x_enc, 1, true, false) + 0.00001);
        let x_enc = x_enc / stdev;
        let x_enc = x_enc.permute([0, 2, 1]);
        let (enc_out, n_vars) = self.patch_embedding.forward(x_enc);
        let (enc_out, attns) = self.encoder.forward(enc_out);
        let enc_out = torch.reshape([enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])]);
        let enc_out = enc_out.permute([0, 1, 3, 2]);
        let dec_out = self.head.forward(enc_out);
        let dec_out = dec_out.permute([0, 2, 1]);
        let dec_out = dec_out * stdev[(/* TODO: ... */, 0, /* TODO: ... */)].unsqueeze_dim(1).repeat_dim(/* TODO: 1, self.pred_len, 1 */);
        let dec_out = dec_out + means[(/* TODO: ... */, 0, /* TODO: ... */)].unsqueeze_dim(1).repeat_dim(/* TODO: 1, self.pred_len, 1 */);
        dec_out
    }
}
impl<B: Backend> Model<B> {
    pub fn imputation(&self, x_enc: Tensor<B, 3>, x_mark_enc: Tensor<B, 3>, x_dec: Tensor<B, 3>, x_mark_dec: Tensor<B, 3>, mask: Tensor<B, 3>) -> Tensor<B, 3> {
        let means = torch.sum_dim(x_enc) / torch.sum_dim(/* TODO: ... */);
        let means = means.unsqueeze_dim(1);
        let x_enc = x_enc - means;
        let x_enc = x_enc.masked_fill(/* TODO: ... */, 0);
        let stdev = torch.sqrt(torch.sum_dim(x_enc * x_enc) / torch.sum_dim(/* TODO: ... */) + 0.00001);
        let stdev = stdev.unsqueeze_dim(1);
        let x_enc = x_enc / stdev;
        let x_enc = x_enc.permute([0, 2, 1]);
        let (enc_out, n_vars) = self.patch_embedding.forward(x_enc);
        let (enc_out, attns) = self.encoder.forward(enc_out);
        let enc_out = torch.reshape([enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])]);
        let enc_out = enc_out.permute([0, 1, 3, 2]);
        let dec_out = self.head.forward(enc_out);
        let dec_out = dec_out.permute([0, 2, 1]);
        let dec_out = dec_out * stdev[(/* TODO: ... */, 0, /* TODO: ... */)].unsqueeze_dim(1).repeat_dim(/* TODO: 1, self.seq_len, 1 */);
        let dec_out = dec_out + means[(/* TODO: ... */, 0, /* TODO: ... */)].unsqueeze_dim(1).repeat_dim(/* TODO: 1, self.seq_len, 1 */);
        dec_out
    }
}
impl<B: Backend> Model<B> {
    pub fn anomaly_detection(&self, x_enc: Tensor<B, 3>) -> Tensor<B, 3> {
        let means = x_enc.mean_dim(1);
        let x_enc = x_enc - means;
        let stdev = torch.sqrt(torch.var(x_enc, 1, true, false) + 0.00001);
        let x_enc = x_enc / stdev;
        let x_enc = x_enc.permute([0, 2, 1]);
        let (enc_out, n_vars) = self.patch_embedding.forward(x_enc);
        let (enc_out, attns) = self.encoder.forward(enc_out);
        let enc_out = torch.reshape([enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])]);
        let enc_out = enc_out.permute([0, 1, 3, 2]);
        let dec_out = self.head.forward(enc_out);
        let dec_out = dec_out.permute([0, 2, 1]);
        let dec_out = dec_out * stdev[(/* TODO: ... */, 0, /* TODO: ... */)].unsqueeze_dim(1).repeat_dim(/* TODO: 1, self.seq_len, 1 */);
        let dec_out = dec_out + means[(/* TODO: ... */, 0, /* TODO: ... */)].unsqueeze_dim(1).repeat_dim(/* TODO: 1, self.seq_len, 1 */);
        dec_out
    }
}
impl<B: Backend> Model<B> {
    pub fn classification(&self, x_enc: Tensor<B, 3>, x_mark_enc: Tensor<B, 3>) -> Tensor<B, 3> {
        let means = x_enc.mean_dim(1);
        let x_enc = x_enc - means;
        let stdev = torch.sqrt(torch.var(x_enc, 1, true, false) + 0.00001);
        let x_enc = x_enc / stdev;
        let x_enc = x_enc.permute([0, 2, 1]);
        let (enc_out, n_vars) = self.patch_embedding.forward(x_enc);
        let (enc_out, attns) = self.encoder.forward(enc_out);
        let enc_out = torch.reshape([enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])]);
        let enc_out = enc_out.permute([0, 1, 3, 2]);
        let output = self.flatten.forward(enc_out);
        let output = self.dropout.forward(output);
        let output = output.reshape([output.shape[0], -1]);
        let output = self.projection.forward(output);
        output
    }
}
impl<B: Backend> Model<B> {
    pub fn forward(&self, x_enc: Tensor<B, 3>, x_mark_enc: Tensor<B, 3>, x_dec: Tensor<B, 3>, x_mark_dec: Tensor<B, 3>, mask: Tensor<B, 3>) -> Tensor<B, 3> {
        /* TODO: If(StmtIf { range: 8596..8842, test: BoolOp(ExprBoolOp { range: 8613..8706, op: Or, values: [Compare(ExprCompare { range: 8613..8651, left: Attribute(ExprAttribute { range: 8613..8627, value: Name(ExprName { range: 8613..8617, id: Identifier("self"), ctx: Load }), attr: Identifier("task_name"), ctx: Load }), ops: [Eq], comparators: [Constant(ExprConstant { range: 8631..8651, value: Str("long_term_forecast"), kind: None })] }), Compare(ExprCompare { range: 8667..8706, left: Attribute(ExprAttribute { range: 8667..8681, value: Name(ExprName { range: 8667..8671, id: Identifier("self"), ctx: Load }), attr: Identifier("task_name"), ctx: Load }), ops: [Eq], comparators: [Constant(ExprConstant { range: 8685..8706, value: Str("short_term_forecast"), kind: None })] })] }), body: [Assign(StmtAssign { range: 8730..8791, targets: [Name(ExprName { range: 8730..8737, id: Identifier("dec_out"), ctx: Store })], value: Call(ExprCall { range: 8740..8791, func: Attribute(ExprAttribute { range: 8740..8753, value: Name(ExprName { range: 8740..8744, id: Identifier("self"), ctx: Load }), attr: Identifier("forecast"), ctx: Load }), args: [Name(ExprName { range: 8754..8759, id: Identifier("x_enc"), ctx: Load }), Name(ExprName { range: 8761..8771, id: Identifier("x_mark_enc"), ctx: Load }), Name(ExprName { range: 8773..8778, id: Identifier("x_dec"), ctx: Load }), Name(ExprName { range: 8780..8790, id: Identifier("x_mark_dec"), ctx: Load })], keywords: [] }), type_comment: None }), Return(StmtReturn { range: 8804..8842, value: Some(Subscript(ExprSubscript { range: 8811..8842, value: Name(ExprName { range: 8811..8818, id: Identifier("dec_out"), ctx: Load }), slice: Tuple(ExprTuple { range: 8819..8841, elts: [Slice(ExprSlice { range: 8819..8820, lower: None, upper: None, step: None }), Slice(ExprSlice { range: 8822..8838, lower: Some(UnaryOp(ExprUnaryOp { range: 8822..8836, op: USub, operand: Attribute(ExprAttribute { range: 8823..8836, value: Name(ExprName { range: 8823..8827, id: Identifier("self"), ctx: Load }), attr: Identifier("pred_len"), ctx: Load }) })), upper: None, step: None }), Slice(ExprSlice { range: 8840..8841, lower: None, upper: None, step: None })], ctx: Load }), ctx: Load })) })], orelse: [] }) */
        /* TODO: If(StmtIf { range: 8864..9007, test: Compare(ExprCompare { range: 8867..8897, left: Attribute(ExprAttribute { range: 8867..8881, value: Name(ExprName { range: 8867..8871, id: Identifier("self"), ctx: Load }), attr: Identifier("task_name"), ctx: Load }), ops: [Eq], comparators: [Constant(ExprConstant { range: 8885..8897, value: Str("imputation"), kind: None })] }), body: [Assign(StmtAssign { range: 8911..8980, targets: [Name(ExprName { range: 8911..8918, id: Identifier("dec_out"), ctx: Store })], value: Call(ExprCall { range: 8921..8980, func: Attribute(ExprAttribute { range: 8921..8936, value: Name(ExprName { range: 8921..8925, id: Identifier("self"), ctx: Load }), attr: Identifier("imputation"), ctx: Load }), args: [Name(ExprName { range: 8937..8942, id: Identifier("x_enc"), ctx: Load }), Name(ExprName { range: 8944..8954, id: Identifier("x_mark_enc"), ctx: Load }), Name(ExprName { range: 8956..8961, id: Identifier("x_dec"), ctx: Load }), Name(ExprName { range: 8963..8973, id: Identifier("x_mark_dec"), ctx: Load }), Name(ExprName { range: 8975..8979, id: Identifier("mask"), ctx: Load })], keywords: [] }), type_comment: None }), Return(StmtReturn { range: 8993..9007, value: Some(Name(ExprName { range: 9000..9007, id: Identifier("dec_out"), ctx: Load })) })], orelse: [] }) */
        /* TODO: If(StmtIf { range: 9029..9149, test: Compare(ExprCompare { range: 9032..9069, left: Attribute(ExprAttribute { range: 9032..9046, value: Name(ExprName { range: 9032..9036, id: Identifier("self"), ctx: Load }), attr: Identifier("task_name"), ctx: Load }), ops: [Eq], comparators: [Constant(ExprConstant { range: 9050..9069, value: Str("anomaly_detection"), kind: None })] }), body: [Assign(StmtAssign { range: 9083..9122, targets: [Name(ExprName { range: 9083..9090, id: Identifier("dec_out"), ctx: Store })], value: Call(ExprCall { range: 9093..9122, func: Attribute(ExprAttribute { range: 9093..9115, value: Name(ExprName { range: 9093..9097, id: Identifier("self"), ctx: Load }), attr: Identifier("anomaly_detection"), ctx: Load }), args: [Name(ExprName { range: 9116..9121, id: Identifier("x_enc"), ctx: Load })], keywords: [] }), type_comment: None }), Return(StmtReturn { range: 9135..9149, value: Some(Name(ExprName { range: 9142..9149, id: Identifier("dec_out"), ctx: Load })) })], orelse: [] }) */
        /* TODO: If(StmtIf { range: 9171..9297, test: Compare(ExprCompare { range: 9174..9208, left: Attribute(ExprAttribute { range: 9174..9188, value: Name(ExprName { range: 9174..9178, id: Identifier("self"), ctx: Load }), attr: Identifier("task_name"), ctx: Load }), ops: [Eq], comparators: [Constant(ExprConstant { range: 9192..9208, value: Str("classification"), kind: None })] }), body: [Assign(StmtAssign { range: 9222..9270, targets: [Name(ExprName { range: 9222..9229, id: Identifier("dec_out"), ctx: Store })], value: Call(ExprCall { range: 9232..9270, func: Attribute(ExprAttribute { range: 9232..9251, value: Name(ExprName { range: 9232..9236, id: Identifier("self"), ctx: Load }), attr: Identifier("classification"), ctx: Load }), args: [Name(ExprName { range: 9252..9257, id: Identifier("x_enc"), ctx: Load }), Name(ExprName { range: 9259..9269, id: Identifier("x_mark_enc"), ctx: Load })], keywords: [] }), type_comment: None }), Return(StmtReturn { range: 9283..9297, value: Some(Name(ExprName { range: 9290..9297, id: Identifier("dec_out"), ctx: Load })) })], orelse: [] }) */
        None
    }
}
