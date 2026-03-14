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
        let linear = LinearConfig::new(nf, target_window).init(device);
        let dropout = DropoutConfig::new(head_dropout).init();
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
#[derive(Config, Debug)]
pub struct EnEmbeddingConfig {
    pub n_vars:usize ,
    pub d_model:usize ,
    pub patch_len:usize ,
    pub dropout:usize ,
#[config(
 default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}")]
    pub initializer: Initializer,
}

impl EnEmbeddingConfig {
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> EnEmbedding<B> {
        /* TODO: super(EnEmbedding, self).__init__() */
        let patch_len = patch_len;
        let value_embedding = LinearConfig::new(patch_len, d_model).bias(false).init(device);
        let glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model));
        let position_embedding = PositionalEmbedding(d_model);
        let dropout = DropoutConfig::new(dropout).init();
EnEmbedding {
            patch_len,
            value_embedding,
            glb_token,
            position_embedding,
            dropout,
    }
}
}
struct EnEmbedding<B: Backend> {
    patch_len:,
    value_embedding:,
    glb_token:,
    position_embedding:,
    dropout:,
}
impl<B: Backend> Model<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let n_vars = x.shape[1];
        let glb = self.glb_token.repeat_dim(/* TODO: (x.shape[0], 1, 1, 1) */);
        let x = x.unfold(-1, self.patch_len, self.patch_len);
        let x = torch.reshape([x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3])]);
        let x = self.value_embedding.forward(x) + self.position_embedding.forward(x);
        let x = torch.reshape([x, (-1, n_vars, x.shape[-2], x.shape[-1])]);
        let x = Tensor::cat([x, glb].to_vec(), 2);
        let x = torch.reshape([x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3])]);
        (self.dropout.forward(x), n_vars)
    }
}
#[derive(Config, Debug)]
pub struct EncoderConfig {
    pub layers:usize ,
    pub norm_layer:usize ,
    pub projection:usize ,
#[config(
 default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}")]
    pub initializer: Initializer,
}

impl EncoderConfig {
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Encoder<B> {
        /* TODO: super(Encoder, self).__init__() */
        let layers = nn.ModuleList(layers);
        let norm = norm_layer;
        let projection = projection;
Encoder {
            layers,
            norm,
            projection,
    }
}
}
struct Encoder<B: Backend> {
    layers:,
    norm:,
    projection:,
}
impl<B: Backend> Model<B> {
    pub fn forward(&self, x: Tensor<B, 3>, cross: Tensor<B, 3>, x_mask: Tensor<B, 3>, cross_mask: Tensor<B, 3>, tau: Tensor<B, 3>, delta: Tensor<B, 3>) -> Tensor<B, 3> {
        /* TODO: For(StmtFor { range: 2092..2209, target: Name(ExprName { range: 2096..2101, id: Identifier("layer"), ctx: Store }), iter: Attribute(ExprAttribute { range: 2105..2116, value: Name(ExprName { range: 2105..2109, id: Identifier("self"), ctx: Load }), attr: Identifier("layers"), ctx: Load }), body: [Assign(StmtAssign { range: 2130..2209, targets: [Name(ExprName { range: 2130..2131, id: Identifier("x"), ctx: Store })], value: Call(ExprCall { range: 2134..2209, func: Name(ExprName { range: 2134..2139, id: Identifier("layer"), ctx: Load }), args: [Name(ExprName { range: 2140..2141, id: Identifier("x"), ctx: Load }), Name(ExprName { range: 2143..2148, id: Identifier("cross"), ctx: Load })], keywords: [Keyword { range: 2150..2163, arg: Some(Identifier("x_mask")), value: Name(ExprName { range: 2157..2163, id: Identifier("x_mask"), ctx: Load }) }, Keyword { range: 2165..2186, arg: Some(Identifier("cross_mask")), value: Name(ExprName { range: 2176..2186, id: Identifier("cross_mask"), ctx: Load }) }, Keyword { range: 2188..2195, arg: Some(Identifier("tau")), value: Name(ExprName { range: 2192..2195, id: Identifier("tau"), ctx: Load }) }, Keyword { range: 2197..2208, arg: Some(Identifier("delta")), value: Name(ExprName { range: 2203..2208, id: Identifier("delta"), ctx: Load }) }] }), type_comment: None })], orelse: [], type_comment: None }) */
        /* TODO: If(StmtIf { range: 2219..2273, test: Compare(ExprCompare { range: 2222..2243, left: Attribute(ExprAttribute { range: 2222..2231, value: Name(ExprName { range: 2222..2226, id: Identifier("self"), ctx: Load }), attr: Identifier("norm"), ctx: Load }), ops: [IsNot], comparators: [Constant(ExprConstant { range: 2239..2243, value: None, kind: None })] }), body: [Assign(StmtAssign { range: 2257..2273, targets: [Name(ExprName { range: 2257..2258, id: Identifier("x"), ctx: Store })], value: Call(ExprCall { range: 2261..2273, func: Attribute(ExprAttribute { range: 2261..2270, value: Name(ExprName { range: 2261..2265, id: Identifier("self"), ctx: Load }), attr: Identifier("norm"), ctx: Load }), args: [Name(ExprName { range: 2271..2272, id: Identifier("x"), ctx: Load })], keywords: [] }), type_comment: None })], orelse: [] }) */
        /* TODO: If(StmtIf { range: 2283..2349, test: Compare(ExprCompare { range: 2286..2313, left: Attribute(ExprAttribute { range: 2286..2301, value: Name(ExprName { range: 2286..2290, id: Identifier("self"), ctx: Load }), attr: Identifier("projection"), ctx: Load }), ops: [IsNot], comparators: [Constant(ExprConstant { range: 2309..2313, value: None, kind: None })] }), body: [Assign(StmtAssign { range: 2327..2349, targets: [Name(ExprName { range: 2327..2328, id: Identifier("x"), ctx: Store })], value: Call(ExprCall { range: 2331..2349, func: Attribute(ExprAttribute { range: 2331..2346, value: Name(ExprName { range: 2331..2335, id: Identifier("self"), ctx: Load }), attr: Identifier("projection"), ctx: Load }), args: [Name(ExprName { range: 2347..2348, id: Identifier("x"), ctx: Load })], keywords: [] }), type_comment: None })], orelse: [] }) */
        x
    }
}
#[derive(Config, Debug)]
pub struct EncoderLayerConfig {
    pub self_attention:usize ,
    pub cross_attention:usize ,
    pub d_model:usize ,
    pub d_ff:usize ,
    pub dropout:usize ,
    pub activation:usize ,
#[config(
 default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}")]
    pub initializer: Initializer,
}

impl EncoderLayerConfig {
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> EncoderLayer<B> {
        /* TODO: super(EncoderLayer, self).__init__() */
        let d_ff = /* TODO: ... */;
        let self_attention = self_attention;
        let cross_attention = cross_attention;
        let conv1 = Conv1dConfig::new().init(device);
        let conv2 = Conv1dConfig::new().init(device);
        let norm1 = LayerNormConfig::new(d_model).init(device);
        let norm2 = LayerNormConfig::new(d_model).init(device);
        let norm3 = LayerNormConfig::new(d_model).init(device);
        let dropout = DropoutConfig::new(dropout).init();
        let activation = if /* TODO: ... */ { F.relu } else { F.gelu };
EncoderLayer {
            self_attention,
            cross_attention,
            conv1,
            conv2,
            norm1,
            norm2,
            norm3,
            dropout,
            activation,
    }
}
}
struct EncoderLayer<B: Backend> {
    self_attention:,
    cross_attention:,
    conv1:,
    conv2:,
    norm1:,
    norm2:,
    norm3:,
    dropout:,
    activation:,
}
impl<B: Backend> Model<B> {
    pub fn forward(&self, x: Tensor<B, 3>, cross: Tensor<B, 3>, x_mask: Tensor<B, 3>, cross_mask: Tensor<B, 3>, tau: Tensor<B, 3>, delta: Tensor<B, 3>) -> Tensor<B, 3> {
        let (B, L, D) = cross.shape;
        let x = x + self.dropout.forward(self.self_attention.forward(x, x, x)[0]);
        let x = self.norm1.forward(x);
        let x_glb_ori = x[(/* TODO: ... */, -1, /* TODO: ... */)].unsqueeze_dim(1);
        let x_glb = torch.reshape([x_glb_ori, (B, -1, D)]);
        let x_glb_attn = self.dropout.forward(self.cross_attention.forward(x_glb, cross, cross)[0]);
        let x_glb_attn = torch.reshape([x_glb_attn, (x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2])]).unsqueeze_dim(1);
        let x_glb = x_glb_ori + x_glb_attn;
        let x_glb = self.norm2.forward(x_glb);
        let y, x = Tensor::cat([x[(/* TODO: ... */, /* TODO: ... */, /* TODO: ... */)], x_glb].to_vec(), 1);
        let y = self.dropout.forward(self.activation.forward(self.conv1.forward(y.swap_dims(-1, 1))));
        let y = self.dropout.forward(self.conv2.forward(y).swap_dims(-1, 1));
        self.norm3.forward(x + y)
    }
}
#[derive(Debug, Clone, Deserialize, Serialize, Args)]
pub struct ModelArgs {
    #[arg(long, default_value_t = )]
    pub task_name:usize ,
    #[arg(long, default_value_t = )]
    pub features:usize ,
    #[arg(long, default_value_t = )]
    pub seq_len:usize ,
    #[arg(long, default_value_t = )]
    pub pred_len:usize ,
    #[arg(long, default_value_t = )]
    pub use_norm:usize ,
    #[arg(long, default_value_t = )]
    pub patch_len:usize ,
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
        /* TODO: super(Model, self).__init__() */
        let task_name = self.model_args.task_name;
        let features = self.model_args.features;
        let seq_len = self.model_args.seq_len;
        let pred_len = self.model_args.pred_len;
        let use_norm = self.model_args.use_norm;
        let patch_len = self.model_args.patch_len;
        let patch_num = int(configs.seq_len / configs.patch_len);
        let n_vars = if /* TODO: ... */ { 1 } else { configs.enc_in };
        let en_embedding = EnEmbedding(self.n_vars, configs.d_model, self.patch_len, configs.dropout);
        let ex_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout);
        let encoder = Encoder(/* TODO: ... */);
        let head_nf = configs.d_model * self.patch_num + 1;
        let head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len);
        Model {
            task_name,
            features,
            seq_len,
            pred_len,
            use_norm,
            patch_len,
            patch_num,
            n_vars,
            en_embedding,
            ex_embedding,
            encoder,
            head_nf,
            head,
    }
}
}
impl<B: Backend> Model<B> {
    pub fn forecast(&self, x_enc: Tensor<B, 3>, x_mark_enc: Tensor<B, 3>, x_dec: Tensor<B, 3>, x_mark_dec: Tensor<B, 3>) -> Tensor<B, 3> {
        /* TODO: If(StmtIf { range: 6131..6419, test: Attribute(ExprAttribute { range: 6134..6147, value: Name(ExprName { range: 6134..6138, id: Identifier("self"), ctx: Load }), attr: Identifier("use_norm"), ctx: Load }), body: [Assign(StmtAssign { range: 6221..6265, targets: [Name(ExprName { range: 6221..6226, id: Identifier("means"), ctx: Store })], value: Call(ExprCall { range: 6229..6265, func: Attribute(ExprAttribute { range: 6229..6263, value: Call(ExprCall { range: 6229..6256, func: Attribute(ExprAttribute { range: 6229..6239, value: Name(ExprName { range: 6229..6234, id: Identifier("x_enc"), ctx: Load }), attr: Identifier("mean"), ctx: Load }), args: [Constant(ExprConstant { range: 6240..6241, value: Int(1), kind: None })], keywords: [Keyword { range: 6243..6255, arg: Some(Identifier("keepdim")), value: Constant(ExprConstant { range: 6251..6255, value: Bool(true), kind: None }) }] }), attr: Identifier("detach"), ctx: Load }), args: [], keywords: [] }), type_comment: None }), Assign(StmtAssign { range: 6278..6299, targets: [Name(ExprName { range: 6278..6283, id: Identifier("x_enc"), ctx: Store })], value: BinOp(ExprBinOp { range: 6286..6299, left: Name(ExprName { range: 6286..6291, id: Identifier("x_enc"), ctx: Load }), op: Sub, right: Name(ExprName { range: 6294..6299, id: Identifier("means"), ctx: Load }) }), type_comment: None }), Assign(StmtAssign { range: 6312..6392, targets: [Name(ExprName { range: 6312..6317, id: Identifier("stdev"), ctx: Store })], value: Call(ExprCall { range: 6320..6392, func: Attribute(ExprAttribute { range: 6320..6330, value: Name(ExprName { range: 6320..6325, id: Identifier("torch"), ctx: Load }), attr: Identifier("sqrt"), ctx: Load }), args: [BinOp(ExprBinOp { range: 6331..6391, left: Call(ExprCall { range: 6331..6384, func: Attribute(ExprAttribute { range: 6331..6340, value: Name(ExprName { range: 6331..6336, id: Identifier("torch"), ctx: Load }), attr: Identifier("var"), ctx: Load }), args: [Name(ExprName { range: 6341..6346, id: Identifier("x_enc"), ctx: Load })], keywords: [Keyword { range: 6348..6353, arg: Some(Identifier("dim")), value: Constant(ExprConstant { range: 6352..6353, value: Int(1), kind: None }) }, Keyword { range: 6355..6367, arg: Some(Identifier("keepdim")), value: Constant(ExprConstant { range: 6363..6367, value: Bool(true), kind: None }) }, Keyword { range: 6369..6383, arg: Some(Identifier("unbiased")), value: Constant(ExprConstant { range: 6378..6383, value: Bool(false), kind: None }) }] }), op: Add, right: Constant(ExprConstant { range: 6387..6391, value: Float(1e-5), kind: None }) })], keywords: [] }), type_comment: None }), AugAssign(StmtAugAssign { range: 6405..6419, target: Name(ExprName { range: 6405..6410, id: Identifier("x_enc"), ctx: Store }), op: Div, value: Name(ExprName { range: 6414..6419, id: Identifier("stdev"), ctx: Load }) })], orelse: [] }) */
        let (_, _, N) = x_enc.shape;
        let (en_embed, n_vars) = self.en_embedding.forward(x_enc[(/* TODO: ... */, /* TODO: ... */, -1)].unsqueeze_dim(-1).permute([0, 2, 1]));
        let ex_embed = self.ex_embedding.forward(x_enc[(/* TODO: ... */, /* TODO: ... */, /* TODO: ... */)], x_mark_enc);
        let enc_out = self.encoder.forward(en_embed, ex_embed);
        let enc_out = torch.reshape([enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])]);
        let enc_out = enc_out.permute([0, 1, 3, 2]);
        let dec_out = self.head.forward(enc_out);
        let dec_out = dec_out.permute([0, 2, 1]);
        /* TODO: If(StmtIf { range: 6989..7253, test: Attribute(ExprAttribute { range: 6992..7005, value: Name(ExprName { range: 6992..6996, id: Identifier("self"), ctx: Load }), attr: Identifier("use_norm"), ctx: Load }), body: [Assign(StmtAssign { range: 7082..7161, targets: [Name(ExprName { range: 7082..7089, id: Identifier("dec_out"), ctx: Store })], value: BinOp(ExprBinOp { range: 7092..7161, left: Name(ExprName { range: 7092..7099, id: Identifier("dec_out"), ctx: Load }), op: Mult, right: Call(ExprCall { range: 7103..7160, func: Attribute(ExprAttribute { range: 7103..7139, value: Call(ExprCall { range: 7103..7132, func: Attribute(ExprAttribute { range: 7103..7129, value: Subscript(ExprSubscript { range: 7103..7119, value: Name(ExprName { range: 7103..7108, id: Identifier("stdev"), ctx: Load }), slice: Tuple(ExprTuple { range: 7109..7118, elts: [Slice(ExprSlice { range: 7109..7110, lower: None, upper: None, step: None }), Constant(ExprConstant { range: 7112..7113, value: Int(0), kind: None }), Slice(ExprSlice { range: 7115..7118, lower: Some(UnaryOp(ExprUnaryOp { range: 7115..7117, op: USub, operand: Constant(ExprConstant { range: 7116..7117, value: Int(1), kind: None }) })), upper: None, step: None })], ctx: Load }), ctx: Load }), attr: Identifier("unsqueeze"), ctx: Load }), args: [Constant(ExprConstant { range: 7130..7131, value: Int(1), kind: None })], keywords: [] }), attr: Identifier("repeat"), ctx: Load }), args: [Constant(ExprConstant { range: 7140..7141, value: Int(1), kind: None }), Attribute(ExprAttribute { range: 7143..7156, value: Name(ExprName { range: 7143..7147, id: Identifier("self"), ctx: Load }), attr: Identifier("pred_len"), ctx: Load }), Constant(ExprConstant { range: 7158..7159, value: Int(1), kind: None })], keywords: [] }) }), type_comment: None }), Assign(StmtAssign { range: 7174..7253, targets: [Name(ExprName { range: 7174..7181, id: Identifier("dec_out"), ctx: Store })], value: BinOp(ExprBinOp { range: 7184..7253, left: Name(ExprName { range: 7184..7191, id: Identifier("dec_out"), ctx: Load }), op: Add, right: Call(ExprCall { range: 7195..7252, func: Attribute(ExprAttribute { range: 7195..7231, value: Call(ExprCall { range: 7195..7224, func: Attribute(ExprAttribute { range: 7195..7221, value: Subscript(ExprSubscript { range: 7195..7211, value: Name(ExprName { range: 7195..7200, id: Identifier("means"), ctx: Load }), slice: Tuple(ExprTuple { range: 7201..7210, elts: [Slice(ExprSlice { range: 7201..7202, lower: None, upper: None, step: None }), Constant(ExprConstant { range: 7204..7205, value: Int(0), kind: None }), Slice(ExprSlice { range: 7207..7210, lower: Some(UnaryOp(ExprUnaryOp { range: 7207..7209, op: USub, operand: Constant(ExprConstant { range: 7208..7209, value: Int(1), kind: None }) })), upper: None, step: None })], ctx: Load }), ctx: Load }), attr: Identifier("unsqueeze"), ctx: Load }), args: [Constant(ExprConstant { range: 7222..7223, value: Int(1), kind: None })], keywords: [] }), attr: Identifier("repeat"), ctx: Load }), args: [Constant(ExprConstant { range: 7232..7233, value: Int(1), kind: None }), Attribute(ExprAttribute { range: 7235..7248, value: Name(ExprName { range: 7235..7239, id: Identifier("self"), ctx: Load }), attr: Identifier("pred_len"), ctx: Load }), Constant(ExprConstant { range: 7250..7251, value: Int(1), kind: None })], keywords: [] }) }), type_comment: None })], orelse: [] }) */
        dec_out
    }
}
impl<B: Backend> Model<B> {
    pub fn forecast_multi(&self, x_enc: Tensor<B, 3>, x_mark_enc: Tensor<B, 3>, x_dec: Tensor<B, 3>, x_mark_dec: Tensor<B, 3>) -> Tensor<B, 3> {
        /* TODO: If(StmtIf { range: 7356..7644, test: Attribute(ExprAttribute { range: 7359..7372, value: Name(ExprName { range: 7359..7363, id: Identifier("self"), ctx: Load }), attr: Identifier("use_norm"), ctx: Load }), body: [Assign(StmtAssign { range: 7446..7490, targets: [Name(ExprName { range: 7446..7451, id: Identifier("means"), ctx: Store })], value: Call(ExprCall { range: 7454..7490, func: Attribute(ExprAttribute { range: 7454..7488, value: Call(ExprCall { range: 7454..7481, func: Attribute(ExprAttribute { range: 7454..7464, value: Name(ExprName { range: 7454..7459, id: Identifier("x_enc"), ctx: Load }), attr: Identifier("mean"), ctx: Load }), args: [Constant(ExprConstant { range: 7465..7466, value: Int(1), kind: None })], keywords: [Keyword { range: 7468..7480, arg: Some(Identifier("keepdim")), value: Constant(ExprConstant { range: 7476..7480, value: Bool(true), kind: None }) }] }), attr: Identifier("detach"), ctx: Load }), args: [], keywords: [] }), type_comment: None }), Assign(StmtAssign { range: 7503..7524, targets: [Name(ExprName { range: 7503..7508, id: Identifier("x_enc"), ctx: Store })], value: BinOp(ExprBinOp { range: 7511..7524, left: Name(ExprName { range: 7511..7516, id: Identifier("x_enc"), ctx: Load }), op: Sub, right: Name(ExprName { range: 7519..7524, id: Identifier("means"), ctx: Load }) }), type_comment: None }), Assign(StmtAssign { range: 7537..7617, targets: [Name(ExprName { range: 7537..7542, id: Identifier("stdev"), ctx: Store })], value: Call(ExprCall { range: 7545..7617, func: Attribute(ExprAttribute { range: 7545..7555, value: Name(ExprName { range: 7545..7550, id: Identifier("torch"), ctx: Load }), attr: Identifier("sqrt"), ctx: Load }), args: [BinOp(ExprBinOp { range: 7556..7616, left: Call(ExprCall { range: 7556..7609, func: Attribute(ExprAttribute { range: 7556..7565, value: Name(ExprName { range: 7556..7561, id: Identifier("torch"), ctx: Load }), attr: Identifier("var"), ctx: Load }), args: [Name(ExprName { range: 7566..7571, id: Identifier("x_enc"), ctx: Load })], keywords: [Keyword { range: 7573..7578, arg: Some(Identifier("dim")), value: Constant(ExprConstant { range: 7577..7578, value: Int(1), kind: None }) }, Keyword { range: 7580..7592, arg: Some(Identifier("keepdim")), value: Constant(ExprConstant { range: 7588..7592, value: Bool(true), kind: None }) }, Keyword { range: 7594..7608, arg: Some(Identifier("unbiased")), value: Constant(ExprConstant { range: 7603..7608, value: Bool(false), kind: None }) }] }), op: Add, right: Constant(ExprConstant { range: 7612..7616, value: Float(1e-5), kind: None }) })], keywords: [] }), type_comment: None }), AugAssign(StmtAugAssign { range: 7630..7644, target: Name(ExprName { range: 7630..7635, id: Identifier("x_enc"), ctx: Store }), op: Div, value: Name(ExprName { range: 7639..7644, id: Identifier("stdev"), ctx: Load }) })], orelse: [] }) */
        let (_, _, N) = x_enc.shape;
        let (en_embed, n_vars) = self.en_embedding.forward(x_enc.permute([0, 2, 1]));
        let ex_embed = self.ex_embedding.forward(x_enc, x_mark_enc);
        let enc_out = self.encoder.forward(en_embed, ex_embed);
        let enc_out = torch.reshape([enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])]);
        let enc_out = enc_out.permute([0, 1, 3, 2]);
        let dec_out = self.head.forward(enc_out);
        let dec_out = dec_out.permute([0, 2, 1]);
        /* TODO: If(StmtIf { range: 8179..8439, test: Attribute(ExprAttribute { range: 8182..8195, value: Name(ExprName { range: 8182..8186, id: Identifier("self"), ctx: Load }), attr: Identifier("use_norm"), ctx: Load }), body: [Assign(StmtAssign { range: 8272..8349, targets: [Name(ExprName { range: 8272..8279, id: Identifier("dec_out"), ctx: Store })], value: BinOp(ExprBinOp { range: 8282..8349, left: Name(ExprName { range: 8282..8289, id: Identifier("dec_out"), ctx: Load }), op: Mult, right: Call(ExprCall { range: 8293..8348, func: Attribute(ExprAttribute { range: 8293..8327, value: Call(ExprCall { range: 8293..8320, func: Attribute(ExprAttribute { range: 8293..8317, value: Subscript(ExprSubscript { range: 8293..8307, value: Name(ExprName { range: 8293..8298, id: Identifier("stdev"), ctx: Load }), slice: Tuple(ExprTuple { range: 8299..8306, elts: [Slice(ExprSlice { range: 8299..8300, lower: None, upper: None, step: None }), Constant(ExprConstant { range: 8302..8303, value: Int(0), kind: None }), Slice(ExprSlice { range: 8305..8306, lower: None, upper: None, step: None })], ctx: Load }), ctx: Load }), attr: Identifier("unsqueeze"), ctx: Load }), args: [Constant(ExprConstant { range: 8318..8319, value: Int(1), kind: None })], keywords: [] }), attr: Identifier("repeat"), ctx: Load }), args: [Constant(ExprConstant { range: 8328..8329, value: Int(1), kind: None }), Attribute(ExprAttribute { range: 8331..8344, value: Name(ExprName { range: 8331..8335, id: Identifier("self"), ctx: Load }), attr: Identifier("pred_len"), ctx: Load }), Constant(ExprConstant { range: 8346..8347, value: Int(1), kind: None })], keywords: [] }) }), type_comment: None }), Assign(StmtAssign { range: 8362..8439, targets: [Name(ExprName { range: 8362..8369, id: Identifier("dec_out"), ctx: Store })], value: BinOp(ExprBinOp { range: 8372..8439, left: Name(ExprName { range: 8372..8379, id: Identifier("dec_out"), ctx: Load }), op: Add, right: Call(ExprCall { range: 8383..8438, func: Attribute(ExprAttribute { range: 8383..8417, value: Call(ExprCall { range: 8383..8410, func: Attribute(ExprAttribute { range: 8383..8407, value: Subscript(ExprSubscript { range: 8383..8397, value: Name(ExprName { range: 8383..8388, id: Identifier("means"), ctx: Load }), slice: Tuple(ExprTuple { range: 8389..8396, elts: [Slice(ExprSlice { range: 8389..8390, lower: None, upper: None, step: None }), Constant(ExprConstant { range: 8392..8393, value: Int(0), kind: None }), Slice(ExprSlice { range: 8395..8396, lower: None, upper: None, step: None })], ctx: Load }), ctx: Load }), attr: Identifier("unsqueeze"), ctx: Load }), args: [Constant(ExprConstant { range: 8408..8409, value: Int(1), kind: None })], keywords: [] }), attr: Identifier("repeat"), ctx: Load }), args: [Constant(ExprConstant { range: 8418..8419, value: Int(1), kind: None }), Attribute(ExprAttribute { range: 8421..8434, value: Name(ExprName { range: 8421..8425, id: Identifier("self"), ctx: Load }), attr: Identifier("pred_len"), ctx: Load }), Constant(ExprConstant { range: 8436..8437, value: Int(1), kind: None })], keywords: [] }) }), type_comment: None })], orelse: [] }) */
        dec_out
    }
}
impl<B: Backend> Model<B> {
    pub fn forward(&self, x_enc: Tensor<B, 3>, x_mark_enc: Tensor<B, 3>, x_dec: Tensor<B, 3>, x_mark_dec: Tensor<B, 3>, mask: Tensor<B, 3>) -> Tensor<B, 3> {
        /* TODO: If(StmtIf { range: 8545..9019, test: BoolOp(ExprBoolOp { range: 8548..8629, op: Or, values: [Compare(ExprCompare { range: 8548..8586, left: Attribute(ExprAttribute { range: 8548..8562, value: Name(ExprName { range: 8548..8552, id: Identifier("self"), ctx: Load }), attr: Identifier("task_name"), ctx: Load }), ops: [Eq], comparators: [Constant(ExprConstant { range: 8566..8586, value: Str("long_term_forecast"), kind: None })] }), Compare(ExprCompare { range: 8590..8629, left: Attribute(ExprAttribute { range: 8590..8604, value: Name(ExprName { range: 8590..8594, id: Identifier("self"), ctx: Load }), attr: Identifier("task_name"), ctx: Load }), ops: [Eq], comparators: [Constant(ExprConstant { range: 8608..8629, value: Str("short_term_forecast"), kind: None })] })] }), body: [If(StmtIf { range: 8643..8968, test: Compare(ExprCompare { range: 8646..8666, left: Attribute(ExprAttribute { range: 8646..8659, value: Name(ExprName { range: 8646..8650, id: Identifier("self"), ctx: Load }), attr: Identifier("features"), ctx: Load }), ops: [Eq], comparators: [Constant(ExprConstant { range: 8663..8666, value: Str("M"), kind: None })] }), body: [Assign(StmtAssign { range: 8684..8751, targets: [Name(ExprName { range: 8684..8691, id: Identifier("dec_out"), ctx: Store })], value: Call(ExprCall { range: 8694..8751, func: Attribute(ExprAttribute { range: 8694..8713, value: Name(ExprName { range: 8694..8698, id: Identifier("self"), ctx: Load }), attr: Identifier("forecast_multi"), ctx: Load }), args: [Name(ExprName { range: 8714..8719, id: Identifier("x_enc"), ctx: Load }), Name(ExprName { range: 8721..8731, id: Identifier("x_mark_enc"), ctx: Load }), Name(ExprName { range: 8733..8738, id: Identifier("x_dec"), ctx: Load }), Name(ExprName { range: 8740..8750, id: Identifier("x_mark_dec"), ctx: Load })], keywords: [] }), type_comment: None }), Return(StmtReturn { range: 8768..8805, value: Some(Subscript(ExprSubscript { range: 8775..8805, value: Name(ExprName { range: 8775..8782, id: Identifier("dec_out"), ctx: Load }), slice: Tuple(ExprTuple { range: 8783..8804, elts: [Slice(ExprSlice { range: 8783..8784, lower: None, upper: None, step: None }), Slice(ExprSlice { range: 8786..8801, lower: Some(UnaryOp(ExprUnaryOp { range: 8786..8800, op: USub, operand: Attribute(ExprAttribute { range: 8787..8800, value: Name(ExprName { range: 8787..8791, id: Identifier("self"), ctx: Load }), attr: Identifier("pred_len"), ctx: Load }) })), upper: None, step: None }), Slice(ExprSlice { range: 8803..8804, lower: None, upper: None, step: None })], ctx: Load }), ctx: Load })) })], orelse: [Assign(StmtAssign { range: 8853..8914, targets: [Name(ExprName { range: 8853..8860, id: Identifier("dec_out"), ctx: Store })], value: Call(ExprCall { range: 8863..8914, func: Attribute(ExprAttribute { range: 8863..8876, value: Name(ExprName { range: 8863..8867, id: Identifier("self"), ctx: Load }), attr: Identifier("forecast"), ctx: Load }), args: [Name(ExprName { range: 8877..8882, id: Identifier("x_enc"), ctx: Load }), Name(ExprName { range: 8884..8894, id: Identifier("x_mark_enc"), ctx: Load }), Name(ExprName { range: 8896..8901, id: Identifier("x_dec"), ctx: Load }), Name(ExprName { range: 8903..8913, id: Identifier("x_mark_dec"), ctx: Load })], keywords: [] }), type_comment: None }), Return(StmtReturn { range: 8931..8968, value: Some(Subscript(ExprSubscript { range: 8938..8968, value: Name(ExprName { range: 8938..8945, id: Identifier("dec_out"), ctx: Load }), slice: Tuple(ExprTuple { range: 8946..8967, elts: [Slice(ExprSlice { range: 8946..8947, lower: None, upper: None, step: None }), Slice(ExprSlice { range: 8949..8964, lower: Some(UnaryOp(ExprUnaryOp { range: 8949..8963, op: USub, operand: Attribute(ExprAttribute { range: 8950..8963, value: Name(ExprName { range: 8950..8954, id: Identifier("self"), ctx: Load }), attr: Identifier("pred_len"), ctx: Load }) })), upper: None, step: None }), Slice(ExprSlice { range: 8966..8967, lower: None, upper: None, step: None })], ctx: Load }), ctx: Load })) })] })], orelse: [Return(StmtReturn { range: 9008..9019, value: Some(Constant(ExprConstant { range: 9015..9019, value: None, kind: None })) })] }) */
    }
}
