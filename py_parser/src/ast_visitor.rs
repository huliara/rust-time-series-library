use crate::layer_map::{LayerInfo, fn_table, layer_table};
use anyhow::{Result, bail};
use rustpython_parser::ast::*;
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Data structures
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ModelField {
    pub name: String,
    pub layer_info: LayerInfo,
    pub init_expr: String, // Burn の Config::new(...).init(device) 形式
}

#[derive(Debug, Default, Clone)]
pub struct ModelInfo {
    pub class_name: String,
    pub fields: Vec<ModelField>,
    pub forward_args: Vec<String>,
    pub forward_stmts: Vec<String>,
    pub return_expr: Option<String>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────────────────

pub fn extract_models(source: &str) -> Result<Vec<ModelInfo>> {
    let ast = rustpython_parser::parse(source, rustpython_parser::Mode::Module, "<input>")?;
    let Mod::Module(module) = ast else {
        bail!("expected Module");
    };

    let mut models = Vec::new();
    for stmt in &module.body {
        if let Stmt::ClassDef(cls) = stmt
            && is_nn_module(cls)
            && let Some(info) = extract_class(cls)
        {
            models.push(info);
        }
    }
    Ok(models)
}

// ─────────────────────────────────────────────────────────────────────────────
// Class-level helpers
// ─────────────────────────────────────────────────────────────────────────────

fn is_nn_module(cls: &StmtClassDef) -> bool {
    cls.bases.iter().any(|base| {
        let s = expr_to_raw(base);
        matches!(
            s.as_str(),
            "nn.Module" | "torch.nn.Module" | "Module" | "pl.LightningModule"
        )
    })
}

fn extract_class(cls: &StmtClassDef) -> Option<ModelInfo> {
    let mut info = ModelInfo {
        class_name: cls.name.to_string(),
        ..Default::default()
    };
    let table = layer_table();

    let mut has_init = false;
    for stmt in &cls.body {
        match stmt {
            Stmt::FunctionDef(f) if f.name.as_str() == "__init__" => {
                has_init = true;
                extract_init(f, &table, &mut info);
            }
            Stmt::FunctionDef(f) if f.name.as_str() == "forward" => {
                let fields_snap: Vec<ModelField> = info.fields.clone();
                extract_forward(f, &fields_snap, &table, &mut info);
            }
            _ => {}
        }
    }

    if !has_init {
        return None;
    }
    Some(info)
}

// ─────────────────────────────────────────────────────────────────────────────
// __init__ extraction
// ─────────────────────────────────────────────────────────────────────────────

fn extract_init(func: &StmtFunctionDef, table: &HashMap<&str, LayerInfo>, info: &mut ModelInfo) {
    for stmt in &func.body {
        if let Stmt::Assign(assign) = stmt {
            // self.xxx = nn.Yyy(...)
            if assign.targets.len() != 1 {
                continue;
            }
            let target = &assign.targets[0];
            let Some(field_name) = self_attr_name(target) else {
                continue;
            };
            let Some((layer_key, call)) = extract_nn_call(&assign.value) else {
                continue;
            };
            let Some(layer_info) = table.get(layer_key.as_str()).cloned() else {
                // Unknown layer – emit a comment
                info.fields.push(ModelField {
                    name: field_name,
                    layer_info: LayerInfo {
                        burn_type: "/* unknown */",
                        burn_import: "",
                        config_type: "",
                        needs_device: false,
                        needs_backend: false,
                        is_module: true,
                    },
                    init_expr: format!("/* TODO: {} */", expr_to_raw(&assign.value)),
                });
                continue;
            };
            let init_expr = build_init_expr(&layer_info, &layer_key, call);
            info.fields.push(ModelField {
                name: field_name,
                layer_info,
                init_expr,
            });
        }
    }
}

/// nn.XXX(...) / torch.nn.XXX(...) の `(layer_key, call)` を返す
fn extract_nn_call(expr: &Expr) -> Option<(String, &ExprCall)> {
    let Expr::Call(call) = expr else {
        return None;
    };
    let func_str = expr_to_raw(&call.func);
    // "nn.Linear" / "torch.nn.Linear" / "Linear" など
    for prefix in &["nn.", "torch.nn.", ""] {
        if let Some(rest) = func_str.strip_prefix(prefix) {
            // rest が table のキーになる
            if !rest.is_empty() && !rest.contains('.') {
                return Some((rest.to_string(), call));
            }
        }
    }
    None
}

/// Config::new(...).init(device) 形式の文字列を組み立てる
fn build_init_expr(info: &LayerInfo, layer_key: &str, call: &ExprCall) -> String {
    let args: Vec<String> = call.args.iter().map(expr_to_burn).collect();

    // Config::new の引数マッピング (先に計算して consumed_kwargs を確定する)
    // consumed_kwargs: Config::new() の引数として使ったキーワード名 → chain から除外
    let consumed_kwargs: &[&str] = match layer_key {
        "Dropout" | "AlphaDropout" => &["p"],
        "LeakyReLU" => &["negative_slope"],
        "Conv1d" | "Conv2d" | "ConvTranspose1d" | "ConvTranspose2d" => {
            &["in_channels", "out_channels", "kernel_size"]
        }
        "BatchNorm1d" | "BatchNorm2d" | "InstanceNorm1d" | "InstanceNorm2d" => &["num_features"],
        "LayerNorm" => &["normalized_shape"],
        "GroupNorm" => &["num_groups", "num_channels"],
        "Embedding" => &["num_embeddings", "embedding_dim"],
        "LSTM" | "GRU" => &["input_size", "hidden_size"],
        "MultiheadAttention" => &["embed_dim", "num_heads"],
        _ => &[],
    };

    let kwargs: Vec<String> = call
        .keywords
        .iter()
        .filter_map(|kw| {
            let name = kw.arg.as_ref()?.as_str().to_string();
            if consumed_kwargs.contains(&name.as_str()) {
                return None; // Config::new() で使用済み
            }
            Some(format!(".{}({})", name, expr_to_burn(&kw.value)))
        })
        .collect();

    // Config::new の引数マッピング
    let config_args = match layer_key {
        "Linear" => args.join(", "),
        "Conv1d" | "Conv2d" | "ConvTranspose1d" | "ConvTranspose2d" => {
            // (in_channels, out_channels, kernel_size, ...)
            if args.len() >= 3 {
                format!("[{}, {}], [{}]", args[0], args[1], args[2])
            } else {
                args.join(", ")
            }
        }
        "Dropout" | "AlphaDropout" => {
            // p=0.5 -> first arg or keyword
            let p = if !args.is_empty() {
                args[0].clone()
            } else {
                call.keywords
                    .iter()
                    .find(|k| k.arg.as_deref() == Some("p"))
                    .map(|k| expr_to_burn(&k.value))
                    .unwrap_or_else(|| "0.5".to_string())
            };
            p
        }
        "BatchNorm1d" | "BatchNorm2d" => args.first().cloned().unwrap_or_default(),
        "LayerNorm" => {
            // normalized_shape: list or int
            if !args.is_empty() {
                let s = &args[0];
                // if it's a list literal like "[512]", strip brackets
                if s.starts_with('[') && s.ends_with(']') {
                    s[1..s.len() - 1].to_string()
                } else {
                    s.clone()
                }
            } else {
                String::new()
            }
        }
        "GroupNorm" => {
            if args.len() >= 2 {
                format!("{}, {}", args[0], args[1])
            } else {
                args.join(", ")
            }
        }
        "Embedding" => {
            if args.len() >= 2 {
                format!("{}, {}", args[0], args[1])
            } else {
                args.join(", ")
            }
        }
        "MultiheadAttention" => {
            // (embed_dim, num_heads, ...)
            if args.len() >= 2 {
                format!("{}, {}", args[0], args[1])
            } else {
                args.join(", ")
            }
        }
        "LSTM" | "GRU" => {
            // (input_size, hidden_size, ...)
            if args.len() >= 2 {
                format!("{}, {}", args[0], args[1])
            } else {
                args.join(", ")
            }
        }
        "LeakyReLU" => {
            let neg_slope = if !args.is_empty() {
                args[0].clone()
            } else {
                call.keywords
                    .iter()
                    .find(|k| k.arg.as_deref() == Some("negative_slope"))
                    .map(|k| expr_to_burn(&k.value))
                    .unwrap_or_else(|| "0.01".to_string())
            };
            neg_slope
        }
        _ => args.join(", "),
    };

    let config_name = info.config_type;
    if config_name.is_empty() {
        return format!("/* TODO: {} */", layer_key);
    }

    let mut s = format!("{}::new({})", config_name, config_args);
    for k in &kwargs {
        s.push_str(k);
    }
    if info.needs_device {
        s.push_str(".init(device)");
    } else {
        s.push_str(".init()");
    }
    s
}

// ─────────────────────────────────────────────────────────────────────────────
// forward extraction
// ─────────────────────────────────────────────────────────────────────────────

fn extract_forward(
    func: &StmtFunctionDef,
    fields: &[ModelField],
    table: &HashMap<&str, LayerInfo>,
    info: &mut ModelInfo,
) {
    // args (excluding self)
    for arg in func.args.args.iter().skip(1) {
        info.forward_args.push(arg.def.arg.to_string());
    }

    let module_fields: HashMap<String, &LayerInfo> = fields
        .iter()
        .map(|f| (f.name.clone(), &f.layer_info))
        .collect();

    for stmt in &func.body {
        match stmt {
            Stmt::Return(ret) => {
                if let Some(val) = &ret.value {
                    info.return_expr = Some(stmt_expr_to_burn(val, &module_fields, table));
                }
            }
            Stmt::Assign(assign) => {
                let target = if assign.targets.len() == 1 {
                    expr_to_raw(&assign.targets[0])
                } else {
                    assign
                        .targets
                        .iter()
                        .map(expr_to_raw)
                        .collect::<Vec<_>>()
                        .join(", ")
                };
                let rhs = stmt_expr_to_burn(&assign.value, &module_fields, table);
                info.forward_stmts
                    .push(format!("let {} = {};", target, rhs));
            }
            Stmt::AugAssign(aug) => {
                let target = expr_to_raw(&aug.target);
                let op = binop_to_burn(&aug.op);
                let rhs = stmt_expr_to_burn(&aug.value, &module_fields, table);
                info.forward_stmts
                    .push(format!("let {} = {} {} {};", target, target, op, rhs));
            }
            Stmt::Expr(e) => {
                let s = stmt_expr_to_burn(&e.value, &module_fields, table);
                info.forward_stmts.push(format!("{};", s));
            }
            _ => {
                info.forward_stmts
                    .push(format!("/* TODO: {} */", stmt_to_raw(stmt)));
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Expression conversion: Python → Burn Rust
// ─────────────────────────────────────────────────────────────────────────────

pub fn stmt_expr_to_burn(
    expr: &Expr,
    module_fields: &HashMap<String, &LayerInfo>,
    table: &HashMap<&str, LayerInfo>,
) -> String {
    let fn_map = fn_table();
    convert_expr(expr, module_fields, &fn_map, table)
}

fn convert_expr(
    expr: &Expr,
    mf: &HashMap<String, &LayerInfo>,
    fn_map: &HashMap<&str, &str>,
    table: &HashMap<&str, LayerInfo>,
) -> String {
    match expr {
        // ── constants ───────────────────────────────────────────────────────
        Expr::Constant(c) => match &c.value {
            Constant::Int(i) => i.to_string(),
            Constant::Float(f) => format!("{f}"),
            Constant::Complex { real, imag } => format!("({real} + {imag}i)"),
            Constant::Str(s) => format!("\"{}\"", s),
            Constant::Bool(b) => b.to_string(),
            Constant::None => "None".to_string(),
            Constant::Bytes(b) => format!("{:?}", b),
            _ => "...".to_string(),
        },

        // ── name (variable) ─────────────────────────────────────────────────
        Expr::Name(n) => n.id.to_string(),

        // ── attribute access ────────────────────────────────────────────────
        Expr::Attribute(attr) => {
            let owner = convert_expr(&attr.value, mf, fn_map, table);
            let field = attr.attr.as_str();
            // self.xxx → keep as self.xxx (forward() in Burn calls methods directly)
            format!("{}.{}", owner, field)
        }

        // ── function call ────────────────────────────────────────────────────
        Expr::Call(call) => convert_call(call, mf, fn_map, table),

        // ── binary operator ──────────────────────────────────────────────────
        Expr::BinOp(binop) => {
            let left = convert_expr(&binop.left, mf, fn_map, table);
            let right = convert_expr(&binop.right, mf, fn_map, table);
            match &binop.op {
                Operator::MatMult => format!("{}.matmul({})", left, right),
                Operator::Pow => format!("{}.powf({})", left, right),
                op => format!("{} {} {}", left, binop_to_burn(op), right),
            }
        }

        // ── unary operator ───────────────────────────────────────────────────
        Expr::UnaryOp(u) => {
            let operand = convert_expr(&u.operand, mf, fn_map, table);
            match &u.op {
                UnaryOp::USub => format!("-{}", operand),
                UnaryOp::UAdd => operand,
                UnaryOp::Not => format!("!{}", operand),
                UnaryOp::Invert => format!("!{}", operand),
            }
        }

        // ── tuple / list ─────────────────────────────────────────────────────
        Expr::Tuple(t) => {
            let elts: Vec<_> = t
                .elts
                .iter()
                .map(|e| convert_expr(e, mf, fn_map, table))
                .collect();
            format!("({})", elts.join(", "))
        }
        Expr::List(l) => {
            let elts: Vec<_> = l
                .elts
                .iter()
                .map(|e| convert_expr(e, mf, fn_map, table))
                .collect();
            format!("[{}]", elts.join(", "))
        }

        // ── subscript (indexing) ─────────────────────────────────────────────
        Expr::Subscript(s) => {
            let val = convert_expr(&s.value, mf, fn_map, table);
            let idx = convert_expr(&s.slice, mf, fn_map, table);
            format!("{}[{}]", val, idx)
        }

        // ── ternary (if-else) ────────────────────────────────────────────────
        Expr::IfExp(ife) => {
            let cond = convert_expr(&ife.test, mf, fn_map, table);
            let body = convert_expr(&ife.body, mf, fn_map, table);
            let orelse = convert_expr(&ife.orelse, mf, fn_map, table);
            format!("if {} {{ {} }} else {{ {} }}", cond, body, orelse)
        }

        _ => format!("/* TODO: {} */", expr_to_raw(expr)),
    }
}

fn convert_call(
    call: &ExprCall,
    mf: &HashMap<String, &LayerInfo>,
    fn_map: &HashMap<&str, &str>,
    table: &HashMap<&str, LayerInfo>,
) -> String {
    let func_raw = expr_to_raw(&call.func);

    // ── torch.cat / torch.stack ───────────────────────────────────────────
    if matches!(func_raw.as_str(), "torch.cat" | "torch.stack") {
        return convert_cat_stack(call, &func_raw, mf, fn_map, table);
    }

    // ── torch.zeros / ones / rand ─────────────────────────────────────────
    if let Some(burn_ctor) = tensor_ctor(&func_raw) {
        let args: Vec<_> = call
            .args
            .iter()
            .map(|a| convert_expr(a, mf, fn_map, table))
            .collect();
        return format!(
            "Tensor::<B, _>::{}([{}], device)",
            burn_ctor,
            args.join(", ")
        );
    }

    // ── known functional mapping (relu, sigmoid, …) ───────────────────────
    if let Some(&burn_fn) = fn_map.get(func_raw.as_str()) {
        let args: Vec<_> = call
            .args
            .iter()
            .map(|a| convert_expr(a, mf, fn_map, table))
            .collect();
        let kw_dim = call.keywords.iter().find_map(|k| {
            if k.arg.as_deref() == Some("dim") {
                Some(convert_expr(&k.value, mf, fn_map, table))
            } else {
                None
            }
        });
        if let Some(dim) = kw_dim {
            return format!("{}({}, {})", burn_fn, args.join(", "), dim);
        }
        return format!("{}({})", burn_fn, args.join(", "));
    }

    // ── method call on a variable ─────────────────────────────────────────
    if let Expr::Attribute(attr) = call.func.as_ref() {
        let receiver = convert_expr(&attr.value, mf, fn_map, table);
        let method = attr.attr.as_str();
        let args: Vec<_> = call
            .args
            .iter()
            .map(|a| convert_expr(a, mf, fn_map, table))
            .collect();
        let kwargs: Vec<_> = call
            .keywords
            .iter()
            .filter_map(|k| {
                let name = k.arg.as_ref()?.as_str().to_string();
                Some((name, convert_expr(&k.value, mf, fn_map, table)))
            })
            .collect();

        // self.layer(x)  → self.layer.forward(x)  (if layer is a module)
        if let Expr::Name(n) = attr.value.as_ref() {
            if n.id.as_str() == "self" {
                let field_name = attr.attr.as_str();
                if let Some(li) = mf.get(field_name) {
                    if li.is_module {
                        return format!("self.{}.forward({})", field_name, args.join(", "));
                    }
                }
            }
        }

        return convert_method_call(&receiver, method, &args, &kwargs);
    }

    // ── fallback ──────────────────────────────────────────────────────────
    let func_burn = convert_expr(&call.func, mf, fn_map, table);
    let args: Vec<_> = call
        .args
        .iter()
        .map(|a| convert_expr(a, mf, fn_map, table))
        .collect();
    format!("{}({})", func_burn, args.join(", "))
}

/// Tensor method calls like `.view()`, `.permute()`, `.reshape()` etc.
fn convert_method_call(
    receiver: &str,
    method: &str,
    args: &[String],
    kwargs: &[(String, String)],
) -> String {
    let dim_arg = || {
        if !args.is_empty() {
            args[0].clone()
        } else {
            kwargs
                .iter()
                .find(|(k, _)| k == "dim")
                .map(|(_, v)| v.clone())
                .unwrap_or_default()
        }
    };

    match method {
        // shape ops
        "view" | "reshape" => {
            format!("{}.reshape([{}])", receiver, args.join(", "))
        }
        "permute" => format!("{}.permute([{}])", receiver, args.join(", ")),
        "transpose" if args.len() == 2 => {
            format!("{}.swap_dims({}, {})", receiver, args[0], args[1])
        }
        "contiguous" | "clone" => format!("{}.clone()", receiver),
        "detach" => receiver.to_string(),
        "float" | "half" | "double" => {
            format!("/* TODO: cast */ {}", receiver)
        }
        "unsqueeze" => format!("{}.unsqueeze_dim({})", receiver, dim_arg()),
        "squeeze" => {
            if args.is_empty() {
                format!("{}.squeeze(/* dim */)", receiver)
            } else {
                format!("{}.squeeze({})", receiver, args[0])
            }
        }
        "mean" => {
            let dim = dim_arg();
            if dim.is_empty() {
                format!("{}.mean()", receiver)
            } else {
                format!("{}.mean_dim({})", receiver, dim)
            }
        }
        "sum" => {
            let dim = dim_arg();
            if dim.is_empty() {
                format!("{}.sum()", receiver)
            } else {
                format!("{}.sum_dim({})", receiver, dim)
            }
        }
        "max" => {
            let dim = dim_arg();
            if dim.is_empty() {
                format!("{}.max()", receiver)
            } else {
                format!("{}.max_dim({})", receiver, dim)
            }
        }
        "min" => {
            let dim = dim_arg();
            if dim.is_empty() {
                format!("{}.min()", receiver)
            } else {
                format!("{}.min_dim({})", receiver, dim)
            }
        }
        "flatten" => {
            if args.len() >= 2 {
                format!("{}.flatten({}, {})", receiver, args[0], args[1])
            } else {
                format!("{}.flatten(0, /* END_DIM */)", receiver)
            }
        }
        "expand" | "broadcast_to" => {
            format!("{}.expand([{}])", receiver, args.join(", "))
        }
        "repeat" => {
            format!("{}.repeat_dim(/* TODO: {} */)", receiver, args.join(", "))
        }
        "softmax" => {
            let dim = dim_arg();
            format!("burn::tensor::activation::softmax({}, {})", receiver, dim)
        }
        "sigmoid" => format!("burn::tensor::activation::sigmoid({})", receiver),
        "tanh" => format!("burn::tensor::activation::tanh({})", receiver),
        "relu" => format!("burn::tensor::activation::relu({})", receiver),
        "gelu" => format!("burn::tensor::activation::gelu({})", receiver),
        "size" | "shape" => {
            if args.is_empty() {
                format!("{}.dims()", receiver)
            } else {
                format!("{}.dims()[{}]", receiver, args[0])
            }
        }
        "to" | "cuda" | "cpu" => {
            format!("/* TODO: device */ {}", receiver)
        }
        _ => {
            let all_args: Vec<_> = args
                .iter()
                .chain(kwargs.iter().map(|(_, v)| v))
                .cloned()
                .collect();
            format!("{}.{}({})", receiver, method, all_args.join(", "))
        }
    }
}

fn convert_cat_stack(
    call: &ExprCall,
    func_raw: &str,
    mf: &HashMap<String, &LayerInfo>,
    fn_map: &HashMap<&str, &str>,
    table: &HashMap<&str, LayerInfo>,
) -> String {
    let tensors = if let Some(first) = call.args.first() {
        convert_expr(first, mf, fn_map, table)
    } else {
        String::from("/* tensors */")
    };
    let dim = call
        .args
        .get(1)
        .map(|a| convert_expr(a, mf, fn_map, table))
        .or_else(|| {
            call.keywords
                .iter()
                .find(|k| k.arg.as_deref() == Some("dim"))
                .map(|k| convert_expr(&k.value, mf, fn_map, table))
        })
        .unwrap_or_else(|| "0".to_string());

    if func_raw.contains("stack") {
        format!("Tensor::stack({}.to_vec(), {})", tensors, dim)
    } else {
        format!("Tensor::cat({}.to_vec(), {})", tensors, dim)
    }
}

fn tensor_ctor(name: &str) -> Option<&'static str> {
    match name {
        "torch.zeros" | "torch.zeros_like" => Some("zeros"),
        "torch.ones" | "torch.ones_like" => Some("ones"),
        "torch.rand" | "torch.rand_like" => Some("random"),
        "torch.empty" => Some("empty"),
        _ => None,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility helpers
// ─────────────────────────────────────────────────────────────────────────────

fn binop_to_burn(op: &Operator) -> &'static str {
    match op {
        Operator::Add => "+",
        Operator::Sub => "-",
        Operator::Mult => "*",
        Operator::Div => "/",
        Operator::Mod => "%",
        Operator::FloorDiv => "/",
        Operator::BitAnd => "&",
        Operator::BitOr => "|",
        Operator::BitXor => "^",
        Operator::LShift => "<<",
        Operator::RShift => ">>",
        _ => "/* op */",
    }
}

/// self.xxx → Some("xxx")
fn self_attr_name(expr: &Expr) -> Option<String> {
    let Expr::Attribute(attr) = expr else {
        return None;
    };
    let Expr::Name(n) = attr.value.as_ref() else {
        return None;
    };
    if n.id.as_str() != "self" {
        return None;
    }
    Some(attr.attr.to_string())
}

/// AST 式を生の Python 文字列として再現する (変換なし、デバッグ/フォールバック用)
pub fn expr_to_raw(expr: &Expr) -> String {
    match expr {
        Expr::Name(n) => n.id.to_string(),
        Expr::Attribute(a) => format!("{}.{}", expr_to_raw(&a.value), a.attr.as_str()),
        Expr::Constant(c) => match &c.value {
            Constant::Int(i) => i.to_string(),
            Constant::Float(f) => f.to_string(),
            Constant::Complex { real, imag } => format!("({real}+{imag}j)"),
            Constant::Str(s) => format!("\"{}\"", s),
            Constant::Bool(b) => b.to_string(),
            Constant::None => "None".to_string(),
            _ => "...".to_string(),
        },
        Expr::Call(c) => {
            let func = expr_to_raw(&c.func);
            let args: Vec<_> = c.args.iter().map(expr_to_raw).collect();
            let kwargs: Vec<_> = c
                .keywords
                .iter()
                .filter_map(|k| {
                    let name = k.arg.as_ref()?.as_str().to_string();
                    Some(format!("{}={}", name, expr_to_raw(&k.value)))
                })
                .collect();
            let all: Vec<_> = args.iter().chain(kwargs.iter()).cloned().collect();
            format!("{}({})", func, all.join(", "))
        }
        Expr::List(l) => {
            let elts: Vec<_> = l.elts.iter().map(expr_to_raw).collect();
            format!("[{}]", elts.join(", "))
        }
        Expr::Tuple(t) => {
            let elts: Vec<_> = t.elts.iter().map(expr_to_raw).collect();
            format!("({})", elts.join(", "))
        }
        Expr::BinOp(b) => {
            format!(
                "{} {:?} {}",
                expr_to_raw(&b.left),
                b.op,
                expr_to_raw(&b.right)
            )
        }
        Expr::Subscript(s) => format!("{}[{}]", expr_to_raw(&s.value), expr_to_raw(&s.slice)),
        Expr::UnaryOp(u) => format!("{:?}({})", u.op, expr_to_raw(&u.operand)),
        _ => "...".to_string(),
    }
}

/// AST Stmt を生の文字列にする (フォールバック用)
fn stmt_to_raw(stmt: &Stmt) -> String {
    match stmt {
        Stmt::Assign(a) => {
            let targets: Vec<_> = a.targets.iter().map(expr_to_raw).collect();
            format!("{} = {}", targets.join(", "), expr_to_raw(&a.value))
        }
        Stmt::Return(r) => {
            format!(
                "return {}",
                r.value.as_ref().map(|v| expr_to_raw(v)).unwrap_or_default()
            )
        }
        Stmt::Expr(e) => expr_to_raw(&e.value),
        _ => format!("{:?}", stmt),
    }
}

/// raw string から Burn 式に変換（`expr_to_raw` 経由で使う変換）
fn expr_to_burn(expr: &Expr) -> String {
    // 定数はそのまま、他はとりあえず raw で
    match expr {
        Expr::Constant(c) => match &c.value {
            Constant::Int(i) => i.to_string(),
            Constant::Float(f) => f.to_string(),
            Constant::Complex { real, imag } => format!("({real}+{imag}j)"),
            Constant::Str(s) => format!("\"{}\"", s),
            Constant::Bool(b) => b.to_string(),
            Constant::None => "None".to_string(),
            _ => "...".to_string(),
        },
        Expr::List(l) => {
            let elts: Vec<_> = l.elts.iter().map(expr_to_burn).collect();
            format!("[{}]", elts.join(", "))
        }
        Expr::Tuple(t) => {
            let elts: Vec<_> = t.elts.iter().map(expr_to_burn).collect();
            format!("[{}]", elts.join(", ")) // Vec/array 形式
        }
        _ => expr_to_raw(expr),
    }
}
