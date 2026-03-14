use crate::{
    build_forward_str::build_fn_str,
    build_main_model_config::build_main_model_config,
    build_sub_model_config::build_sub_model_config,
    layer_map::{LayerInfo, fn_table, layer_table},
};
use anyhow::{Result, bail};
use rustpython_parser::ast::*;
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────────────────

pub fn extract_models(source: &str) -> Result<String> {
    let ast = rustpython_parser::parse(source, rustpython_parser::Mode::Module, "<input>")?;
    let Mod::Module(module) = ast else {
        bail!("expected Module");
    };

    let mut models: String = String::new();
    for stmt in &module.body {
        if let Stmt::ClassDef(cls) = stmt
            && is_nn_module(cls)
            && let Some(info) = extract_class(cls)
        {
            models.push_str(&info);
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

fn extract_class(cls: &StmtClassDef) -> Option<String> {
    let mut code = String::new();
    let table = layer_table();
    let class_name = cls.name.to_string();

    let mut has_init = false;
    for stmt in &cls.body {
        match stmt {
            Stmt::FunctionDef(f) if f.name.as_str() == "__init__" => {
                has_init = true;
                code.push_str(&extract_init(&class_name, f, &table));
            }
            Stmt::FunctionDef(f) if f.name.as_str() == "forward" => {
                code.push_str(&extract_forward(f, &table))
            }
            Stmt::FunctionDef(f) => code.push_str(&extract_forward(f, &table)),
            _ => {}
        }
    }

    if !has_init {
        return None;
    }
    Some(code)
}

fn extract_config_arg(value: Expr) -> Option<String> {
    if let Expr::Attribute(attribute) = value
        && let Expr::Name(name) = *attribute.value
        && (*name.id == *"configs" || *name.id == *"config")
    {
        Some(attribute.attr.to_string())
    } else {
        None
    }
}

fn extract_value(value: Expr, table: &HashMap<&str, LayerInfo>) -> (String, Option<String>) {
    let value_str: std::string::String;

    if let Some(init_arg) = extract_config_arg(value.clone()) {
        value_str = format!("self.model_args.{}", init_arg);
        return (value_str, Some(init_arg));
    } else if let Some((layer_key, call)) = extract_nn_call(&value)
        && let Some(info) = table.get(layer_key.as_str())
    {
        value_str = build_init_expr(info, layer_key.as_str(), call);
    } else {
        value_str = convert_expr(&value, &fn_table(), table);
    };
    (value_str, None)
}

// ─────────────────────────────────────────────────────────────────────────────
// __init__ extraction
// ─────────────────────────────────────────────────────────────────────────────

fn extract_init(
    class_name: &String,
    func: &StmtFunctionDef,
    table: &HashMap<&str, LayerInfo>,
) -> String {
    let mut init_args: Vec<String> = Vec::new();
    let mut model_fields: Vec<String> = Vec::new();
    let mut body: Vec<String> = Vec::new();

    for arg in func.args.args.iter().skip(1) {
        if &arg.def.arg != "device" && &arg.def.arg != "configs" {
            init_args.push(arg.def.arg.to_string());
        }
    }

    for stmt in &func.body {
        match stmt {
            Stmt::AnnAssign(assign) => {
                let target = &assign.target;
                let value: Expr = *assign.value.clone().unwrap();
                let target_name: std::string::String;
                let (value_str, additional_init_args) = extract_value(value, table);
                if let Some(arg) = additional_init_args {
                    init_args.push(arg);
                }

                if let Some(field_name) = self_attr_name(target) {
                    model_fields.push(field_name.clone());
                    target_name = field_name.clone();
                    body.push(format!("let self.{} = {};", target_name, value_str));
                } else {
                    target_name = expr_to_raw(target);
                    body.push(format!("let {} = {};", target_name, value_str));
                };
            }
            Stmt::Assign(assign) => {
                let value: Expr = *assign.value.clone();
                let (value_str, additional_init_args) = extract_value(value, table);
                if let Some(arg) = additional_init_args {
                    init_args.push(arg);
                }

                if let Some(target_name) = self_attr_name(&assign.targets[0]) {
                    model_fields.push(target_name.clone());
                    body.push(format!("let {} = {};", target_name, value_str));
                } else {
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
                    body.push(format!("let {} = {};", target, value_str));
                }
            }
            _ => {
                body.push(format!("/* TODO: {} */", stmt_to_raw(stmt)));
            }
        }
    }
    if class_name == "Model" {
        build_main_model_config(class_name, init_args, model_fields, body)
    } else {
        build_sub_model_config(class_name, init_args, model_fields, body)
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

fn extract_forward(func: &StmtFunctionDef, table: &HashMap<&str, LayerInfo>) -> String {
    let mut body: Vec<String> = Vec::new();
    let mut args: Vec<String> = Vec::new();

    for arg in func.args.args.iter().skip(1) {
        args.push(arg.def.arg.to_string());
    }

    for stmt in &func.body {
        match stmt {
            Stmt::Return(ret) => {
                if let Some(val) = &ret.value {
                    body.push(stmt_expr_to_burn(val, table));
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
                let rhs = stmt_expr_to_burn(&assign.value, table);
                body.push(format!("let {} = {};", target, rhs));
            }
            Stmt::AugAssign(aug) => {
                let target = expr_to_raw(&aug.target);
                let op = binop_to_burn(&aug.op);
                let rhs = stmt_expr_to_burn(&aug.value, table);
                body.push(format!("let {} = {} {} {};", target, target, op, rhs));
            }
            Stmt::Expr(e) => {
                let s = stmt_expr_to_burn(&e.value, table);
                body.push(format!("{};", s));
            }
            _ => {
                body.push(format!("/* TODO: {} */", stmt_to_raw(stmt)));
            }
        }
    }
    build_fn_str(func.name.to_string(), args, body)
}

// ─────────────────────────────────────────────────────────────────────────────
// Expression conversion: Python → Burn Rust
// ─────────────────────────────────────────────────────────────────────────────

pub fn stmt_expr_to_burn(expr: &Expr, table: &HashMap<&str, LayerInfo>) -> String {
    let fn_map = fn_table();
    convert_expr(expr, &fn_map, table)
}

fn convert_expr(
    expr: &Expr,
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
            let owner = convert_expr(&attr.value, fn_map, table);
            let field = attr.attr.as_str();
            // self.xxx → keep as self.xxx (forward() in Burn calls methods directly)
            format!("{}.{}", owner, field)
        }

        // ── function call ────────────────────────────────────────────────────
        Expr::Call(call) => convert_call(call, fn_map, table),

        // ── binary operator ──────────────────────────────────────────────────
        Expr::BinOp(binop) => {
            let left = convert_expr(&binop.left, fn_map, table);
            let right = convert_expr(&binop.right, fn_map, table);
            match &binop.op {
                Operator::MatMult => format!("{}.matmul({})", left, right),
                Operator::Pow => format!("{}.powf({})", left, right),
                op => format!("{} {} {}", left, binop_to_burn(op), right),
            }
        }

        // ── unary operator ───────────────────────────────────────────────────
        Expr::UnaryOp(u) => {
            let operand = convert_expr(&u.operand, fn_map, table);
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
                .map(|e| convert_expr(e, fn_map, table))
                .collect();
            format!("({})", elts.join(", "))
        }
        Expr::List(l) => {
            let elts: Vec<_> = l
                .elts
                .iter()
                .map(|e| convert_expr(e, fn_map, table))
                .collect();
            format!("[{}]", elts.join(", "))
        }

        // ── subscript (indexing) ─────────────────────────────────────────────
        Expr::Subscript(s) => {
            let val = convert_expr(&s.value, fn_map, table);
            let idx = convert_expr(&s.slice, fn_map, table);
            format!("{}[{}]", val, idx)
        }

        // ── ternary (if-else) ────────────────────────────────────────────────
        Expr::IfExp(ife) => {
            let cond = convert_expr(&ife.test, fn_map, table);
            let body = convert_expr(&ife.body, fn_map, table);
            let orelse = convert_expr(&ife.orelse, fn_map, table);
            format!("if {} {{ {} }} else {{ {} }}", cond, body, orelse)
        }

        _ => format!("/* TODO: {} */", expr_to_raw(expr)),
    }
}

fn convert_call(
    call: &ExprCall,
    fn_map: &HashMap<&str, &str>,
    table: &HashMap<&str, LayerInfo>,
) -> String {
    let func_raw = expr_to_raw(&call.func);

    // ── torch.cat / torch.stack ───────────────────────────────────────────
    if matches!(func_raw.as_str(), "torch.cat" | "torch.stack") {
        return convert_cat_stack(call, &func_raw, fn_map, table);
    }

    // ── torch.zeros / ones / rand ─────────────────────────────────────────
    if let Some(burn_ctor) = tensor_ctor(&func_raw) {
        let args: Vec<_> = call
            .args
            .iter()
            .map(|a| convert_expr(a, fn_map, table))
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
            .map(|a| convert_expr(a, fn_map, table))
            .collect();
        let kw_dim = call.keywords.iter().find_map(|k| {
            if k.arg.as_deref() == Some("dim") {
                Some(convert_expr(&k.value, fn_map, table))
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
        let receiver = convert_expr(&attr.value, fn_map, table);
        let method = attr.attr.as_str();
        let args: Vec<_> = call
            .args
            .iter()
            .map(|a| convert_expr(a, fn_map, table))
            .collect();
        let kwargs: Vec<_> = call
            .keywords
            .iter()
            .filter_map(|k| {
                let name = k.arg.as_ref()?.as_str().to_string();
                Some((name, convert_expr(&k.value, fn_map, table)))
            })
            .collect();

        // self.layer(x)  → self.layer.forward(x)  (if layer is a module)
        if let Expr::Name(n) = attr.value.as_ref()
            && n.id.as_str() == "self"
        {
            let field_name = attr.attr.as_str();

            return format!("self.{}.forward({})", field_name, args.join(", "));
        }

        return convert_method_call(&receiver, method, &args, &kwargs);
    }

    // ── fallback ──────────────────────────────────────────────────────────
    let func_burn = convert_expr(&call.func, fn_map, table);
    let args: Vec<_> = call
        .args
        .iter()
        .map(|a| convert_expr(a, fn_map, table))
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
    fn_map: &HashMap<&str, &str>,
    table: &HashMap<&str, LayerInfo>,
) -> String {
    let tensors = if let Some(first) = call.args.first() {
        convert_expr(first, fn_map, table)
    } else {
        String::from("/* tensors */")
    };
    let dim = call
        .args
        .get(1)
        .map(|a| convert_expr(a, fn_map, table))
        .or_else(|| {
            call.keywords
                .iter()
                .find(|k| k.arg.as_deref() == Some("dim"))
                .map(|k| convert_expr(&k.value, fn_map, table))
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
            format!("let {} = {}", targets.join(", "), expr_to_raw(&a.value))
        }
        Stmt::AnnAssign(assign) => {
            let target = expr_to_raw(&assign.target);
            let value = assign
                .value
                .as_ref()
                .map(|v| expr_to_raw(v))
                .unwrap_or_default();
            format!("let {} = {}", target, value)
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
