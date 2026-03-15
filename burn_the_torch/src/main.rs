mod ast_visitor;
mod build_forward_str;
mod build_main_model_config;
mod build_sub_model_config;
mod layer_map;
use anyhow::{Context, Result};
use clap::Parser;
use std::{
    fs,
    io::{self, Write},
    path::PathBuf,
};

/// Convert PyTorch Python code to Rust Burn code.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Input Python file (omit to read from stdin)
    #[arg(short, long, value_name = "FILE")]
    input: PathBuf,

    /// Output Rust file (omit to write to stdout)
    #[arg(short, long, value_name = "FILE")]
    output: Option<PathBuf>,

    /// Print extracted model info (JSON-like) instead of generating Rust code
    #[arg(long)]
    dump: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // ── 入力読み込み ──────────────────────────────────────────────────────
    let source = fs::read_to_string(&cli.input)
        .with_context(|| format!("cannot read {:?}", cli.input))
        .unwrap();

    // ── AST 解析 ─────────────────────────────────────────────────────────
    let models = ast_visitor::extract_models(&source).context("failed to parse Python source")?;

    if models.is_empty() {
        eprintln!(
            "Warning: no nn.Module subclass found in the input. \
             Make sure the class inherits from `nn.Module` or `torch.nn.Module`."
        );
    }

    // ── 出力 ─────────────────────────────────────────────────────────────
    let output_text = models;

    match &cli.output {
        Some(path) => {
            fs::write(path, &output_text).with_context(|| format!("cannot write {:?}", path))?;
            eprintln!("Written to {:?}", path);
        }
        None => {
            io::stdout().write_all(output_text.as_bytes())?;
        }
    }

    Ok(())
}
