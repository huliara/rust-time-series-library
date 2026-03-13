pub fn build_sub_model_config(
    class_name: &String,
    init_args: Vec<String>,
    model_fields: Vec<String>,
    init_body: Vec<String>,
) -> String {
    let mut out = String::new();

    out.push_str("#[derive(Config, Debug)]\n");
    out.push_str(format!("pub struct {}Config {{\n", class_name).as_str());
    for field in &init_args {
        out.push_str(&format!("    pub {}:usize ,\n", field));
    }
    out.push_str(    "#[config(\n default = \"Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}\")]\n");
    out.push_str("    pub initializer: Initializer,\n");
    out.push_str("}\n\n");
    // ── impl ブロック ─────────────────────────────────────────────────────
    out.push_str(format!("impl {}Config {{\n", class_name).as_str());
    out.push_str("    pub fn init<B: Backend>(\n");
    out.push_str("        &self,\n");
    out.push_str("        device: &B::Device,\n");
    out.push_str(format!("    ) -> {}<B> {{\n", class_name).as_str());
    for body in &init_body {
        out.push_str(&format!("        {}\n", body));
    }
    out.push_str(format!("{} {{\n", class_name).as_str());
    for field in &model_fields {
        out.push_str(&format!("            {},\n", field));
    }
    out.push_str("    }\n");
    out.push_str("}\n");
    out.push_str("}\n");
    out.push_str(format!("struct {}<B: Backend> {{\n", class_name).as_str());
    for field in &model_fields {
        out.push_str(&format!("    {}:,\n", field));
    }
    out.push_str("}\n");
    out
}
