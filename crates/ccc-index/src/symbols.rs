//! Finds definitions in source files.

use tree_sitter::{Language, Node, Parser};

/// A named definition in a source file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Symbol {
    /// The name as written.
    pub name: String,
    /// The grammar's name for what it is, such as `function_item`.
    pub kind: String,
    /// The line it starts on, counted from one.
    pub line: usize,
}

/// Node kinds that define something, by language. Each has a `name` field.
const DEFINITIONS: &[&str] = &[
    // Rust
    "function_item",
    "struct_item",
    "enum_item",
    "trait_item",
    "type_item",
    "const_item",
    "static_item",
    "mod_item",
    "macro_definition",
    "function_signature_item",
    // Python
    "function_definition",
    "class_definition",
    // JavaScript and TypeScript
    "function_declaration",
    "class_declaration",
    "method_definition",
    "interface_declaration",
    "type_alias_declaration",
    "enum_declaration",
    // Go
    "method_declaration",
    "type_spec",
];

fn language_for(extension: &str) -> Option<Language> {
    Some(match extension {
        "rs" => tree_sitter_rust::LANGUAGE.into(),
        "py" => tree_sitter_python::LANGUAGE.into(),
        "js" | "jsx" | "mjs" | "cjs" => tree_sitter_javascript::LANGUAGE.into(),
        "ts" | "mts" | "cts" => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        "tsx" => tree_sitter_typescript::LANGUAGE_TSX.into(),
        "go" => tree_sitter_go::LANGUAGE.into(),
        _ => return None,
    })
}

/// Lists the definitions in `source`, judged by the file extension. Returns nothing for
/// languages without a grammar here.
#[must_use]
pub fn extract_symbols(extension: &str, source: &str) -> Vec<Symbol> {
    let Some(language) = language_for(extension) else {
        return Vec::new();
    };
    let mut parser = Parser::new();
    if parser.set_language(&language).is_err() {
        return Vec::new();
    }
    let Some(tree) = parser.parse(source, None) else {
        return Vec::new();
    };
    let mut found = Vec::new();
    let mut pending = vec![tree.root_node()];
    while let Some(node) = pending.pop() {
        record_definition(node, source, &mut found);
        let mut cursor = node.walk();
        pending.extend(node.children(&mut cursor));
    }
    found.sort_by_key(|symbol: &Symbol| symbol.line);
    found
}

fn record_definition(node: Node<'_>, source: &str, found: &mut Vec<Symbol>) {
    if !DEFINITIONS.contains(&node.kind()) {
        return;
    }
    let Some(name) = node.child_by_field_name("name") else {
        return;
    };
    if let Ok(text) = name.utf8_text(source.as_bytes()) {
        found.push(Symbol {
            name: text.to_owned(),
            kind: node.kind().to_owned(),
            line: node.start_position().row + 1,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(extension: &str, source: &str) -> Vec<String> {
        extract_symbols(extension, source)
            .into_iter()
            .map(|symbol| symbol.name)
            .collect()
    }

    #[test]
    fn finds_rust_definitions_including_nested_ones() {
        let source = "pub struct Config;\nimpl Config {\n    fn load() {}\n}\nenum Mode { A }\ntrait Run {}\n";
        assert_eq!(names("rs", source), ["Config", "load", "Mode", "Run"]);
    }

    #[test]
    fn finds_python_definitions() {
        let source = "class Parser:\n    def parse(self):\n        pass\n\ndef main():\n    pass\n";
        assert_eq!(names("py", source), ["Parser", "parse", "main"]);
    }

    #[test]
    fn finds_javascript_and_typescript_definitions() {
        let js = "function run() {}\nclass Job { start() {} }\n";
        assert_eq!(names("js", js), ["run", "Job", "start"]);
        let ts = "interface Opts { a: string }\ntype Id = string;\nfunction go(o: Opts) {}\n";
        assert_eq!(names("ts", ts), ["Opts", "Id", "go"]);
        let tsx = "function App() { return <div />; }\n";
        assert_eq!(names("tsx", tsx), ["App"]);
    }

    #[test]
    fn finds_go_definitions() {
        let source =
            "package main\ntype Server struct{}\nfunc (s Server) Run() {}\nfunc main() {}\n";
        assert_eq!(names("go", source), ["Server", "Run", "main"]);
    }

    #[test]
    fn records_the_line_and_survives_broken_or_unknown_input() {
        let symbols = extract_symbols("rs", "\n\nfn third() {}\n");
        assert_eq!(symbols[0].line, 3);
        assert_eq!(symbols[0].kind, "function_item");
        assert!(extract_symbols("txt", "fn nope() {}").is_empty());
        let _ = extract_symbols("rs", "fn broken( {{{ !!!");
    }
}
