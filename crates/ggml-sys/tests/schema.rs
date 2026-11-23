//! JSON schema to GBNF through llama.cpp's converter (shim/schema.cpp, ADR 0083 group B). No model.

use tetraeder_ggml_sys::llama::json_schema_to_grammar;

#[test]
fn object_schema_gives_a_root_rule() {
    let g = json_schema_to_grammar(r#"{"type": "object"}"#).unwrap();
    assert!(g.contains("root ::="), "{g}");
}

#[test]
fn properties_become_rules() {
    let g = json_schema_to_grammar(
        r#"{"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}"#,
    )
    .unwrap();
    assert!(g.contains("name"), "{g}");
    assert!(g.contains("string"), "{g}");
}

#[test]
fn bad_input_is_an_error_not_a_crash() {
    assert!(json_schema_to_grammar("{not json").is_err());
    assert!(json_schema_to_grammar(r##"{"$ref": "#/nope"}"##).is_err());
}
