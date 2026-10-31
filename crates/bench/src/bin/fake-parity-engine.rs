//! Test double for both sides of the parity check. Like `llama-simple` it prints the prompt (the last argument)
//! and a continuation; like `tetraeder run` (first argument `run`) it prints the continuation only.
//! `FAKE_PARITY_DIFF=1` changes the continuation of the `run` mode, so the outputs differ.

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().is_some_and(|a| a == "run") {
        let diff = std::env::var_os("FAKE_PARITY_DIFF").is_some();
        println!("{}", if diff { " Lyon." } else { " Paris." });
    } else {
        let prompt = args.last().cloned().unwrap_or_default();
        println!("{prompt} Paris.");
    }
}
