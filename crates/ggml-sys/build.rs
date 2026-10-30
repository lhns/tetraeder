//! Builds vendor/llama.cpp (ggml + libllama, static) with cmake and generates bindings with bindgen.
//!
//! Env: `TETRAEDER_CUDA_ARCHS` (default "86;89;90;120"), `TETRAEDER_GGML_NATIVE` (ON/OFF, ggml
//! default when unset), `LIBCLANG_PATH` for bindgen.
use std::path::{Path, PathBuf};

fn main() {
    let manifest = PathBuf::from(env("CARGO_MANIFEST_DIR"));
    // No canonicalize: its \?\ prefix on Windows breaks cmake/MSBuild paths.
    let src = manifest
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .join("vendor/llama.cpp");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed=../../vendor/PIN");
    println!("cargo:rerun-if-changed=../../patches");
    println!("cargo:rerun-if-env-changed=TETRAEDER_CUDA_ARCHS");
    println!("cargo:rerun-if-env-changed=TETRAEDER_GGML_NATIVE");
    println!("cargo:rerun-if-env-changed=LIBCLANG_PATH");

    if !src.join("CMakeLists.txt").is_file() {
        panic!(
            "vendor/llama.cpp is empty. Run `git submodule update --init --depth 1 vendor/llama.cpp` \
             (then `patches/apply.sh` or `patches/apply.ps1` to apply the patch queue)."
        );
    }

    let cuda = cfg!(feature = "cuda");
    let vulkan = cfg!(feature = "vulkan");

    let mut cfg = cmake::Config::new(&src);
    cfg.profile("Release")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("GGML_OPENMP", "OFF")
        .define("LLAMA_BUILD_COMMON", "OFF")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_TOOLS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_SERVER", "OFF")
        .define("LLAMA_BUILD_APP", "OFF")
        .define("LLAMA_BUILD_MTMD", "OFF")
        .define("LLAMA_CURL", "OFF")
        .define("LLAMA_OPENSSL", "OFF")
        .define("GGML_CUDA", onoff(cuda))
        // Single-GPU until Phase 5; NCCL found on the box image would otherwise add an unlinked dependency.
        .define("GGML_CUDA_NCCL", "OFF")
        .define("GGML_VULKAN", onoff(vulkan))
        .build_target("install");
    // ggml default: native ISA. Set TETRAEDER_GGML_NATIVE=OFF for portable binaries (AVX2 etc.
    // then come from ggml's own defaults).
    if let Ok(v) = std::env::var("TETRAEDER_GGML_NATIVE") {
        cfg.define("GGML_NATIVE", v);
    }
    if cuda {
        let archs = std::env::var("TETRAEDER_CUDA_ARCHS").unwrap_or_else(|_| "86;89;90;120".into());
        cfg.define("CMAKE_CUDA_ARCHITECTURES", archs);
    }
    // cmake-rs replaces CMAKE_<LANG>_FLAGS_RELEASE on MSVC with its own flags, which drop /O2 and
    // /DNDEBUG: ggml would build unoptimized with asserts on. Restore CMake's MSVC Release defaults.
    // The CRT flag must match Rust's (/MD unless crt-static).
    if env("CARGO_CFG_TARGET_ENV") == "msvc" {
        let static_crt = std::env::var("CARGO_CFG_TARGET_FEATURE")
            .is_ok_and(|f| f.split(',').any(|x| x == "crt-static"));
        let crt = if static_crt { "/MT" } else { "/MD" };
        for lang in ["C", "CXX"] {
            cfg.define(
                format!("CMAKE_{lang}_FLAGS_RELEASE"),
                format!("-nologo {crt} -Brepro /O2 /Ob2 /DNDEBUG"),
            );
        }
    }
    let out = cfg.build();

    for dir in ["lib", "lib64"] {
        let p = out.join(dir);
        if p.is_dir() {
            println!("cargo:rustc-link-search=native={}", p.display());
        }
    }
    // Order matters for GNU ld: dependents before dependencies.
    println!("cargo:rustc-link-lib=static=llama");
    println!("cargo:rustc-link-lib=static=ggml");
    if cuda {
        println!("cargo:rustc-link-lib=static=ggml-cuda");
    }
    if vulkan {
        println!("cargo:rustc-link-lib=static=ggml-vulkan");
    }
    println!("cargo:rustc-link-lib=static=ggml-cpu");
    println!("cargo:rustc-link-lib=static=ggml-base");

    let target_os = env("CARGO_CFG_TARGET_OS");
    match target_os.as_str() {
        "linux" => {
            println!("cargo:rustc-link-lib=stdc++");
            println!("cargo:rustc-link-lib=m");
            println!("cargo:rustc-link-lib=pthread");
            println!("cargo:rustc-link-lib=dl");
        }
        "macos" => println!("cargo:rustc-link-lib=c++"),
        "windows" => println!("cargo:rustc-link-lib=advapi32"),
        _ => {}
    }
    if cuda {
        link_cuda(&target_os);
    }
    if vulkan {
        link_vulkan(&target_os);
    }

    generate_bindings(&src);
}

fn env(k: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| panic!("{k} not set"))
}

fn onoff(b: bool) -> &'static str {
    if b { "ON" } else { "OFF" }
}

fn link_cuda(target_os: &str) {
    let root = std::env::var("CUDA_PATH")
        .or_else(|_| std::env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".into());
    let root = Path::new(&root);
    let dirs: &[&str] = if target_os == "windows" {
        &["lib/x64"]
    } else {
        &["lib64", "lib64/stubs", "lib"]
    };
    for d in dirs {
        let p = root.join(d);
        if p.is_dir() {
            println!("cargo:rustc-link-search=native={}", p.display());
        }
    }
    for l in ["cudart", "cublas", "cublasLt", "cuda"] {
        println!("cargo:rustc-link-lib={l}");
    }
}

fn link_vulkan(target_os: &str) {
    if target_os == "windows" {
        if let Ok(sdk) = std::env::var("VULKAN_SDK") {
            println!("cargo:rustc-link-search=native={sdk}/Lib");
        }
        println!("cargo:rustc-link-lib=vulkan-1");
    } else {
        println!("cargo:rustc-link-lib=vulkan");
    }
}

fn generate_bindings(src: &Path) {
    let manifest = PathBuf::from(env("CARGO_MANIFEST_DIR"));
    let builder = bindgen::Builder::default()
        .header(manifest.join("wrapper.h").to_string_lossy())
        .clang_arg(format!("-I{}", src.join("include").display()))
        .clang_arg(format!("-I{}", src.join("ggml/include").display()))
        .allowlist_function("(llama|ggml|gguf)_.*")
        .allowlist_type("(llama|ggml|gguf)_.*")
        .allowlist_var("(LLAMA|GGML|GGUF|llama|ggml|gguf)_.*")
        .default_enum_style(bindgen::EnumVariation::Consts)
        .prepend_enum_name(false)
        .derive_default(true)
        .layout_tests(false)
        .generate_comments(false)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()));
    let bindings = builder.generate().unwrap_or_else(|e| {
        panic!(
            "bindgen failed ({e}). It needs libclang: install LLVM (Windows: `winget install LLVM.LLVM`,
             Debian/Ubuntu: `apt install libclang-dev`) and set LIBCLANG_PATH to the directory
             holding libclang.dll / libclang.so."
        )
    });
    let dest = PathBuf::from(env("OUT_DIR")).join("bindings.rs");
    bindings.write_to_file(dest).expect("write bindings");
}
