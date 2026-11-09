//! Builds vendor/llama.cpp (ggml + libllama) with cmake and generates bindings with bindgen.
//!
//! Two modes (ADR 0052):
//! - default (DL): shared libllama, ggml and ggml-base; every backend is a loadable module
//!   (`ggml-cpu-<tag>` per x86 level, `ggml-cuda`, `ggml-vulkan`) that `llama::init()` loads at
//!   run time, so one build runs at the best ISA of each host. Sets `cfg(ggml_dl)`.
//! - `TETRAEDER_GGML_NATIVE=ON`: static build with the native ISA of the build machine.
//!
//! Env: `TETRAEDER_GGML_NATIVE` (ON: static native; OFF or unset: DL), `TETRAEDER_CUDA_ARCHS`
//! (default "86;89;90;120"), `TETRAEDER_GGML_BUILD_DIR` (see below), `LIBCLANG_PATH` for bindgen. Run time (DL): `TETRAEDER_GGML_BACKEND_DIR`,
//! `TETRAEDER_GGML_CPU_VARIANT`.
//!
//! Cargo runs this script once per build flavor (clippy, build, test: separate `OUT_DIR`s), and each run
//! compiled all 14 CPU variants again. `TETRAEDER_GGML_BUILD_DIR` points every run at one cmake tree
//! (one subdirectory per configuration), so the later runs only relink. CI sets it; unset, the build
//! stays in `OUT_DIR`.
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
    println!("cargo:rerun-if-env-changed=TETRAEDER_GGML_BUILD_DIR");
    println!("cargo:rerun-if-env-changed=LIBCLANG_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo::rustc-check-cfg=cfg(ggml_dl)");

    if !src.join("CMakeLists.txt").is_file() {
        panic!(
            "vendor/llama.cpp is empty. Run `git submodule update --init --depth 1 vendor/llama.cpp` \
             (then `patches/apply.sh` or `patches/apply.ps1` to apply the patch queue)."
        );
    }

    check_patches(&src);

    let native = std::env::var("TETRAEDER_GGML_NATIVE").is_ok_and(|v| v == "ON");
    let dl = !native;
    let cuda = cfg!(feature = "cuda");
    let vulkan = cfg!(feature = "vulkan");

    let mut cfg = cmake::Config::new(&src);
    if let Ok(base) = std::env::var("TETRAEDER_GGML_BUILD_DIR") {
        // Everything that changes the cmake configuration is in the name, so trees never mix.
        let archs = std::env::var("TETRAEDER_CUDA_ARCHS").unwrap_or_default();
        let name = format!(
            "{}-{}-cuda{}-vk{}-{}-{}",
            env("CARGO_CFG_TARGET_OS"),
            if native { "native" } else { "dl" },
            u8::from(cuda),
            u8::from(vulkan),
            archs.replace(';', "_"),
            env("PROFILE"),
        );
        cfg.out_dir(Path::new(&base).join(name));
    }
    cfg.profile("Release")
        .define("BUILD_SHARED_LIBS", onoff(dl))
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
    if dl {
        // GGML_CPU_ALL_VARIANTS builds one ggml-cpu module per x86 level; ggml scores them at load.
        cfg.define("GGML_BACKEND_DL", "ON")
            .define("GGML_CPU_ALL_VARIANTS", "ON")
            .define("GGML_NATIVE", "OFF");
    } else {
        cfg.define("GGML_NATIVE", "ON");
    }
    if cuda {
        let archs = std::env::var("TETRAEDER_CUDA_ARCHS").unwrap_or_else(|_| "86;89;90;120".into());
        cfg.define("CMAKE_CUDA_ARCHITECTURES", archs);
        // ggml's default list plus q8_0-q4_0 (K Q8_0, V Q4_0 KV cache, ADR 0050).
        cfg.define(
            "GGML_CUDA_FA_QUANTS",
            "q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16;q8_0-q4_0",
        );
    }
    // cmake-rs replaces CMAKE_<LANG>_FLAGS_RELEASE on MSVC with its own flags, which drop /O2 and
    // /DNDEBUG: ggml would build unoptimized with asserts on. Restore CMake's MSVC Release defaults.
    // The CRT flag must match Rust's (/MD unless crt-static).
    if env("CARGO_CFG_TARGET_ENV") == "msvc" {
        let static_crt = std::env::var("CARGO_CFG_TARGET_FEATURE")
            .is_ok_and(|f| f.split(',').any(|x| x == "crt-static"));
        let crt = if static_crt { "/MT" } else { "/MD" };
        // Debug builds (and CMake's compiler checks) default to /Zi with a shared PDB, which sccache cannot
        // cache; embed the debug info (/Z7) instead. Needs policy CMP0141 (CMake 3.25).
        cfg.define("CMAKE_POLICY_DEFAULT_CMP0141", "NEW")
            .define("CMAKE_MSVC_DEBUG_INFORMATION_FORMAT", "Embedded");
        for lang in ["C", "CXX"] {
            cfg.define(
                format!("CMAKE_{lang}_FLAGS_RELEASE"),
                format!("-nologo {crt} -Brepro /O2 /Ob2 /DNDEBUG"),
            );
        }
    }
    let out = cfg.build();

    // DL: Windows DLLs and all modules install to bin. The link-search dirs inside target/ also
    // end up on PATH / LD_LIBRARY_PATH for `cargo test` and `cargo run`.
    for dir in ["lib", "lib64", "bin"] {
        let p = out.join(dir);
        if p.is_dir() {
            println!("cargo:rustc-link-search=native={}", p.display());
        }
    }
    if dl {
        println!("cargo::rustc-cfg=ggml_dl");
        println!(
            "cargo:rustc-env=TETRAEDER_GGML_MODULE_DIR={}",
            out.join("bin").display()
        );
        for l in ["llama", "ggml", "ggml-base"] {
            println!("cargo:rustc-link-lib=dylib={l}");
        }
        copy_runtime_files(&out, &env("CARGO_CFG_TARGET_OS"));
    } else {
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
    }

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
    // In DL mode the GPU modules link their own libraries.
    if cuda && !dl {
        link_cuda(&target_os);
    }
    if vulkan && !dl {
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

/// Fails the build when a patch of `patches/series` is not applied to `vendor/llama.cpp`
/// (an unpatched tree would build without the fix and fail later in odd ways). Silent without git.
fn check_patches(src: &Path) {
    let root = src.parent().and_then(Path::parent).expect("workspace root");
    let Ok(series) = std::fs::read_to_string(root.join("patches/series")) else {
        return;
    };
    for line in series.lines() {
        let name = line.split('#').next().unwrap_or("").trim();
        if name.is_empty() {
            continue;
        }
        let patch = root.join("patches").join(name);
        let check = std::process::Command::new("git")
            .arg("-C")
            .arg(src)
            .args(["apply", "--reverse", "--check"])
            .arg(&patch)
            .output();
        match check {
            Err(_) => return, // no git
            Ok(o) if !o.status.success() => panic!(
                "patch {name} is not applied to vendor/llama.cpp: run patches/apply.sh (or \
                 patches/apply.ps1)."
            ),
            Ok(_) => {}
        }
    }
}

/// Copies the shared libraries and modules next to the binaries so they run without Cargo's
/// library path (pytest and the box scripts call `target/<profile>/tetraeder`). `OUT_DIR` is
/// `target[/<triple>]/<profile>/build/<pkg>/out`, so the profile dir is its fourth ancestor.
fn copy_runtime_files(out: &Path, target_os: &str) {
    let dest = PathBuf::from(env("OUT_DIR"))
        .ancestors()
        .nth(3)
        .expect("profile dir")
        .to_path_buf();
    let mut files = Vec::new();
    let list = |dir: PathBuf, keep: &dyn Fn(&str) -> bool, files: &mut Vec<PathBuf>| {
        for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            if keep(&name) {
                files.push(e.path());
            }
        }
    };
    if target_os == "windows" {
        list(out.join("bin"), &|n| n.ends_with(".dll"), &mut files);
    } else {
        // Real files under every name, SONAME names included: symlinks do not survive all copies.
        let core = |n: &str| {
            ["libllama.so", "libggml.so", "libggml-base.so"]
                .iter()
                .any(|p| {
                    n.strip_prefix(p)
                        .is_some_and(|r| r.is_empty() || r.starts_with('.'))
                })
        };
        for d in ["lib", "lib64"] {
            list(out.join(d), &core, &mut files);
        }
        list(
            out.join("bin"),
            &|n| n.starts_with("libggml-") && n.ends_with(".so"),
            &mut files,
        );
    }
    for f in files {
        let to = dest.join(f.file_name().expect("file name"));
        let same = match (std::fs::read(&f), std::fs::read(&to)) {
            (Ok(a), Ok(b)) => a == b,
            _ => false,
        };
        if !same {
            std::fs::copy(&f, &to)
                .unwrap_or_else(|e| panic!("copy {} to {}: {e}", f.display(), to.display()));
        }
    }
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
