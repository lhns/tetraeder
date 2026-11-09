//! The loaded CPU backend: which ISA level ggml picked, and the CPU-only entry points (ADR 0052).
//!
//! ggml-cpu is a loadable module in the default build, so nothing here links against it. Entry
//! points that only the CPU backend has (threadpools, thread count) are resolved through the CPU
//! registry's `get_proc_address`, in both build modes.

use super::{GgmlError, Result, cstr};
use crate::raw as sys;
use std::ffi::{CStr, c_int, c_void};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// How the CPU backend got into the process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Dispatch {
    /// Loaded as a module at run time (default build).
    Dl,
    /// Linked statically (`TETRAEDER_GGML_NATIVE=ON`).
    Static,
}

impl Dispatch {
    pub fn as_str(self) -> &'static str {
        match self {
            Dispatch::Dl => "dl",
            Dispatch::Static => "static",
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct CpuBackendInfo {
    pub dispatch: Dispatch,
    /// DL only: the directory the modules came from.
    pub module_dir: Option<PathBuf>,
    /// `TETRAEDER_GGML_CPU_VARIANT`, when set.
    pub forced: Option<String>,
    /// Features the loaded CPU backend was compiled with (ggml's names, value "1" only).
    pub isa: Vec<String>,
    pub tier: &'static str,
    /// Tracked features the host has.
    pub host_isa: Vec<String>,
    /// Host features the build can provide and the loaded backend lacks.
    pub missing: Vec<String>,
    /// Tracked features the loaded backend has and the host lacks: a wrong variant was loaded.
    pub extra: Vec<String>,
}

/// The features compared between host and backend, in ggml's names.
pub const TRACKED: [&str; 11] = [
    "AVX",
    "AVX2",
    "FMA",
    "F16C",
    "BMI2",
    "AVX_VNNI",
    "AVX512",
    "AVX512_VBMI",
    "AVX512_VNNI",
    "AVX512_BF16",
    "AMX_INT8",
];

/// Tracked features this build has a module for. MSVC builds have no BF16 and no AMX variants.
pub fn provided() -> Vec<&'static str> {
    TRACKED
        .iter()
        .copied()
        .filter(|f| !(cfg!(target_env = "msvc") && matches!(*f, "AVX512_BF16" | "AMX_INT8")))
        .collect()
}

/// The highest ISA class of a feature list.
pub fn tier(isa: &[String]) -> &'static str {
    let has = |f: &str| isa.iter().any(|x| x == f);
    if has("AMX_INT8") {
        "amx_int8"
    } else if has("AVX512_BF16") {
        "avx512_bf16"
    } else if has("AVX512_VNNI") {
        "avx512_vnni"
    } else if has("AVX512") {
        "avx512"
    } else if has("AVX_VNNI") {
        "avx_vnni"
    } else if has("AVX2") {
        "avx2"
    } else if has("AVX") {
        "avx"
    } else {
        "sse"
    }
}

/// Features in `host` that are in `provided` and not in `isa`. ggml has no variant with both
/// AVX512 and AVX_VNNI (Sapphire Rapids): AVX512_VNNI covers it.
pub fn missing(host: &[String], isa: &[String], provided: &[&str]) -> Vec<String> {
    let has = |f: &str| isa.iter().any(|x| x == f);
    host.iter()
        .filter(|f| provided.contains(&f.as_str()) && !isa.contains(f))
        .filter(|f| !(f.as_str() == "AVX_VNNI" && has("AVX512_VNNI")))
        .cloned()
        .collect()
}

/// Tracked features in `isa` that `host` lacks.
pub fn extra(host: &[String], isa: &[String]) -> Vec<String> {
    isa.iter()
        .filter(|f| TRACKED.contains(&f.as_str()) && !host.contains(f))
        .cloned()
        .collect()
}

/// Tracked features of this host (empty off x86-64).
pub fn host_features() -> Vec<String> {
    #[allow(unused_mut)]
    let mut v: Vec<&str> = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::is_x86_feature_detected as d;
        let mut add = |on: bool, name| {
            if on {
                v.push(name)
            }
        };
        add(d!("avx"), "AVX");
        add(d!("avx2"), "AVX2");
        add(d!("fma"), "FMA");
        add(d!("f16c"), "F16C");
        add(d!("bmi2"), "BMI2");
        add(d!("avxvnni"), "AVX_VNNI");
        add(
            d!("avx512f") && d!("avx512cd") && d!("avx512vl") && d!("avx512dq") && d!("avx512bw"),
            "AVX512",
        );
        add(d!("avx512vbmi"), "AVX512_VBMI");
        add(d!("avx512vnni"), "AVX512_VNNI");
        add(d!("avx512bf16"), "AVX512_BF16");
        // `amx-int8` is not stable in is_x86_feature_detected! (Rust 1.95).
        add(cpuinfo_has_flag("amx_int8"), "AMX_INT8");
    }
    v.into_iter().map(String::from).collect()
}

#[cfg(target_arch = "x86_64")]
fn cpuinfo_has_flag(flag: &str) -> bool {
    std::fs::read_to_string("/proc/cpuinfo")
        .map(|s| {
            s.lines()
                .filter(|l| l.starts_with("flags"))
                .any(|l| l.split_whitespace().any(|w| w == flag))
        })
        .unwrap_or(false)
}

/// File name of a ggml module on this OS: `ggml-cpu-haswell` is `libggml-cpu-haswell.so` or
/// `ggml-cpu-haswell.dll`.
pub fn module_file_name(module: &str) -> String {
    if cfg!(windows) {
        format!("{module}.dll")
    } else {
        format!("lib{module}.so")
    }
}

/// The file name of the CPU module of variant `tag`.
pub fn cpu_module_file_name(tag: &str) -> String {
    module_file_name(&format!("ggml-cpu-{tag}"))
}

/// True when `dir` holds at least one ggml-cpu module file.
pub fn dir_has_cpu_module(dir: &Path) -> bool {
    let (head, tail) = if cfg!(windows) {
        ("ggml-cpu", ".dll")
    } else {
        ("libggml-cpu", ".so")
    };
    std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .any(|e| {
            let n = e.file_name().to_string_lossy().into_owned();
            n.starts_with(head) && n.ends_with(tail)
        })
}

/// Where ggml modules are loaded from, first match wins: the env override, the executable's
/// directory (when it holds a CPU module), the directory baked in at build time.
pub fn resolve_module_dir(
    env: Option<&Path>,
    exe_dir: Option<&Path>,
    built: &Path,
    has_cpu_module: impl Fn(&Path) -> bool,
) -> PathBuf {
    if let Some(d) = env {
        return d.to_path_buf();
    }
    if let Some(d) = exe_dir.filter(|d| has_cpu_module(d)) {
        return d.to_path_buf();
    }
    built.to_path_buf()
}

/// Resolves a CPU-registry entry point.
fn cpu_proc(name: &CStr) -> Option<*mut c_void> {
    // SAFETY: registry lookups only; the CPU device exists once `llama::init()` succeeded.
    let p = unsafe {
        let dev = sys::ggml_backend_dev_by_type(sys::GGML_BACKEND_DEVICE_TYPE_CPU);
        if dev.is_null() {
            return None;
        }
        sys::ggml_backend_reg_get_proc_address(sys::ggml_backend_dev_backend_reg(dev), name.as_ptr())
    };
    (!p.is_null()).then_some(p)
}

type NewPool = unsafe extern "C" fn(*mut sys::ggml_threadpool_params) -> sys::ggml_threadpool_t;
type FreePool = unsafe extern "C" fn(sys::ggml_threadpool_t);
type SetPool = unsafe extern "C" fn(sys::ggml_backend_t, sys::ggml_threadpool_t);
type SetThreads = unsafe extern "C" fn(sys::ggml_backend_t, c_int);

/// CPU-only entry points, resolved once through the CPU registry.
pub(crate) struct CpuProcs {
    pub set_n_threads: SetThreads,
    pub set_threadpool: SetPool,
    pub threadpool_new: NewPool,
    pub threadpool_free: FreePool,
}

pub(crate) fn procs() -> Result<&'static CpuProcs> {
    static PROCS: OnceLock<std::result::Result<CpuProcs, String>> = OnceLock::new();
    crate::llama::init_checked().map_err(GgmlError::NoBackend)?;
    PROCS
        .get_or_init(|| {
            let get = |name: &CStr| {
                cpu_proc(name).ok_or_else(|| {
                    format!("CPU backend entry point {}", name.to_string_lossy())
                })
            };
            // SAFETY: the CPU registry returns these functions with exactly these signatures
            // (ggml-cpu.cpp, ggml_backend_cpu_get_proc_address).
            unsafe {
                Ok(CpuProcs {
                    set_n_threads: std::mem::transmute::<*mut c_void, SetThreads>(get(
                        c"ggml_backend_set_n_threads",
                    )?),
                    set_threadpool: std::mem::transmute::<*mut c_void, SetPool>(get(
                        c"ggml_backend_cpu_set_threadpool",
                    )?),
                    threadpool_new: std::mem::transmute::<*mut c_void, NewPool>(get(
                        c"ggml_threadpool_new",
                    )?),
                    threadpool_free: std::mem::transmute::<*mut c_void, FreePool>(get(
                        c"ggml_threadpool_free",
                    )?),
                })
            }
        })
        .as_ref()
        .map_err(|e| GgmlError::NoBackend(e.clone()))
}

/// Features of the loaded CPU backend (ggml's names, value "1" only).
fn backend_features() -> std::result::Result<Vec<String>, String> {
    let p = cpu_proc(c"ggml_backend_get_features").ok_or("CPU backend has no feature list")?;
    // SAFETY: the registry returns `ggml_backend_cpu_get_features`, which returns a static array
    // of name/value pairs terminated by a null name.
    unsafe {
        let f = std::mem::transmute::<*mut c_void, sys::ggml_backend_get_features_t>(p)
            .ok_or("CPU backend has no feature list")?;
        let dev = sys::ggml_backend_dev_by_type(sys::GGML_BACKEND_DEVICE_TYPE_CPU);
        let mut it = f(sys::ggml_backend_dev_backend_reg(dev));
        let mut out = Vec::new();
        while !it.is_null() && !(*it).name.is_null() {
            if cstr((*it).value) == "1" {
                out.push(cstr((*it).name));
            }
            it = it.add(1);
        }
        Ok(out)
    }
}

/// The CPU backend that is loaded in this process. Runs `llama::init()`; an error is the module
/// load error.
pub fn backend_info() -> Result<&'static CpuBackendInfo> {
    static INFO: OnceLock<std::result::Result<CpuBackendInfo, String>> = OnceLock::new();
    INFO.get_or_init(|| {
        let loaded = crate::llama::init_checked()?;
        let isa = backend_features()?;
        let host_isa = host_features();
        // Off x86-64 there is no host list to compare with.
        let comparable = cfg!(target_arch = "x86_64");
        Ok(CpuBackendInfo {
            dispatch: if cfg!(ggml_dl) {
                Dispatch::Dl
            } else {
                Dispatch::Static
            },
            module_dir: loaded.module_dir.clone(),
            forced: loaded.forced.clone(),
            tier: tier(&isa),
            missing: if comparable {
                missing(&host_isa, &isa, &provided())
            } else {
                Vec::new()
            },
            extra: if comparable {
                extra(&host_isa, &isa)
            } else {
                Vec::new()
            },
            host_isa,
            isa,
        })
    })
    .as_ref()
    .map_err(|e| GgmlError::NoBackend(e.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(xs: &[&str]) -> Vec<String> {
        xs.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn tier_from_features() {
        assert_eq!(tier(&v(&["SSE3"])), "sse");
        assert_eq!(tier(&v(&["AVX"])), "avx");
        assert_eq!(tier(&v(&["AVX", "AVX2", "FMA"])), "avx2");
        assert_eq!(tier(&v(&["AVX2", "AVX_VNNI"])), "avx_vnni");
        assert_eq!(tier(&v(&["AVX2", "AVX512"])), "avx512");
        assert_eq!(tier(&v(&["AVX512", "AVX512_VNNI"])), "avx512_vnni");
        assert_eq!(tier(&v(&["AVX512_VNNI", "AVX512_BF16"])), "avx512_bf16");
        assert_eq!(tier(&v(&["AVX512_BF16", "AMX_INT8"])), "amx_int8");
    }

    #[test]
    fn missing_and_extra_sets() {
        let host = v(&["AVX", "AVX2", "AVX512", "AVX512_BF16"]);
        let isa = v(&["AVX", "AVX2", "AVX512", "LLAMAFILE"]);
        assert_eq!(
            missing(&host, &isa, &["AVX", "AVX2", "AVX512", "AVX512_BF16"]),
            v(&["AVX512_BF16"])
        );
        // A feature the build cannot provide is not missing.
        assert!(missing(&host, &isa, &["AVX", "AVX2", "AVX512"]).is_empty());
        assert!(extra(&host, &isa).is_empty());
        let spr = v(&["AVX512_VNNI", "AVX_VNNI"]);
        assert!(missing(&spr, &v(&["AVX512_VNNI"]), &TRACKED).is_empty());
        // Untracked features (LLAMAFILE, SSE3) never count as extra.
        let low = v(&["AVX", "AVX2"]);
        assert_eq!(extra(&low, &v(&["AVX2", "AVX512_VBMI", "SSE3"])), v(&["AVX512_VBMI"]));
    }

    #[test]
    fn module_dir_order() {
        let (env, exe, built) = (Path::new("e"), Path::new("x"), Path::new("b"));
        let yes = |_: &Path| true;
        let no = |_: &Path| false;
        assert_eq!(resolve_module_dir(Some(env), Some(exe), built, yes), env);
        assert_eq!(resolve_module_dir(None, Some(exe), built, yes), exe);
        assert_eq!(resolve_module_dir(None, Some(exe), built, no), built);
        assert_eq!(resolve_module_dir(None, None, built, yes), built);
    }

    #[test]
    fn module_file_names() {
        if cfg!(windows) {
            assert_eq!(cpu_module_file_name("haswell"), "ggml-cpu-haswell.dll");
            assert_eq!(module_file_name("ggml-cuda"), "ggml-cuda.dll");
        } else {
            assert_eq!(cpu_module_file_name("haswell"), "libggml-cpu-haswell.so");
            assert_eq!(module_file_name("ggml-cuda"), "libggml-cuda.so");
        }
    }

    #[test]
    fn provided_is_subset() {
        assert!(provided().iter().all(|f| TRACKED.contains(f)));
        assert_eq!(
            provided().contains(&"AMX_INT8"),
            !cfg!(target_env = "msvc")
        );
    }
}
