//! `tetraeder cluster cert --node <id> --out <dir>` (ADR 0091 step 1, note iii): a key pair and a
//! self-signed certificate naming the node, the key restricted to its owner and never printed.
//!
//! Hardware label: none.

use std::path::PathBuf;
use std::process::Command;
use tetraeder_transport::Identity;

fn out_dir(name: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("cluster-cert-{name}"));
    let _ = std::fs::remove_dir_all(&d);
    d
}

fn cert(dir: &std::path::Path, node: &str, force: bool) -> std::process::Output {
    let mut c = Command::new(env!("CARGO_BIN_EXE_tetraeder"));
    c.args(["cluster", "cert", "--node", node, "--out"])
        .arg(dir);
    if force {
        c.arg("--force");
    }
    c.output().unwrap()
}

// hardware: none
#[test]
fn it_writes_a_key_and_a_certificate_that_name_the_node_and_prints_no_key() {
    let dir = out_dir("ok");
    let o = cert(&dir, "node-a", false);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let shown = format!(
        "{}{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    );
    assert!(!shown.contains("PRIVATE"), "the key was printed");
    let key = std::fs::read(dir.join("key.pem")).unwrap();
    let pem = std::fs::read(dir.join("cert.pem")).unwrap();
    assert!(String::from_utf8_lossy(&key).contains("PRIVATE KEY"));
    let id = Identity::from_pem(&pem, &key).unwrap();
    assert_eq!(id.node_id().unwrap().0, "node-a");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(dir.join("key.pem"))
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o600);
    }
}

// hardware: none
#[test]
fn it_does_not_replace_a_key_without_force_and_rejects_a_bad_id() {
    let dir = out_dir("again");
    assert!(cert(&dir, "n1", false).status.success());
    let key = std::fs::read(dir.join("key.pem")).unwrap();
    let again = cert(&dir, "n1", false);
    assert!(!again.status.success());
    assert_eq!(std::fs::read(dir.join("key.pem")).unwrap(), key);
    assert!(cert(&dir, "n1", true).status.success());
    assert_ne!(std::fs::read(dir.join("key.pem")).unwrap(), key);
    let bad = cert(&out_dir("bad"), "has space", false);
    assert!(!bad.status.success());
}
