//! `tetraeder cluster ...`: tooling for a multi-node cluster (ADR 0091).

use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use std::path::PathBuf;
use tetraeder_transport::identity::{generate_self_signed, write_private};

#[derive(Subcommand)]
pub enum ClusterCommand {
    /// Make this node's key pair and self-signed certificate. Run it on the box: the key is
    /// written there and never leaves it. Put the certificate (`cert.pem`) in `cluster.toml` as
    /// the node's anchor on every node.
    Cert(CertArgs),
}

#[derive(Args)]
pub struct CertArgs {
    /// The node id, the `tetraeder-node:<id>` name in the certificate.
    #[arg(long)]
    pub node: String,
    /// Directory for `key.pem` (owner-only) and `cert.pem`; created if missing.
    #[arg(long)]
    pub out: PathBuf,
    /// Replace existing files (a node that changes its key must be re-pinned everywhere).
    #[arg(long)]
    pub force: bool,
}

pub fn run(cmd: &ClusterCommand) -> Result<()> {
    match cmd {
        ClusterCommand::Cert(a) => cert(a),
    }
}

fn cert(a: &CertArgs) -> Result<()> {
    let g = generate_self_signed(&a.node)?;
    std::fs::create_dir_all(&a.out)
        .with_context(|| format!("cannot create {}", a.out.display()))?;
    let (key, cert) = (a.out.join("key.pem"), a.out.join("cert.pem"));
    // The key first: if it cannot be written with restricted access, no certificate is left
    // behind that has no key.
    write_private(&key, g.key_pem.as_bytes(), a.force)?;
    if let Err(e) = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .create_new(!a.force)
        .truncate(true)
        .open(&cert)
        .and_then(|mut f| std::io::Write::write_all(&mut f, g.cert_pem.as_bytes()))
    {
        let _ = std::fs::remove_file(&key);
        return Err(anyhow::Error::new(e).context(format!("cannot write {}", cert.display())));
    }
    println!("node {}: certificate {}", a.node, cert.display());
    println!(
        "private key {} (owner only, keep it on this box)",
        key.display()
    );
    Ok(())
}
