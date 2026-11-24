//! Node identity and peer verification (ADR 0091, "Identity and mTLS").
//!
//! A node's identity is the id in the URI subject alternative name `tetraeder-node:<id>` of its
//! certificate, never an IP or DNS name (Vast addresses are NAT mappings). A peer is accepted when
//! its certificate chains to one of the trust anchors and names a configured node. An anchor is
//! either a node's own self-signed certificate (pinning: only that exact certificate is
//! accepted) or a CA certificate (every certificate it signs is accepted).

use crate::NodeId;
use anyhow::{Context, Result, anyhow, bail};
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::crypto::{CryptoProvider, WebPkiSupportedAlgorithms};
use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, ServerName, TrustAnchor, UnixTime};
use rustls::server::danger::{ClientCertVerified, ClientCertVerifier};
use rustls::{DigitallySignedStruct, DistinguishedName, SignatureScheme};
use std::collections::HashSet;
use std::sync::Arc;
use webpki::{EndEntityCert, KeyUsage};

pub const SAN_PREFIX: &str = "tetraeder-node:";

/// This node's certificate and private key. The key is never printed: `Debug` shows the id only.
pub struct Identity {
    pub cert: CertificateDer<'static>,
    pub key: PrivateKeyDer<'static>,
}

impl std::fmt::Debug for Identity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Identity")
            .field("node", &node_id_of(&self.cert).ok())
            .finish_non_exhaustive()
    }
}

impl Identity {
    /// From the PEM text of a certificate and its private key.
    pub fn from_pem(cert_pem: &[u8], key_pem: &[u8]) -> Result<Self> {
        let cert = CertificateDer::from_pem_slice(cert_pem).context("certificate PEM")?;
        let key = PrivateKeyDer::from_pem_slice(key_pem).context("private key PEM")?;
        node_id_of(&cert).map_err(|e| anyhow!(e))?;
        Ok(Self { cert, key })
    }

    pub fn node_id(&self) -> Result<NodeId> {
        node_id_of(&self.cert).map_err(|e| anyhow!(e))
    }
}

/// Every certificate in a PEM file (an anchor file holds one node's certificate or the CA's).
pub fn certs_from_pem(pem: &[u8]) -> Result<Vec<CertificateDer<'static>>> {
    let certs = CertificateDer::pem_slice_iter(pem)
        .collect::<std::result::Result<Vec<_>, _>>()
        .context("certificate PEM")?;
    if certs.is_empty() {
        bail!("no certificate in the PEM file");
    }
    Ok(certs)
}

/// A node id may hold letters, digits and `.`, `_`, `-`: it ends up in a URI.
pub fn check_node_id(id: &str) -> Result<()> {
    let ok = !id.is_empty()
        && id != "."
        && id != ".."
        && id.len() <= 63
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'));
    if ok {
        Ok(())
    } else {
        bail!("`{id}` is not a node id (1 to 63 of letters, digits, `.`, `_`, `-`)")
    }
}

/// The node id a certificate names: its one URI SAN `tetraeder-node:<id>`.
pub fn node_id_of(cert: &CertificateDer<'_>) -> std::result::Result<NodeId, String> {
    use x509_parser::extensions::{GeneralName, ParsedExtension};
    let (_, x) = x509_parser::parse_x509_certificate(cert.as_ref())
        .map_err(|e| format!("not an X.509 certificate: {e}"))?;
    let mut ids = Vec::new();
    for ext in x.extensions() {
        if let ParsedExtension::SubjectAlternativeName(san) = ext.parsed_extension() {
            for n in &san.general_names {
                if let GeneralName::URI(u) = n
                    && let Some(id) = u.strip_prefix(SAN_PREFIX)
                {
                    ids.push(id.to_owned());
                }
            }
        }
    }
    match ids.as_slice() {
        [id] if check_node_id(id).is_ok() => Ok(NodeId(id.clone())),
        [] => Err(format!("the certificate has no `{SAN_PREFIX}<id>` name")),
        _ => Err(format!(
            "the certificate must name exactly one node `{SAN_PREFIX}<id>`"
        )),
    }
}

/// Why a certificate was refused; `reason` is the `reason` attribute of
/// `tetraeder.cluster.tls.rejected`.
#[derive(Debug)]
pub struct Reject {
    pub reason: &'static str,
    pub msg: String,
}

impl std::fmt::Display for Reject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.reason, self.msg)
    }
}

impl std::error::Error for Reject {}

impl From<Reject> for rustls::Error {
    fn from(r: Reject) -> Self {
        rustls::CertificateError::Other(rustls::OtherError(Arc::new(r))).into()
    }
}

/// The `reason` of a refused handshake, from the error the TLS stack returned.
pub fn reject_reason(e: &std::io::Error) -> &'static str {
    let Some(tls) = e.get_ref().and_then(|i| i.downcast_ref::<rustls::Error>()) else {
        return "handshake";
    };
    match tls {
        rustls::Error::InvalidCertificate(rustls::CertificateError::Other(o)) => {
            o.0.downcast_ref::<Reject>()
                .map_or("untrusted", |r| r.reason)
        }
        rustls::Error::InvalidCertificate(_) | rustls::Error::NoCertificatesPresented => {
            "untrusted"
        }
        _ => "handshake",
    }
}

#[derive(Clone, Copy)]
enum Usage {
    Server,
    Client,
}

/// Trust anchors plus the configured node ids. An anchor that is a CA certificate (basic
/// constraints `cA`) vouches for every certificate it signs. Any other anchor is a pinned node
/// certificate: webpki does not check `cA` on an anchor, so the key of a pinned node could sign
/// a certificate for another node. A pinned certificate therefore vouches only for itself: the
/// peer's certificate must be the same bytes.
pub struct Anchors {
    cas: Vec<TrustAnchor<'static>>,
    pinned: Vec<(CertificateDer<'static>, TrustAnchor<'static>)>,
    nodes: HashSet<NodeId>,
    algs: WebPkiSupportedAlgorithms,
}

/// A CA certificate: `cA` is set and it names no node. A node certificate stays pinned even if a
/// tool marked it `cA` (openssl self-signed certificates default to it).
fn is_ca(cert: &CertificateDer<'_>) -> bool {
    node_id_of(cert).is_err()
        && x509_parser::parse_x509_certificate(cert.as_ref())
            .ok()
            .and_then(|(_, x)| x.basic_constraints().ok().flatten().map(|b| b.value.ca))
            .unwrap_or(false)
}

impl Anchors {
    pub fn new(
        certs: &[CertificateDer<'static>],
        nodes: &[NodeId],
        algs: WebPkiSupportedAlgorithms,
    ) -> Result<Self> {
        let (mut cas, mut pinned) = (Vec::new(), Vec::new());
        for c in certs {
            let a = webpki::anchor_from_trusted_cert(c)
                .map(|a| a.to_owned())
                .map_err(|e| anyhow!("not usable as a trust anchor: {e}"))?;
            if is_ca(c) {
                cas.push(a);
            } else {
                pinned.push((c.clone(), a));
            }
        }
        if cas.is_empty() && pinned.is_empty() {
            bail!("no trust anchor");
        }
        Ok(Self {
            cas,
            pinned,
            nodes: nodes.iter().cloned().collect(),
            algs,
        })
    }

    fn verify(
        &self,
        ee: &CertificateDer<'_>,
        intermediates: &[CertificateDer<'_>],
        now: UnixTime,
        usage: Usage,
    ) -> std::result::Result<NodeId, Reject> {
        let cert = EndEntityCert::try_from(ee).map_err(|e| Reject {
            reason: "untrusted",
            msg: e.to_string(),
        })?;
        let usage = match usage {
            Usage::Server => KeyUsage::server_auth(),
            Usage::Client => KeyUsage::client_auth(),
        };
        let one;
        let anchors: &[TrustAnchor<'static>] =
            if let Some((_, a)) = self.pinned.iter().find(|(c, _)| c.as_ref() == ee.as_ref()) {
                one = [a.clone()];
                &one
            } else {
                &self.cas
            };
        cert.verify_for_usage(
            self.algs.all,
            anchors,
            intermediates,
            now,
            usage,
            None,
            None,
        )
        .map_err(|e| Reject {
            reason: "untrusted",
            msg: e.to_string(),
        })?;
        let id = node_id_of(ee).map_err(|msg| Reject {
            reason: "no_identity",
            msg,
        })?;
        if self.nodes.contains(&id) {
            Ok(id)
        } else {
            Err(Reject {
                reason: "unknown_node",
                msg: format!("`{id}` is not a configured node"),
            })
        }
    }
}

/// Verifies the server's certificate for a client; the name check is the node id (the caller
/// compares it with the node it dialed), so the server name is not used.
#[derive(Debug)]
pub(crate) struct PeerVerifier {
    anchors: Arc<Anchors>,
    algs: WebPkiSupportedAlgorithms,
}

impl std::fmt::Debug for Anchors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Anchors")
            .field("cas", &self.cas.len())
            .field("pinned", &self.pinned.len())
            .field("nodes", &self.nodes)
            .finish_non_exhaustive()
    }
}

impl PeerVerifier {
    pub(crate) fn new(anchors: Arc<Anchors>, provider: &CryptoProvider) -> Arc<Self> {
        Arc::new(Self {
            anchors,
            algs: provider.signature_verification_algorithms,
        })
    }
}

impl ServerCertVerifier for PeerVerifier {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp: &[u8],
        now: UnixTime,
    ) -> std::result::Result<ServerCertVerified, rustls::Error> {
        self.anchors
            .verify(end_entity, intermediates, now, Usage::Server)?;
        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        m: &[u8],
        c: &CertificateDer<'_>,
        d: &DigitallySignedStruct,
    ) -> std::result::Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls12_signature(m, c, d, &self.algs)
    }

    fn verify_tls13_signature(
        &self,
        m: &[u8],
        c: &CertificateDer<'_>,
        d: &DigitallySignedStruct,
    ) -> std::result::Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(m, c, d, &self.algs)
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.algs.supported_schemes()
    }
}

impl ClientCertVerifier for PeerVerifier {
    fn root_hint_subjects(&self) -> &[DistinguishedName] {
        &[]
    }

    fn verify_client_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        intermediates: &[CertificateDer<'_>],
        now: UnixTime,
    ) -> std::result::Result<ClientCertVerified, rustls::Error> {
        self.anchors
            .verify(end_entity, intermediates, now, Usage::Client)?;
        Ok(ClientCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        m: &[u8],
        c: &CertificateDer<'_>,
        d: &DigitallySignedStruct,
    ) -> std::result::Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls12_signature(m, c, d, &self.algs)
    }

    fn verify_tls13_signature(
        &self,
        m: &[u8],
        c: &CertificateDer<'_>,
        d: &DigitallySignedStruct,
    ) -> std::result::Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(m, c, d, &self.algs)
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.algs.supported_schemes()
    }
}

/// A key pair and a self-signed certificate for `id`, as PEM text. The caller writes the key to a
/// file only the owner can read ([`write_private`]) and prints nothing of it.
pub struct Generated {
    pub cert_pem: String,
    pub key_pem: String,
}

pub fn generate_self_signed(id: &str) -> Result<Generated> {
    check_node_id(id)?;
    let key = rcgen::KeyPair::generate()?;
    let cert = node_params(id)?.self_signed(&key)?;
    Ok(Generated {
        cert_pem: cert.pem(),
        key_pem: key.serialize_pem(),
    })
}

fn node_params(id: &str) -> Result<rcgen::CertificateParams> {
    let mut p = rcgen::CertificateParams::default();
    p.subject_alt_names = vec![rcgen::SanType::URI(
        format!("{SAN_PREFIX}{id}")
            .try_into()
            .map_err(|e| anyhow!("{e:?}"))?,
    )];
    p.distinguished_name
        .push(rcgen::DnType::CommonName, format!("tetraeder node {id}"));
    // Long-lived: pinned in `cluster.toml`, renewed by running the command again.
    p.not_before = rcgen::date_time_ymd(2024, 1, 1);
    p.not_after = rcgen::date_time_ymd(2044, 1, 1);
    Ok(p)
}

/// Writes `bytes` to a new file only its owner can read. Unix: mode 0600 from creation.
/// Windows: the inherited permissions are replaced by a grant to the current user (`icacls`);
/// if that fails the file is removed and the error returned, so a key is never left readable.
pub fn write_private(path: &std::path::Path, bytes: &[u8], overwrite: bool) -> Result<()> {
    use std::io::Write;
    let mut o = std::fs::OpenOptions::new();
    o.write(true);
    if overwrite {
        o.create(true).truncate(true);
    } else {
        o.create_new(true);
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        o.mode(0o600);
    }
    let mut f = o
        .open(path)
        .with_context(|| format!("cannot create {}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        f.set_permissions(std::fs::Permissions::from_mode(0o600))?;
    }
    #[cfg(windows)]
    if let Err(e) = restrict_windows(path) {
        drop(f);
        let _ = std::fs::remove_file(path);
        return Err(e);
    }
    f.write_all(bytes)?;
    f.flush()?;
    Ok(())
}

#[cfg(windows)]
fn restrict_windows(path: &std::path::Path) -> Result<()> {
    let user = std::env::var("USERNAME").context("USERNAME is not set")?;
    let domain = std::env::var("USERDOMAIN").ok();
    let who = domain.map_or(user.clone(), |d| format!("{d}\\{user}"));
    let out = std::process::Command::new("icacls")
        .arg(path)
        .args(["/inheritance:r", "/grant:r"])
        .arg(format!("{who}:(F)"))
        .output()
        .context("cannot run icacls to restrict the key file")?;
    if !out.status.success() {
        bail!("icacls could not restrict {}", path.display());
    }
    Ok(())
}

/// Certificates for tests and rehearsals: a CA and leaves it signed. Production uses
/// [`generate_self_signed`] (owner answer 2, ADR 0091).
pub mod testing {
    use super::{Generated, node_params};
    use anyhow::Result;

    pub struct Ca {
        pub cert_pem: String,
        cert: rcgen::Certificate,
        key: rcgen::KeyPair,
    }

    pub fn ca(name: &str) -> Result<Ca> {
        let key = rcgen::KeyPair::generate()?;
        let mut p = rcgen::CertificateParams::default();
        p.is_ca = rcgen::IsCa::Ca(rcgen::BasicConstraints::Unconstrained);
        p.distinguished_name
            .push(rcgen::DnType::CommonName, name.to_owned());
        p.not_before = rcgen::date_time_ymd(2024, 1, 1);
        p.not_after = rcgen::date_time_ymd(2044, 1, 1);
        let cert = p.self_signed(&key)?;
        Ok(Ca {
            cert_pem: cert.pem(),
            cert,
            key,
        })
    }

    impl Ca {
        pub fn issue(&self, id: &str) -> Result<Generated> {
            let key = rcgen::KeyPair::generate()?;
            let cert = node_params(id)?.signed_by(&key, &self.cert, &self.key)?;
            Ok(Generated {
                cert_pem: cert.pem(),
                key_pem: key.serialize_pem(),
            })
        }
    }

    /// A certificate naming `id`, signed by the key of `signer` (a node's self-signed
    /// certificate, so not a CA): what a holder of that key can mint.
    pub fn forge_signed_by(signer_id: &str, signer: &Generated, id: &str) -> Result<Generated> {
        let skey = rcgen::KeyPair::from_pem(&signer.key_pem)?;
        let scert = node_params(signer_id)?.self_signed(&skey)?;
        let key = rcgen::KeyPair::generate()?;
        let cert = node_params(id)?.signed_by(&key, &scert, &skey)?;
        Ok(Generated {
            cert_pem: cert.pem(),
            key_pem: key.serialize_pem(),
        })
    }

    /// A certificate whose name is not a node id (no `tetraeder-node:` SAN).
    pub fn nameless() -> Result<Generated> {
        let key = rcgen::KeyPair::generate()?;
        let mut p = rcgen::CertificateParams::new(vec!["example.invalid".to_owned()])?;
        p.not_before = rcgen::date_time_ymd(2024, 1, 1);
        p.not_after = rcgen::date_time_ymd(2044, 1, 1);
        let cert = p.self_signed(&key)?;
        Ok(Generated {
            cert_pem: cert.pem(),
            key_pem: key.serialize_pem(),
        })
    }
}
