//! Self-signed certificates and rustls configs for the spike. The client accepts any certificate:
//! this measures transport cost between two boxes we own, it is not a security design (mTLS is Phase 6).

use anyhow::Result;
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::crypto::{CryptoProvider, WebPkiSupportedAlgorithms};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer, ServerName, UnixTime};
use rustls::{ClientConfig, DigitallySignedStruct, ServerConfig, SignatureScheme};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

pub const ALPN: &[u8] = b"tetraeder-spike";
/// Name in the generated certificate and in the client's SNI.
pub const SERVER_NAME: &str = "localhost";

static CHACHA_FIRST: AtomicBool = AtomicBool::new(false);

/// Prefer ChaCha20-Poly1305 over AES-GCM in TLS and QUIC. Hosts without AES-NI (e.g. the QEMU dev VM)
/// are crypto-bound with AES; call before `bind` / `connect`. AES-128-GCM stays available, QUIC needs it for Initial packets.
pub fn prefer_chacha(on: bool) {
    CHACHA_FIRST.store(on, Ordering::Relaxed);
}

fn provider() -> Arc<CryptoProvider> {
    let mut p = rustls::crypto::ring::default_provider();
    if CHACHA_FIRST.load(Ordering::Relaxed) {
        p.cipher_suites
            .sort_by_key(|s| !format!("{:?}", s.suite()).contains("CHACHA20"));
    }
    Arc::new(p)
}

/// Fresh self-signed certificate and key, generated at startup.
pub fn self_signed() -> Result<(CertificateDer<'static>, PrivateKeyDer<'static>)> {
    let ck = rcgen::generate_simple_self_signed(vec![SERVER_NAME.to_string()])?;
    let key = PrivatePkcs8KeyDer::from(ck.key_pair.serialize_der());
    Ok((ck.cert.der().clone(), key.into()))
}

pub fn server_config() -> Result<ServerConfig> {
    let (cert, key) = self_signed()?;
    let mut cfg = ServerConfig::builder_with_provider(provider())
        .with_protocol_versions(&[&rustls::version::TLS13])?
        .with_no_client_auth()
        .with_single_cert(vec![cert], key)?;
    cfg.alpn_protocols = vec![ALPN.to_vec()];
    Ok(cfg)
}

pub fn client_config() -> Result<ClientConfig> {
    let p = provider();
    let mut cfg = ClientConfig::builder_with_provider(p.clone())
        .with_protocol_versions(&[&rustls::version::TLS13])?
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(AcceptAny(p.signature_verification_algorithms)))
        .with_no_client_auth();
    cfg.alpn_protocols = vec![ALPN.to_vec()];
    Ok(cfg)
}

#[derive(Debug)]
struct AcceptAny(WebPkiSupportedAlgorithms);

impl ServerCertVerifier for AcceptAny {
    fn verify_server_cert(
        &self,
        _: &CertificateDer<'_>,
        _: &[CertificateDer<'_>],
        _: &ServerName<'_>,
        _: &[u8],
        _: UnixTime,
    ) -> Result<ServerCertVerified, rustls::Error> {
        Ok(ServerCertVerified::assertion())
    }
    fn verify_tls12_signature(
        &self,
        m: &[u8],
        c: &CertificateDer<'_>,
        d: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls12_signature(m, c, d, &self.0)
    }
    fn verify_tls13_signature(
        &self,
        m: &[u8],
        c: &CertificateDer<'_>,
        d: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(m, c, d, &self.0)
    }
    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.0.supported_schemes()
    }
}
