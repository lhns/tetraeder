use tetraeder_ggml_sys::ggml::GgmlError;

#[derive(Debug, thiserror::Error)]
pub enum ExecError {
    /// The program breaks the rules of ADR 0069 (every violation, see [`crate::validate`]).
    #[error("invalid program: {}", .0.join("; "))]
    Invalid(Vec<String>),
    #[error(transparent)]
    Ggml(#[from] GgmlError),
}

impl ExecError {
    pub(crate) fn invalid(msg: impl Into<String>) -> ExecError {
        ExecError::Invalid(vec![msg.into()])
    }
}
