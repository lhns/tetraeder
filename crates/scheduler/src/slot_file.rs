//! The file `POST /slots/{id}?action=save` writes (ADR 0090): the slot's tokens and the bytes of
//! its KV cache as the backend snapshots them. It is this server's own format, not llama.cpp's
//! `llama_state_seq_save_file`: the snapshot is the scheduler's (it also fills the prefix cache),
//! and the KV format is recorded so a file is never loaded into a cache of another type.

const MAGIC: &[u8; 4] = b"TTSS";
const VERSION: u32 = 1;

/// The file for `tokens` and their KV snapshot `state` of cache format `kv_format`.
pub fn encode(kv_format: &str, tokens: &[i32], state: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(32 + kv_format.len() + tokens.len() * 4 + state.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&(kv_format.len() as u32).to_le_bytes());
    out.extend_from_slice(kv_format.as_bytes());
    out.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
    for t in tokens {
        out.extend_from_slice(&t.to_le_bytes());
    }
    out.extend_from_slice(&(state.len() as u64).to_le_bytes());
    out.extend_from_slice(state);
    out
}

/// What a slot file holds.
#[derive(Debug, PartialEq, Eq)]
pub struct SlotFile<'a> {
    pub kv_format: String,
    pub tokens: Vec<i32>,
    pub state: &'a [u8],
}

fn next<'a>(data: &'a [u8], at: &mut usize, n: usize) -> Result<&'a [u8], String> {
    let end = at
        .checked_add(n)
        .filter(|&e| e <= data.len())
        .ok_or("the file is truncated")?;
    let s = &data[*at..end];
    *at = end;
    Ok(s)
}

/// Reads a file made by [`encode`]; an error says what is wrong with it.
pub fn decode(data: &[u8]) -> Result<SlotFile<'_>, String> {
    let mut at = 0usize;
    let word = |b: &[u8]| u32::from_le_bytes(b.try_into().expect("4 bytes"));
    let long = |b: &[u8]| u64::from_le_bytes(b.try_into().expect("8 bytes"));
    if next(data, &mut at, 4)? != MAGIC {
        return Err("not a slot save file".into());
    }
    let version = word(next(data, &mut at, 4)?);
    if version != VERSION {
        return Err(format!(
            "slot file version {version}, this server reads {VERSION}"
        ));
    }
    let n = word(next(data, &mut at, 4)?) as usize;
    let kv_format =
        String::from_utf8(next(data, &mut at, n)?.to_vec()).map_err(|_| "bad KV format name")?;
    let n_tokens =
        usize::try_from(long(next(data, &mut at, 8)?)).map_err(|_| "the file is truncated")?;
    let raw = next(
        data,
        &mut at,
        n_tokens.checked_mul(4).ok_or("the file is truncated")?,
    )?;
    let tokens = raw.chunks_exact(4).map(|c| word(c) as i32).collect();
    let n_state =
        usize::try_from(long(next(data, &mut at, 8)?)).map_err(|_| "the file is truncated")?;
    let state = next(data, &mut at, n_state)?;
    Ok(SlotFile {
        kv_format,
        tokens,
        state,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // hardware: none
    #[test]
    fn a_file_reads_back_what_was_written() {
        let f = encode("f16/f16", &[1, -2, 300], &[9, 8, 7]);
        let r = decode(&f).unwrap();
        assert_eq!(r.kv_format, "f16/f16");
        assert_eq!(r.tokens, [1, -2, 300]);
        assert_eq!(r.state, [9, 8, 7]);
        let bytes = encode("q8_0/q8_0", &[], &[]);
        let empty = decode(&bytes).unwrap();
        assert!(empty.tokens.is_empty() && empty.state.is_empty());
    }

    // hardware: none
    #[test]
    fn damaged_files_are_errors_not_panics() {
        let f = encode("f16/f16", &[1, 2], &[1, 2, 3]);
        for cut in 0..f.len() {
            assert!(decode(&f[..cut]).is_err(), "cut at {cut}");
        }
        assert!(
            decode(b"GGSQxxxxxxxxxxxx")
                .unwrap_err()
                .contains("not a slot save file")
        );
        let mut other = f.clone();
        other[4] = 9;
        assert!(decode(&other).unwrap_err().contains("version 9"));
        // a token count far beyond the file
        let mut huge = f.clone();
        let at = 4 + 4 + 4 + "f16/f16".len();
        huge[at..at + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(decode(&huge).is_err());
    }
}
