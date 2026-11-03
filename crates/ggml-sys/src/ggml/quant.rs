//! Row conversion between ggml storage types and f32 (ADR 0047). Pure functions on slices, safe
//! to call from several threads on disjoint buffers.

use super::{GgmlError, Result, Type};
use crate::raw as sys;

fn check_row(ty: Type, n_per_row: usize) -> Result<()> {
    let blck = ty.block_size() as usize;
    if n_per_row == 0 || !n_per_row.is_multiple_of(blck) {
        return Err(GgmlError::Invalid(format!(
            "{n_per_row} elements per row is not a multiple of the {} block size {blck}",
            ty.name()
        )));
    }
    Ok(())
}

/// Dequantizes `src` (whole blocks of `ty`) into `dst`; `dst.len()` is the element count and
/// must be a multiple of the block size.
pub fn to_f32(ty: Type, src: &[u8], dst: &mut [f32]) -> Result<()> {
    check_row(ty, dst.len())?;
    if src.len() != ty.row_size(dst.len() as i64) {
        return Err(GgmlError::Invalid(format!(
            "{} bytes for {} elements of {}",
            src.len(),
            dst.len(),
            ty.name()
        )));
    }
    if ty == Type::F32 {
        for (d, c) in dst.iter_mut().zip(src.chunks_exact(4)) {
            *d = f32::from_ne_bytes([c[0], c[1], c[2], c[3]]);
        }
        return Ok(());
    }
    // SAFETY: valid id; ggml returns a static traits struct.
    let to_float = unsafe { (*sys::ggml_get_type_traits(ty.raw())).to_float };
    let Some(f) = to_float else {
        return Err(GgmlError::Invalid(format!(
            "{} cannot be dequantized",
            ty.name()
        )));
    };
    // SAFETY: `src` holds exactly `dst.len()` elements of `ty` (checked above) and `dst` is
    // valid for that many floats; the function reads and writes only within them.
    unsafe { f(src.as_ptr().cast(), dst.as_mut_ptr(), dst.len() as i64) };
    Ok(())
}

/// Quantizes `n_rows` rows of `n_per_row` f32 values to `ty` (`src` is row-major) and returns
/// the bytes written, `n_rows * row_size`. `imatrix` has `n_per_row` weights when given and is
/// required by [`Type::requires_imatrix`] types. Thread-safe: call it on disjoint row ranges to
/// split a tensor.
pub fn quantize(
    ty: Type,
    src: &[f32],
    n_rows: usize,
    n_per_row: usize,
    imatrix: Option<&[f32]>,
    dst: &mut [u8],
) -> Result<usize> {
    check_row(ty, n_per_row)?;
    let row = ty.row_size(n_per_row as i64);
    if src.len() != n_rows * n_per_row || dst.len() != n_rows * row {
        return Err(GgmlError::Invalid(format!(
            "quantize to {}: {} floats and {} bytes for {n_rows} rows of {n_per_row}",
            ty.name(),
            src.len(),
            dst.len()
        )));
    }
    if imatrix.is_some_and(|m| m.len() != n_per_row) {
        return Err(GgmlError::Invalid(
            "imatrix length differs from the row".into(),
        ));
    }
    if ty.requires_imatrix() && imatrix.is_none() {
        return Err(GgmlError::Invalid(format!(
            "{} needs an importance matrix",
            ty.name()
        )));
    }
    if n_rows == 0 {
        return Ok(0);
    }
    // SAFETY: `src` holds n_rows * n_per_row floats and `dst` n_rows * row_size bytes (checked
    // above); start 0 keeps every offset inside them; ggml_quantize_chunk initializes its
    // tables under its own lock. The imatrix, when present, has n_per_row entries.
    let n = unsafe {
        sys::ggml_quantize_chunk(
            ty.raw(),
            src.as_ptr(),
            dst.as_mut_ptr().cast(),
            0,
            n_rows as i64,
            n_per_row as i64,
            imatrix.map_or(std::ptr::null(), <[f32]>::as_ptr),
        )
    };
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0)
            .collect()
    }

    #[test]
    fn roundtrip_error_shrinks_with_bits() {
        let (rows, cols) = (4, 256);
        let src = data(rows * cols);
        let mut last = f32::MAX;
        for name in ["q2_K", "q4_K", "q6_K", "q8_0", "f16"] {
            let ty = Type::from_name(name).unwrap();
            let mut q = vec![0u8; rows * ty.row_size(cols as i64)];
            assert_eq!(
                quantize(ty, &src, rows, cols, None, &mut q).unwrap(),
                q.len()
            );
            let mut back = vec![0f32; rows * cols];
            to_f32(ty, &q, &mut back).unwrap();
            let err = src
                .iter()
                .zip(&back)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(err < last, "{name}: {err} not below {last}");
            last = err;
        }
    }

    #[test]
    fn f32_and_bf16_targets() {
        let src = data(64);
        let mut out = vec![0u8; 64 * 4];
        quantize(Type::F32, &src, 1, 64, None, &mut out).unwrap();
        let mut back = vec![0f32; 64];
        to_f32(Type::F32, &out, &mut back).unwrap();
        assert_eq!(back, src);
        let mut out = vec![0u8; 64 * 2];
        quantize(Type::BF16, &src, 2, 32, None, &mut out).unwrap();
        to_f32(Type::BF16, &out, &mut back).unwrap();
        assert!(src.iter().zip(&back).all(|(a, b)| (a - b).abs() < 0.01));
    }

    #[test]
    fn rows_split_across_threads_equal_one_call() {
        let (rows, cols) = (8, 512);
        let src = data(rows * cols);
        let ty = Type::from_name("iq4_xs").unwrap();
        let row = ty.row_size(cols as i64);
        let mut whole = vec![0u8; rows * row];
        quantize(ty, &src, rows, cols, None, &mut whole).unwrap();
        let mut parts = vec![0u8; rows * row];
        std::thread::scope(|s| {
            for (d, x) in parts.chunks_mut(2 * row).zip(src.chunks(2 * cols)) {
                s.spawn(move || quantize(ty, x, 2, cols, None, d).unwrap());
            }
        });
        assert_eq!(whole, parts);
    }

    #[test]
    fn errors() {
        let q4 = Type::from_name("q4_K").unwrap();
        assert!(quantize(q4, &[0.0; 100], 1, 100, None, &mut [0; 144]).is_err());
        assert!(to_f32(q4, &[0; 10], &mut [0.0; 256]).is_err());
        let iq2 = Type::from_name("iq2_xxs").unwrap();
        assert!(quantize(iq2, &[0.0; 256], 1, 256, None, &mut [0; 66]).is_err());
    }
}
