//! Direct reads of unaligned ranges return the bytes a buffered read returns (ADR 0031), on
//! Windows (FILE_FLAG_NO_BUFFERING, overlapped) and Linux (O_DIRECT, io_uring or thread pool).
//! The test file lives under the target directory, not /tmp, which is often tmpfs (no O_DIRECT).

use proptest::prelude::*;
use std::fs;
use std::path::PathBuf;
use std::sync::OnceLock;
use tetraeder_memory::{DirectFile, IoOptions, ReadReq};

/// Not a multiple of the 4 KiB block, and several blocks long.
const SIZE: usize = 5 * 4096 + 1234;

fn data() -> &'static (PathBuf, Vec<u8>) {
    static DATA: OnceLock<(PathBuf, Vec<u8>)> = OnceLock::new();
    DATA.get_or_init(|| {
        let mut x = 0x2545_F491_4F6C_DD1Du64;
        let bytes: Vec<u8> = (0..SIZE)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x >> 24) as u8
            })
            .collect();
        let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("dio-test.bin");
        fs::write(&path, &bytes).unwrap();
        (path, bytes)
    })
}

fn read(f: &DirectFile, offset: u64, len: usize) -> Vec<u8> {
    let mut out = vec![0u8; len];
    f.read_at(offset, &mut out).unwrap();
    out
}

#[test]
fn reports_how_the_file_is_read() {
    let (path, _) = data();
    let f = DirectFile::open(path).unwrap();
    eprintln!(
        "direct={} backend={} fallback={:?}",
        f.is_direct(),
        f.backend(),
        f.fallback()
    );
    assert_eq!(f.len(), SIZE as u64);
    // Direct, or buffered with the reason.
    assert!(f.is_direct() || f.fallback().is_some());
    if cfg!(windows) {
        assert!(f.is_direct(), "{:?}", f.fallback());
        assert_eq!(f.backend(), "overlapped");
    }
}

#[test]
fn unaligned_ranges_equal_buffered_reads() {
    let (path, bytes) = data();
    let f = DirectFile::open(path).unwrap();
    let cases = [
        (0u64, 1usize),
        (0, 4096),
        (1, 4095),
        (5, 100),
        (4095, 2),
        (4096, 4096),
        (4000, 5000),
        (100, 3 * 4096 + 7),
        (SIZE as u64 - 1, 1),
        (SIZE as u64 - 1234, 1234),
        (SIZE as u64 - 10, 10),
        (3, SIZE - 3),
        (0, SIZE),
    ];
    for (offset, len) in cases {
        let got = read(&f, offset, len);
        let want = &bytes[offset as usize..offset as usize + len];
        assert_eq!(got, want, "range {offset}+{len}");
    }
    // The empty range is fine anywhere inside the file.
    assert!(read(&f, 17, 0).is_empty());
}

#[test]
fn batch_reads_many_ranges_at_once() {
    let (path, bytes) = data();
    let f = DirectFile::open(path).unwrap();
    let ranges = [
        (7u64, 300usize),
        (4090, 20),
        (8192, 4096),
        (20000, 1000),
        (0, 1),
    ];
    let mut bufs: Vec<Vec<u8>> = ranges.iter().map(|&(_, n)| vec![0; n]).collect();
    let mut reqs: Vec<ReadReq> = ranges
        .iter()
        .zip(bufs.iter_mut())
        .map(|(&(offset, _), dst)| ReadReq { offset, dst })
        .collect();
    let stats = f.read_batch(&mut reqs).unwrap();
    for (&(offset, len), got) in ranges.iter().zip(&bufs) {
        assert_eq!(
            got.as_slice(),
            &bytes[offset as usize..offset as usize + len]
        );
    }
    let asked: usize = ranges.iter().map(|r| r.1).sum();
    assert_eq!(stats.requested, asked as u64);
    if f.is_direct() {
        // Whole aligned blocks came off the device.
        assert!(stats.device >= stats.requested);
    }
}

#[test]
fn reading_past_the_end_is_an_error() {
    let (path, _) = data();
    let f = DirectFile::open(path).unwrap();
    let mut dst = [0u8; 2];
    let e = f.read_at(SIZE as u64 - 1, &mut dst).unwrap_err();
    assert_eq!(e.kind(), std::io::ErrorKind::UnexpectedEof);
    let e = f.read_at(u64::MAX - 1, &mut dst).unwrap_err();
    assert_eq!(e.kind(), std::io::ErrorKind::UnexpectedEof);
}

#[test]
fn buffered_fallback_is_reported_and_correct() {
    let (path, bytes) = data();
    let f = DirectFile::open_with(
        path,
        IoOptions {
            buffered: true,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(!f.is_direct());
    assert!(f.fallback().is_some());
    assert_eq!(read(&f, 4093, 9000), &bytes[4093..4093 + 9000]);
}

#[test]
fn missing_file_is_an_error_not_a_fallback() {
    let e =
        DirectFile::open(&PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("nope.bin")).unwrap_err();
    assert_eq!(e.kind(), std::io::ErrorKind::NotFound);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn any_range_equals_the_buffered_read(offset in 0usize..SIZE, len in 0usize..9000) {
        let (path, bytes) = data();
        let len = len.min(SIZE - offset);
        let f = DirectFile::open(path).unwrap();
        prop_assert_eq!(read(&f, offset as u64, len), bytes[offset..offset + len].to_vec());
    }
}
