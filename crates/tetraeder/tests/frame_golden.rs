//! Golden bytes of the worker frames (ADR 0091 step 1, note ii): the frame code moved into
//! `tetraeder-transport` must write exactly the bytes `remote_backend` wrote before. The
//! expected arrays were recorded from the code before the move.
//!
//! Hardware label: none.

use tetraeder::remote_backend::wire::{
    Info, Reply, Request, SeqState, State, WireError, WireInput,
};
use tetraeder_transport::frame::encode;

fn cases() -> Vec<(&'static str, Vec<u8>)> {
    let state = State {
        kv_free: 3100,
        seqs: vec![
            SeqState {
                len: 5,
                reserved: 10,
            },
            SeqState::default(),
        ],
        bytes_streamed: Some(4096),
    };
    let reply = |err, spans, info| Reply {
        err,
        spans,
        info,
        state: state.clone(),
        step_s: None,
    };
    vec![
        ("info", encode(&Request::Info, &[]).unwrap()),
        (
            "seq_reserve",
            encode(&Request::SeqReserve { seq: 1, n: 64 }, &[]).unwrap(),
        ),
        (
            "seq_rm",
            encode(&Request::SeqRm { seq: 3, keep: 7 }, &[]).unwrap(),
        ),
        (
            "seq_cp",
            encode(
                &Request::SeqCp {
                    src: 0,
                    dst: 2,
                    n: 9,
                },
                &[],
            )
            .unwrap(),
        ),
        (
            "decode",
            encode(
                &Request::Decode {
                    inputs: vec![
                        WireInput {
                            seq: 0,
                            tokens: vec![1, 2, 3],
                            all_logits: false,
                        },
                        WireInput {
                            seq: 1,
                            tokens: vec![-1, 40000],
                            all_logits: true,
                        },
                    ],
                    trace: None,
                },
                &[],
            )
            .unwrap(),
        ),
        (
            "seq_snapshot",
            encode(&Request::SeqSnapshot { seq: 2 }, &[]).unwrap(),
        ),
        (
            "seq_restore",
            encode(&Request::SeqRestore { seq: 1, n: 4 }, &[9, 8, 7, 6]).unwrap(),
        ),
        (
            "reply_info",
            encode(
                &reply(
                    None,
                    Vec::new(),
                    Some(Info {
                        n_vocab: 64,
                        n_seq_max: 4,
                        n_ctx_seq: 800,
                        kv_capacity: 3200,
                        kv_format: "f16".into(),
                        on_gpu: false,
                        kv_row_bytes: 128,
                        max_step_tokens: 512,
                        segment: None,
                    }),
                ),
                &[],
            )
            .unwrap(),
        ),
        (
            "reply_decode",
            encode(
                &reply(None, vec![(0, 1), (1, 3)], None),
                &1.5f32.to_le_bytes(),
            )
            .unwrap(),
        ),
        (
            "reply_kv_full",
            encode(
                &reply(
                    Some(WireError::KvFull {
                        seq: 3,
                        needed: 3201,
                        free: 12,
                    }),
                    Vec::new(),
                    None,
                ),
                &[],
            )
            .unwrap(),
        ),
        (
            "reply_sticky",
            encode(
                &reply(
                    Some(WireError::Failed {
                        msg: "CUDA error: injected".into(),
                        sticky: true,
                    }),
                    Vec::new(),
                    None,
                ),
                &[],
            )
            .unwrap(),
        ),
    ]
}

/// Recorded from `remote_backend::encode` before the frame code moved (hex of the whole frame).
const GOLDEN: &[(&str, &str)] = &[
    ("info", "130000000f0000007b2263616c6c223a22696e666f227d"),
    (
        "seq_reserve",
        "29000000250000007b2263616c6c223a227365715f72657365727665222c22736571223a312c226e223a36347d",
    ),
    (
        "seq_rm",
        "26000000220000007b2263616c6c223a227365715f726d222c22736571223a332c226b656570223a377d",
    ),
    (
        "seq_cp",
        "2b000000270000007b2263616c6c223a227365715f6370222c22737263223a302c22647374223a322c226e223a397d",
    ),
    (
        "decode",
        "7e0000007a0000007b2263616c6c223a226465636f6465222c22696e70757473223a5b7b22736571223a302c22746f6b656e73223a5b312c322c335d2c22616c6c5f6c6f67697473223a66616c73657d2c7b22736571223a312c22746f6b656e73223a5b2d312c34303030305d2c22616c6c5f6c6f67697473223a747275657d5d7d",
    ),
    (
        "seq_snapshot",
        "230000001f0000007b2263616c6c223a227365715f736e617073686f74222c22736571223a327d",
    ),
    (
        "seq_restore",
        "2c000000240000007b2263616c6c223a227365715f726573746f7265222c22736571223a312c226e223a347d09080706",
    ),
    (
        "reply_info",
        "130100000f0100007b22657272223a6e756c6c2c227370616e73223a5b5d2c22696e666f223a7b226e5f766f636162223a36342c226e5f7365715f6d6178223a342c226e5f6374785f736571223a3830302c226b765f6361706163697479223a333230302c226b765f666f726d6174223a22663136222c226f6e5f677075223a66616c73652c226b765f726f775f6279746573223a3132382c226d61785f737465705f746f6b656e73223a3531327d2c227374617465223a7b226b765f66726565223a333130302c2273657173223a5b7b226c656e223a352c227265736572766564223a31307d2c7b226c656e223a302c227265736572766564223a307d5d2c2262797465735f73747265616d6564223a343039367d7d",
    ),
    (
        "reply_decode",
        "9d000000950000007b22657272223a6e756c6c2c227370616e73223a5b5b302c315d2c5b312c335d5d2c22696e666f223a6e756c6c2c227374617465223a7b226b765f66726565223a333130302c2273657173223a5b7b226c656e223a352c227265736572766564223a31307d2c7b226c656e223a302c227265736572766564223a307d5d2c2262797465735f73747265616d6564223a343039367d7d0000c03f",
    ),
    (
        "reply_kv_full",
        "b6000000b20000007b22657272223a7b224b7646756c6c223a7b22736571223a332c226e6565646564223a333230312c2266726565223a31327d7d2c227370616e73223a5b5d2c22696e666f223a6e756c6c2c227374617465223a7b226b765f66726565223a333130302c2273657173223a5b7b226c656e223a352c227265736572766564223a31307d2c7b226c656e223a302c227265736572766564223a307d5d2c2262797465735f73747265616d6564223a343039367d7d",
    ),
    (
        "reply_sticky",
        "c1000000bd0000007b22657272223a7b224661696c6564223a7b226d7367223a2243554441206572726f723a20696e6a6563746564222c22737469636b79223a747275657d7d2c227370616e73223a5b5d2c22696e666f223a6e756c6c2c227374617465223a7b226b765f66726565223a333130302c2273657173223a5b7b226c656e223a352c227265736572766564223a31307d2c7b226c656e223a302c227265736572766564223a307d5d2c2262797465735f73747265616d6564223a343039367d7d",
    ),
];

// hardware: none
#[test]
fn the_moved_frame_code_writes_the_recorded_bytes() {
    let got = cases();
    assert_eq!(got.len(), GOLDEN.len());
    for ((name, bytes), (gname, hex)) in got.iter().zip(GOLDEN) {
        assert_eq!(name, gname);
        let want: Vec<u8> = (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect();
        assert_eq!(bytes, &want, "frame `{name}` changed");
    }
}

// hardware: none
#[test]
fn a_decode_without_a_trace_has_no_trace_field_and_one_with_it_carries_traceparent() {
    let inputs = vec![WireInput {
        seq: 0,
        tokens: vec![1],
        all_logits: false,
    }];
    let plain = encode(
        &Request::Decode {
            inputs: inputs.clone(),
            trace: None,
        },
        &[],
    )
    .unwrap();
    assert!(!String::from_utf8_lossy(&plain).contains("traceparent"));
    let tp = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
    let traced = encode(
        &Request::Decode {
            inputs,
            trace: Some(tp.into()),
        },
        &[],
    )
    .unwrap();
    let f = tetraeder_transport::frame::read_frame(&mut traced.as_slice())
        .unwrap()
        .unwrap();
    assert_eq!(
        tetraeder_transport::frame::traceparent(&f).as_deref(),
        Some(tp)
    );
}
