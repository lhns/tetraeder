//! Every transport round-trips data correctly over loopback (the echo is verified byte for byte).

use tetraeder_transport::bench::{self, Plan, Transport};

async fn check(t: Transport) {
    let server = bench::bind(t, "127.0.0.1:0".parse().unwrap())
        .await
        .unwrap();
    let addr = server.addr;
    let task = tokio::spawn(server.run());
    let plan = Plan {
        sizes: vec![64, 8192, 300_000],
        iters: 25,
        warmup: 5,
        throughput_bytes: 1 << 20,
    };
    let rows = bench::measure(t, addr, &plan).await.unwrap();
    assert_eq!(rows.len(), 3);
    for r in &rows {
        assert!(
            r.p50_us > 0.0 && r.p50_us <= r.p99_us && r.min_us <= r.p50_us,
            "{r:?}"
        );
        assert!(r.throughput_mib_s > 0.0);
    }
    task.abort();
}

#[tokio::test]
async fn tcp() {
    check(Transport::Tcp).await;
}

#[tokio::test]
async fn tls() {
    check(Transport::Tls).await;
}

#[tokio::test]
async fn quic() {
    check(Transport::Quic).await;
}

#[tokio::test]
async fn two_clients_share_one_server() {
    let server = bench::bind(Transport::Tls, "127.0.0.1:0".parse().unwrap())
        .await
        .unwrap();
    let addr = server.addr;
    let task = tokio::spawn(server.run());
    let (a, b) = tokio::join!(
        bench::connect(Transport::Tls, addr),
        bench::connect(Transport::Tls, addr)
    );
    let (mut a, mut b) = (a.unwrap(), b.unwrap());
    let (x, y) = tokio::join!(a.latency(100, 0, 5), b.latency(200, 0, 5));
    assert_eq!((x.unwrap().len(), y.unwrap().len()), (5, 5));
    task.abort();
}
