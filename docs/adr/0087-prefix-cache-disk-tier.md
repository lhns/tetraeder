# 0087. Prefix cache disk tier

Status: Accepted (owner, 2026-11-23, ADR 0085 class 1)

## Context

- The prefix cache (ADR 0049, snapshot pool ADR 0074) keeps saved sequences in host RAM under `--cache-ram` and drops the oldest. A conversation that idles longer than the RAM holds recomputes its whole prompt.
- Strata parks conversations with an SSD spill (ADR 0085, `docs/notes/strata-parity.md`). A restore from NVMe costs far less than a prefill of the same tokens.

## Decision

1. Serve flags `--cache-disk <dir>` and `--cache-disk-mib <N>`; the default 0 is off and changes nothing. The directory is created if missing.
2. An entry the RAM budget evicts is written to one file (magic, version, format string, tokens, KV bytes) when the disk budget allows; the oldest disk entries go first. A prefix replaced by a longer sequence is dropped, not spilled. An entry larger than the disk budget is dropped.
3. Writes run on one background thread behind a queue of two. The queue holds the buffers outside the RAM budget, so a spill that finds it full is dropped. The scheduler step never waits for disk. Bytes are reserved against the disk budget when the write is queued; the entry is findable only once the write completed (written under a temporary name, then renamed).
4. `best()` searches RAM and completed disk entries with the same longest-common-prefix rule (RAM wins a tie). `take()` of a disk entry reads the file into a snapshot pool buffer and deletes it: a restored entry leaves the cache (ADR 0049). The file read runs on the scheduler thread and is part of the restore time.
5. Files are valid for this process: removed at startup (every `tetraeder-prefix-*` file in the directory, so the directory belongs to one server) and when the cache is dropped. A snapshot is only valid for the same model and KV format.
6. Failures degrade to today's behaviour: an unusable directory turns the tier off, a failed write or unreadable file drops the entry, each with a warning. The client never sees an error.
7. Metrics: `tetraeder.kv.prefix.hits` gets `source=disk`, `tetraeder.kv.prefix.bytes` a `tier` attribute (ram, disk), `tetraeder.kv.prefix.restore.duration` a `source` attribute. Alert `PrefixDiskRestoreSlow`: disk restore p90 over 2 s for 10 minutes. The prefix cache dashboard panel shows the disk hits and bytes.

## Consequences

- Idle conversations survive RAM pressure at the cost of NVMe writes (the full KV size per spill) and a disk restore on the request path.
- A busy writer drops spills instead of queueing without bound; the hit rate by source shows it.
- No persistence across restarts and no sharing of the directory between servers.
- Not measured: a paired speed run on a box (which restore time beats recomputing) is a later item. The tests are CPU only.
