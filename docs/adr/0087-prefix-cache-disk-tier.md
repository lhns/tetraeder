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

## Addendum 2026-11-23: the restore reads in the background

Spilling stays as decided. Point 4 changes: `take()` of a disk entry no longer reads the file on the scheduler thread (a 20K-token Qwen3-8B snapshot is about 2.9 GB, about a second of NVMe, and every running sequence stalled for it).

- When placement finds its best match on disk, the scheduler starts a read on a second thread (`tetraeder-prefix-read`) into a snapshot pool buffer and the request waits in a restoring state. It holds no KV rows and no slot and does not block admission of the requests behind it; the other sequences keep decoding.
- The entry stays on disk, marked as being read: it is not offered to a second request and not evicted. When the read is in, the entry and its file are consumed and the request is placed like a RAM hit (the hit still counts as `source=disk`).
- Ordering: a restoring request keeps its queue position. Requests behind it are placed meanwhile; once its read is in, it is placed before any request still waiting behind it. If a better RAM match appears meanwhile, nothing changes.
- A read that fails gives a warning, drops the entry, and the request is placed without reuse of that entry (full prefill). It does not start a second disk read.
- A client that leaves during the read ends its request at the next step. The entry and its file stay (it was not consumed); the reader hands the buffer back to the pool when it finishes.
- At most two reads run at once, so at most two buffers sit outside the RAM budget. A request that finds the limit reached is placed without the disk entry.
- Metrics: `tetraeder.kv.prefix.restore.duration{source=disk}` is now the background file read; the copy into the KV cache is recorded only for RAM entries, so a disk restore adds no second sample. New gauge `tetraeder.kv.prefix.restoring` (requests waiting on a disk read), on the prefix cache panel. `PrefixDiskRestoreSlow` stays, now on the read time alone.
- Not measured: a paired run of disk restore against recompute is still open. The tests are CPU only, with a gated reader.

## Addendum 2026-11-23: review fixes

- Files are named `tetraeder-prefix-<pid>-<instance>-<id>.tkv`, the instance counting the tiers this process opened. A tier removes only its own files on drop, so the tier of a replan (`serve --plan auto` opens the next scheduler on the same directory) no longer deletes the live one's files. At startup a tier removes only files last written before this process first opened a tier, which are an earlier process's. This replaces point 5's "every `tetraeder-prefix-*` file"; the directory still belongs to one server.
- An aborted read keeps its entry marked as reading until the reader has replied (the next `poll` reaps it). The reader frees the buffer itself when the read was given up. The entry is not evicted or deleted under the reader, and the read counts against the limit of two until it is done.
- A spill that cannot make room, because the entries left are being read, is dropped; the disk budget is never exceeded.
- A backend that cannot cut a suffix (a recurrent model) takes a RAM or disk entry only when the prompt starts with the whole entry (`best_whole`); a partial match would need a cut, and taking the entry first destroyed it for nothing.
