"""Append-only audit trail for billing actions."""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class AuditEntry:
    at: str
    actor: str
    action: str
    subject: str
    details: dict = field(default_factory=dict)


class AuditLog:
    def __init__(self, path: Path | None = None, enabled: bool = True):
        self.path = path
        self.enabled = enabled
        self.entries: list[AuditEntry] = []

    def record(self, actor: str, action: str, subject: str, **details) -> AuditEntry | None:
        if not self.enabled:
            return None
        entry = AuditEntry(datetime.now(timezone.utc).isoformat(timespec="seconds"), actor, action, subject, details)
        self.entries.append(entry)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(entry), sort_keys=True) + "\n")
        return entry

    def for_subject(self, subject: str) -> list[AuditEntry]:
        return [e for e in self.entries if e.subject == subject]

    def by_action(self, action: str) -> list[AuditEntry]:
        return [e for e in self.entries if e.action == action]

    def last(self, n: int = 10) -> list[AuditEntry]:
        return self.entries[-n:]

    @classmethod
    def load(cls, path: Path) -> "AuditLog":
        log = cls(path)
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    log.entries.append(AuditEntry(**json.loads(line)))
        return log
