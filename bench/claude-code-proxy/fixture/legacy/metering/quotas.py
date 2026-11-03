"""Soft and hard usage limits with alert levels."""

from dataclasses import dataclass

from legacy.plans import Plan

ALERT_LEVELS = (50, 80, 100)


@dataclass(frozen=True)
class Quota:
    meter: str
    soft_limit: int  # alert, keep serving
    hard_limit: int | None  # block above this; None = never blocked


def default_quotas(plan: Plan) -> dict[str, Quota]:
    """Soft limit = the included allowance (0 means "no alert"), hard limit = 100x the allowance, if any."""
    out = {}
    for name, mp in plan.meters.items():
        if mp.included_units:
            out[name] = Quota(name, mp.included_units, mp.included_units * 100)
        else:
            out[name] = Quota(name, 0, None)
    return out


def percent_used(used: int, quota: Quota) -> int:
    if quota.soft_limit <= 0:
        return 0
    return used * 100 // quota.soft_limit


def alert_level(used: int, quota: Quota) -> int | None:
    """Highest alert threshold reached (50/80/100), or None."""
    pct = percent_used(used, quota)
    reached = [lvl for lvl in ALERT_LEVELS if pct >= lvl]
    return reached[-1] if reached else None


def is_blocked(used: int, quota: Quota) -> bool:
    return quota.hard_limit is not None and used > quota.hard_limit


def new_alerts(before: int, after: int, quota: Quota) -> list[int]:
    """Alert levels crossed when usage grows from `before` to `after`."""
    lo, hi = percent_used(before, quota), percent_used(after, quota)
    return [lvl for lvl in ALERT_LEVELS if lo < lvl <= hi]


def headroom(used: int, quota: Quota) -> int | None:
    """Units left before the hard limit; None when there is none."""
    if quota.hard_limit is None:
        return None
    return max(0, quota.hard_limit - used)


def evaluate(plan: Plan, usage: dict[str, int]) -> dict[str, tuple[int | None, bool]]:
    """meter -> (alert level, blocked) for every metered quota of the plan."""
    quotas = default_quotas(plan)
    return {m: (alert_level(usage.get(m, 0), q), is_blocked(usage.get(m, 0), q)) for m, q in quotas.items()}
