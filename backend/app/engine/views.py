"""Deterministic view-models for the ward and contractor pages. No LLM arithmetic."""
from __future__ import annotations


import pandas as pd

from ..loader import Data
from .flags import Flag

OVER_RATE_X = 1.5
SMALL_SAMPLE = 30
ALERT_ORDER = {"Green": 0, "Yellow": 1, "Orange": 2, "Red": 3}


def ward_slug(code: str) -> str:
    """G/S -> g-s (slugs avoid '/' in URLs)."""
    return code.lower().replace("/", "-")


def money(x: float) -> str:
    """₹3.07 Cr / ₹45 L / ₹12,345 (same rules as frontend/lib/format.ts)."""
    def trim(v: float, d: int) -> str:
        s = f"{v:.{d}f}"
        return s.rstrip("0").rstrip(".") if "." in s else s

    if x >= 1e7:
        return f"₹{trim(x / 1e7, 2)} Cr"
    if x >= 1e5:
        return f"₹{trim(x / 1e5, 1)} L"
    return f"₹{x:,.0f}"


def _split(g: pd.DataFrame) -> tuple[float, float]:
    failed = float(g.loc[g.reopened, "cost_inr"].sum())
    return float(g.cost_inr.sum()) - failed, failed


# ------------------------------------------------------------------ flow
def ward_flow(data: Data, wo: pd.DataFrame, ward_code: str, ward_name: str, budget: float) -> dict:
    w = wo[wo.ward_code == ward_code]
    held, failed = _split(w)
    names = data.contractors.set_index("contractor_id")["contractor_name"].to_dict()

    nodes: list[dict] = [dict(id="spend", label="Ward spend", kind="spend", value_inr=float(w.cost_inr.sum()))]
    idx: dict[str, int] = {"spend": 0}

    def add(node: dict) -> int:
        idx[node["id"]] = len(nodes)
        nodes.append(node)
        return idx[node["id"]]

    by_c = w.groupby("contractor_id").cost_inr.sum().sort_values(ascending=False)
    for cid, v in by_c.items():
        add(dict(id=f"c:{cid}", label=names.get(cid, cid), kind="contractor", value_inr=float(v), contractor_id=cid))
    by_m = w.groupby("repair_method").cost_inr.sum().sort_values(ascending=False)
    for m, v in by_m.items():
        add(dict(id=f"m:{m}", label=m, kind="method", value_inr=float(v)))
    add(dict(id="o:held", label="Held", kind="outcome", value_inr=held, outcome="held"))
    add(dict(id="o:failed", label="Didn't hold", kind="outcome", value_inr=failed, outcome="failed"))

    links: list[dict] = []
    for cid, v in by_c.items():
        links.append(dict(source=0, target=idx[f"c:{cid}"], value=float(v), outcome=None))
    for (cid, m), v in w.groupby(["contractor_id", "repair_method"]).cost_inr.sum().items():
        links.append(dict(source=idx[f"c:{cid}"], target=idx[f"m:{m}"], value=float(v), outcome=None))
    for (m, was_failed), v in w.groupby(["repair_method", "reopened"]).cost_inr.sum().items():
        out = "failed" if was_failed else "held"
        links.append(dict(source=idx[f"m:{m}"], target=idx[f"o:{out}"], value=float(v), outcome=out))

    def rows(key: str, label_key: str, labeler=lambda x: x) -> list[dict]:
        out = []
        for k, g in w.groupby(key):
            h, f = _split(g)
            out.append({label_key: k, **({"contractor_name": labeler(k)} if key == "contractor_id" else {}),
                        "spend_inr": h + f, "held_inr": h, "failed_inr": f,
                        "failed_pct": f / (h + f) * 100 if h + f else 0.0, "work_orders": int(len(g))})
        return sorted(out, key=lambda r: -r["spend_inr"])

    return dict(
        ward_code=ward_code, ward_slug=ward_slug(ward_code), ward_name=ward_name,
        budget_inr=budget, spend_inr=held + failed, held_inr=held, failed_inr=failed,
        nodes=nodes, links=links,
        contractors=rows("contractor_id", "contractor_id", lambda c: names.get(c, c)),
        methods=rows("repair_method", "repair_method"),
    )


# ------------------------------------------------------------------ timeline
def ward_timeline(data: Data, wo: pd.DataFrame, ward_code: str | None) -> dict:
    """Daily rain + repairs that failed. ward_code=None means the whole city (code "ALL")."""
    rain = data.rain.copy()
    daily = rain.groupby("date").rainfall_mm.mean()
    alert = rain.assign(o=rain.imd_alert.map(ALERT_ORDER)).groupby("date").o.max()
    inv = {v: k for k, v in ALERT_ORDER.items()}
    w = wo[wo.reopened] if ward_code is None else wo[(wo.ward_code == ward_code) & wo.reopened]
    start = daily.index.min().normalize()
    end = max(daily.index.max(), w.completed_at.max()).normalize() if len(w) else daily.index.max()
    by_day = {d: g for d, g in w.groupby(w.completed_at.dt.normalize())}
    days = []
    for d in pd.date_range(start, end):
        g = by_day.get(d)
        has_rain = d in daily.index
        days.append(dict(
            date=d.strftime("%Y-%m-%d"),
            rain_mm=round(float(daily[d]), 1) if has_rain else None,
            alert=inv[int(alert[d])] if has_rain else None,
            failed=int(len(g)) if g is not None else 0,
            failed_wo_ids=sorted(g.work_order_id) if g is not None else [],
        ))
    return dict(
        ward_code=ward_code or "ALL", ward_slug=ward_slug(ward_code) if ward_code else "all",
        start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
        rain_data_ends=daily.index.max().strftime("%Y-%m-%d"),
        red_alert_days=int((alert == ALERT_ORDER["Red"]).sum()),
        failed_total=int(len(w)), days=days,
    )


# ------------------------------------------------------------------ contractor card
def _band_from_ci(c: dict) -> str:
    if c["reopen_ci_low_pct"] > c["reopen_expected_pct"]:
        return "worse"
    if c["reopen_ci_high_pct"] < c["reopen_expected_pct"]:
        return "better"
    return "as_expected"


def contractor_card(data: Data, wo: pd.DataFrame, flags: list[Flag], metrics: list[dict],
                    wards: list[dict], cid: str) -> dict | None:
    c = next((m for m in metrics if m["contractor_id"] == cid), None)
    if c is None:
        return None
    g = wo[wo.contractor_id == cid]
    n = int(len(g))
    total_n = int(len(wo))
    meta = data.contractors.set_index("contractor_id").loc[cid]
    over = sum(1 for f in flags if f.rule == "OVER_RATE" and f.contractor_id == cid)
    city_over_share = sum(1 for f in flags if f.rule == "OVER_RATE") / total_n
    over_share = over / n if n else 0.0

    # speed vs what its own deadlines predict: urgent jobs are rarely met by anyone (24 h targets ~3%, 168 h ~87%),
    # so each job is expected to be on time as often as the city manages for that deadline tier
    met = wo.sla_met.eq("Y")
    tier_rate = met.groupby(wo.sla_hours).mean()
    peer_sla = float(g.sla_hours.map(tier_rate).mean() * 100) if n else 0.0
    d_sla = c["sla_met_pct"] - peer_sla
    speed_band = "better" if d_sla >= 5 else "worse" if d_sla <= -5 else "as_expected"
    # durability vs expected (CI rule, same as wards)
    dur_band = _band_from_ci(c)
    # cost: share of jobs billed over 1.5x the rate card vs the city share
    cost_band = ("worse" if over >= 5 and over_share >= 2 * city_over_share
                 else "better" if over_share <= 0.5 * city_over_share else "as_expected")

    grades = [
        dict(key="speed", title="Did it finish on time?", band=speed_band,
             headline=f"{c['sla_met_pct']:.0f}% on time",
             detail=f"For the deadlines it was given, we would expect about {peer_sla:.0f}% on time.",
             value=c["sla_met_pct"], benchmark=peer_sla, scale_min=0, scale_max=100, unit="% on time"),
        dict(key="durability", title="Did the repairs last?", band=dur_band,
             headline=f"{c['reopen_observed_pct']:.0f}% reopened",
             detail=(f"Given the repair methods, rain and flooding in its wards we would expect about "
                     f"{c['reopen_expected_pct']:.0f}% to reopen."),
             value=c["reopen_observed_pct"], benchmark=c["reopen_expected_pct"], scale_min=0, scale_max=60,
             unit="% reopened within 30 days"),
        dict(key="cost", title="Did it bill fairly against the rate card?", band=cost_band,
             headline=f"{c['median_cost_vs_rate']:.2f}× the rate card",
             detail=(f"{over} of {n} jobs were billed over {OVER_RATE_X}× the rate card "
                     f"(across the city: {city_over_share * 100:.0f}% of jobs)."),
             value=c["median_cost_vs_rate"], benchmark=1.0, scale_min=0.5, scale_max=2.0, unit="× rate card (typical job)"),
        dict(key="volume", title="How much work did it do?", band="neutral",
             headline=f"{n} jobs", detail=f"{n / total_n * 100:.0f}% of all repair jobs, across {c['wards_active']} wards.",
             value=float(n), benchmark=None, scale_min=0, scale_max=float(max(m["work_orders"] for m in metrics)),
             unit="jobs"),
    ]

    # per-ward breakdown
    ward_meta = {w["ward_code"]: w for w in wards}
    wrows = []
    for code, gg in g.groupby("ward_code"):
        s = float(gg.cost_inr.sum())
        wrows.append(dict(
            ward_code=code, ward_slug=ward_slug(code), ward_name=ward_meta[code]["ward_name"],
            work_orders=int(len(gg)), spend_inr=s,
            failed_pct=float(gg.loc[gg.reopened, "cost_inr"].sum()) / s * 100 if s else 0.0,
            reopen_expected_pct=float(gg.expected_reopen_p.mean()) * 100,
            small_sample=bool(len(gg) < SMALL_SAMPLE)))
    wrows.sort(key=lambda r: -r["spend_inr"])

    # deterministic plain summary (no accusations); Claude can replace it later
    name = str(meta.contractor_name)
    sp = {"better": "more often than expected for its deadlines", "worse": "less often than expected for its deadlines",
          "as_expected": "about as often as expected for its deadlines"}[speed_band]
    du = {"better": "held up better than expected", "worse": "held up worse than expected",
          "as_expected": "held up about as expected"}[dur_band]
    parts = [
        f"{name} did {n} repair jobs across {c['wards_active']} wards, worth {money(c['spend_inr'])}.",
        f"Jobs were finished on time {sp} ({c['sla_met_pct']:.0f}% against about {peer_sla:.0f}% expected).",
        f"{c['reopen_observed_pct']:.0f}% of repairs were reopened within 30 days; for the methods, rain and "
        f"flooding involved we would expect about {c['reopen_expected_pct']:.0f}%, so repairs {du}.",
        f"The typical bill is {c['median_cost_vs_rate']:.2f}× the rate card, and {over} "
        f"{'job was' if over == 1 else 'jobs were'} billed over {OVER_RATE_X}× and worth checking.",
    ]
    focus = {"speed": "finishing closer to the target time", "durability": "making repairs last longer",
             "cost": "keeping bills closer to the rate card"}
    worst = [k for k, b in (("durability", dur_band), ("speed", speed_band), ("cost", cost_band)) if b == "worse"]
    parts.append(f"One thing to improve: {focus[worst[0]]}." if worst else "Nothing stands out as needing attention.")
    if n < SMALL_SAMPLE:
        parts.append(f"With only {n} jobs, treat this as a first look.")

    return dict(
        contractor_id=cid, contractor_name=name,
        empanelled_since=str(meta.empanelled_since) if pd.notna(meta.empanelled_since) else None,
        wards_active=c["wards_active"], work_orders=n, spend_inr=c["spend_inr"],
        failed_spend_inr=c["failed_spend_inr"], over_rate_count=over, small_sample=bool(n < SMALL_SAMPLE),
        summary=" ".join(parts), summary_source="template", grades=grades, wards=wrows,
    )
