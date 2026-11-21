"""End-to-end evaluation: reading accuracy, review flags, design checks, retrieval, quoting.

Writes results/metrics.json and results/report.md.
"""
import json
from pathlib import Path

import numpy as np

from drawquote.generate import make_history
from drawquote.model import PerKgBaseline, QuoteModel, SimilarJobs, part_fields
from drawquote.validate import codes

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
TARGET_CATCH = 0.95   # each field's threshold is set so the calibration half catches 95% of its errors
CAL_SPLIT = 60        # parts D0000-D0059 calibrate thresholds; D0060-D0119 are the held-out test
FIELDS = ["family", "material", "finish", "qty", "tol", "holes", "threads", "pocket", "envelope"]
CONF_KEY = {"holes": "holes", "threads": "threads", "pocket": None}


def value(f, k):
    if k == "holes":
        return (f.get("n_holes") or 0, round(f.get("hole_dia") or 0, 1))
    if k == "threads":
        return (f.get("n_threads") or 0, f.get("thread") or 0)
    if k == "envelope":
        e = f.get("envelope")
        return tuple(round(v, 1) for v in e) if e else None
    return f.get(k)


THRESH = {}


def conf_of(ex, k):
    c = CONF_KEY.get(k, k)
    return 1.0 if c is None else ex["conf"].get(c, 1.0)


def flagged(ex, k):
    return value(ex["fields"], k) is None or conf_of(ex, k) < THRESH.get(k, 0.0)


MAX_FALSE_ALARM = 0.25  # never flag more than a quarter of correctly read values of a field


def calibrate(cal):
    """Per field, the lowest confidence threshold that catches TARGET_CATCH of the wrongly
    read values on the calibration half, without flagging more than MAX_FALSE_ALARM of the
    correct ones. If no threshold does both, take the one that catches most within the cap."""
    th = {}
    for k in FIELDS:
        pairs = [(conf_of(ex, k), value(ex["fields"], k) == value(tf, k)) for ex, tf in cal
                 if value(ex["fields"], k) is not None]
        wrong = [c for c, ok in pairs if not ok]
        right = [c for c, ok in pairs if ok]
        if not wrong:
            th[k] = 0.0
            continue
        best = 0.0
        for t in sorted({c + 1e-6 for c in wrong}):
            fa = np.mean([c < t for c in right]) if right else 0.0
            if fa > MAX_FALSE_ALARM:
                break
            best = t
            if np.mean([c < t for c in wrong]) >= TARGET_CATCH:
                break
        th[k] = float(best)
    return th


def reviewed(ex, truth):
    """A reviewer corrects every flagged field, nothing else."""
    f = dict(ex["fields"])
    for k in FIELDS:
        if flagged(ex, k):
            if k == "holes":
                f["n_holes"], f["hole_dia"] = truth["n_holes"], truth["hole_dia"]
            elif k == "threads":
                f["n_threads"], f["thread"] = truth["n_threads"], truth["thread"]
            else:
                f[k] = truth[k]
    return f


def ape(pred, true):
    return abs(pred - true) / true if pred == pred else float("nan")


def main():
    OUT.mkdir(exist_ok=True)
    truth = {p["id"]: p for p in json.loads((ROOT / "data/drawings/truth.json").read_text())}
    exs = json.loads((ROOT / "data/extractions.json").read_text())
    hist = make_history(2000)
    hf = [part_fields(vars(p)) for p in hist]
    hp = [p.quote for p in hist]
    gbm, perkg = QuoteModel().fit(hf, hp), PerKgBaseline().fit(hf, hp)
    knn = SimilarJobs().fit(hf, hp, [p.id for p in hist])

    cal = [(ex, part_fields(truth[n[:-7]])) for n, ex in exs.items() if int(n[1:5]) < CAL_SPLIT]
    THRESH.update(calibrate(cal))
    (ROOT / "drawquote" / "thresholds.json").write_text(json.dumps(THRESH, indent=1))
    print("review thresholds:", {k: round(v, 3) for k, v in THRESH.items()})
    rows = []
    for name, ex in sorted(exs.items()):
        if int(name[1:5]) < CAL_SPLIT:
            continue
        pid, lv = name[:-7], int(name[-5])
        t = truth[pid]
        tf = part_fields(t)
        r = {"id": pid, "level": lv, "lang": t["lang"], "family": t["family"]}
        for k in FIELDS:
            ok = value(ex["fields"], k) == value(tf, k)
            r[f"ok_{k}"], r[f"flag_{k}"] = ok, flagged(ex, k)
        rf = reviewed(ex, tf)
        r["ok_all_reviewed"] = all(value(rf, k) == value(tf, k) for k in FIELDS)
        price = t["quote"]
        r["ape_gbm_true"] = ape(gbm.predict(tf), price)
        r["ape_perkg_true"] = ape(perkg.predict(tf), price)
        r["ape_knn_true"] = ape(knn.predict(tf), price)
        r["ape_gbm_raw"] = ape(gbm.predict(ex["fields"]), price)
        r["ape_gbm_reviewed"] = ape(gbm.predict(rf), price)
        hits = knn.query(tf, 5)
        r["p5_true"] = np.mean([h[1]["family"] == tf["family"] and h[1]["material"] == tf["material"] for h in hits])
        hits = knn.query(rf, 5)
        r["p5_reviewed"] = np.mean([h[1]["family"] == tf["family"] and h[1]["material"] == tf["material"] for h in hits])
        ct, ce, cr = codes(tf), codes(ex["fields"]), codes(rf)
        r.update(rule_true=len(ct), rule_tp_raw=len(ct & ce), rule_fp_raw=len(ce - ct),
                 rule_tp_rev=len(ct & cr), rule_fp_rev=len(cr - ct))
        rows.append(r)

    M = {"n_parts_test": len({r["id"] for r in rows}), "n_images_test": len(rows), "catch_target": TARGET_CATCH,
         "thresholds": THRESH, "by_level": {}}
    for lv in (0, 1, 2, "all"):
        R = [r for r in rows if lv == "all" or r["level"] == lv]
        m = {"field_accuracy": {k: float(np.mean([r[f"ok_{k}"] for r in R])) for k in FIELDS}}
        cells = [(r[f"ok_{k}"], r[f"flag_{k}"]) for r in R for k in FIELDS]
        wrong = [fl for ok, fl in cells if not ok]
        m["fields_flagged_pct"] = float(np.mean([fl for _, fl in cells]))
        m["errors_caught_by_flags"] = float(np.mean(wrong)) if wrong else 1.0
        m["silent_error_rate"] = float(np.mean([(not ok) and not fl for ok, fl in cells]))
        m["drawings_fully_correct_raw"] = float(np.mean([all(r[f"ok_{k}"] for k in FIELDS) for r in R]))
        m["drawings_fully_correct_after_review"] = float(np.mean([r["ok_all_reviewed"] for r in R]))
        m["drawings_needing_review"] = float(np.mean([any(r[f"flag_{k}"] for k in FIELDS) for r in R]))
        for k in ("gbm_true", "perkg_true", "knn_true", "gbm_raw", "gbm_reviewed"):
            v = np.array([r[f"ape_{k}"] for r in R], float)
            m[f"median_ape_{k}"] = float(np.nanmedian(v))
            m[f"within20_{k}"] = float(np.nanmean(v <= 0.2))
            m[f"no_price_{k}"] = float(np.mean(np.isnan(v)))
        m["precision_at_5_true"] = float(np.mean([r["p5_true"] for r in R]))
        m["precision_at_5_reviewed"] = float(np.mean([r["p5_reviewed"] for r in R]))
        tot = sum(r["rule_true"] for r in R)
        for s in ("raw", "rev"):
            tp, fp = sum(r[f"rule_tp_{s}"] for r in R), sum(r[f"rule_fp_{s}"] for r in R)
            m[f"rules_recall_{s}"] = tp / tot if tot else 1.0
            m[f"rules_precision_{s}"] = tp / (tp + fp) if tp + fp else 1.0
        m["rule_issues_true"] = tot
        M["by_level"][str(lv)] = m
    (OUT / "metrics.json").write_text(json.dumps(M, indent=1))
    (OUT / "rows.json").write_text(json.dumps(rows, indent=0, default=float))
    print(json.dumps(M["by_level"]["all"], indent=1))
    return M


if __name__ == "__main__":
    main()
