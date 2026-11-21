"""Read one drawing and print a draft quote for a human to approve.

    python quote_drawing.py path/to/drawing.png
"""
import json
import sys
from pathlib import Path

from drawquote.extract import extract
from drawquote.generate import make_history
from drawquote.model import QuoteModel, SimilarJobs, part_fields
from drawquote.validate import check

# per-field review thresholds, calibrated by evaluate.py
TH = json.loads((Path(__file__).parent / "drawquote" / "thresholds.json").read_text())


def main(path):
    ex = extract(path)
    f = ex.fields
    print(f"\nDrawing: {path}")
    print(f"{'field':<10}{'value':<34}{'conf':>6}")
    for k, ck in [("title", "title"), ("part_no", "part_no"), ("material", "material"), ("finish", "finish"),
                  ("qty", "qty"), ("tol", "tol"), ("family", "family"), ("envelope", "envelope")]:
        c = ex.conf.get(ck, 1.0)
        mark = "  <- check" if (k in TH and (f.get(k) is None or c < TH[k])) else ""
        print(f"{k:<10}{str(f.get(k)):<34}{c:6.2f}{mark}")
    for k, txt in [("holes", f"{f['n_holes']} x Ø{f['hole_dia']:g}"), ("threads", f"{f['n_threads']} x M{f['thread']}")]:
        mark = "  <- check" if ex.conf[k] < TH.get(k, 0.8) else ""
        print(f"{k:<10}{txt:<34}{ex.conf[k]:6.2f}{mark}")

    issues = check(f)
    print("\nChecks:" if issues else "\nChecks: none raised")
    for i in issues:
        print(f"  [{i.severity}] {i.message}")

    hist = make_history(2000)
    hf, hp = [part_fields(vars(p)) for p in hist], [p.quote for p in hist]
    knn = SimilarJobs().fit(hf, hp, [p.id for p in hist])
    print("\nSimilar past jobs:")
    for pid, jf, price, _ in knn.query(f, 5):
        e = "x".join(f"{v:g}" for v in jf["envelope"])
        print(f"  {pid}  {jf['family']:<7}{jf['material']:<22}{e:<14} qty {jf['qty']:<4} £{price:,.2f}")

    est = QuoteModel().fit(hf, hp).predict(f)
    blocking = [i for i in issues if i.severity == "error"]
    print(f"\nDraft unit price: £{est:,.2f}" + ("  (NOT READY: resolve errors above)" if blocking else ""))


if __name__ == "__main__":
    main(sys.argv[1])
