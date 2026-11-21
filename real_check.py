"""Domain-shift check on six real CAD drawings (eDOCr test set, MIT licence).

The extractor was built for one synthetic layout. These drawings use other title
blocks, GD&T and materials outside the price list, so the question is not "is it
accurate" but "does it fail safely": wrong or unknown values should be flagged.
"""
import json
import sys
from pathlib import Path

import cv2

from drawquote.extract import extract

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "eDOCr/tests/test_samples")
TH = json.loads((Path(__file__).parent / "drawquote/thresholds.json").read_text())
# hand labels: material as written, stock envelope (mm), plain holes, threads
LABELS = {
    "Adapterplatte.jpg": dict(material="Stahl (steel, grade not given)", envelope=[200, 200, 20], holes=(4, 11.0), threads=(4, 8)),
    "BM_part.jpg": dict(material=None, envelope=[140, 100, 60], holes=(4, 15.0), threads=(0, 0)),
    "Candle_holder.jpg": dict(material="SS 1914 (steel)", envelope=[42, 42, 38], holes=(0, 0.0), threads=(0, 0)),
    "Gripper.jpg": dict(material="PLA (3D-print polymer)", envelope=[70.5, 70.5, 6.15], holes=(4, 6.6), threads=(0, 0)),
    "halter.jpg": dict(material=None, envelope=None, holes=(2, 8.8), threads=(0, 0)),
    "LIU0010.jpg": dict(material="EN AW-6082 (aluminium)", envelope=[25, 19.9, 19.9], holes=(0, 0.0), threads=(1, 8)),
}


def main():
    rows = []
    for name, lab in LABELS.items():
        im = cv2.imread(str(SRC / name), cv2.IMREAD_GRAYSCALE)
        s = 2400 / max(im.shape)
        tmp = Path("/tmp") / f"real_{name}.png"
        cv2.imwrite(str(tmp), cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else im)
        ex = extract(tmp)
        f, c = ex.fields, ex.conf
        got = {"material": f["material"], "envelope": f["envelope"],
               "holes": (f["n_holes"], f["hole_dia"]), "threads": (f["n_threads"], f["thread"])}
        res = {"drawing": name}
        for k in ("material", "envelope", "holes", "threads"):
            truth = lab[k]
            if k == "material":
                ok = False  # none of these materials is in the synthetic price list
            elif k == "envelope":
                ok = truth is not None and got[k] is not None and all(abs(a - b) < 0.5 for a, b in zip(got[k], truth))
            else:
                ok = tuple(got[k]) == tuple(truth)
            flag = got[k] is None or c.get(k, 1.0) < TH.get(k, 0.8)
            res[k] = {"truth": truth, "read": got[k], "correct": ok, "flagged": flag,
                      "outcome": "correct" if ok else ("flagged" if flag else "SILENT ERROR")}
        rows.append(res)
        print(name, {k: res[k]["outcome"] for k in ("material", "envelope", "holes", "threads")}, flush=True)
    Path("results").mkdir(exist_ok=True)
    Path("results/real_check.json").write_text(json.dumps(rows, indent=1, default=str))


if __name__ == "__main__":
    main()
