"""Generate the test drawings and run extraction on every image (2 workers)."""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DRAW = ROOT / "data" / "drawings"


def work(png):
    from drawquote.extract import extract
    ex = extract(png)
    return png.name, {"fields": ex.fields, "conf": ex.conf, "cv": ex.cv}


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    if not (DRAW / "truth.json").exists():
        from drawquote.generate import make_drawings
        make_drawings(DRAW, n)
    pngs = sorted(DRAW.glob("*.png"))
    out = {}
    with Pool(2) as pool:
        for k, (name, res) in enumerate(pool.imap_unordered(work, pngs)):
            out[name] = res
            if k % 20 == 0:
                print(k, len(pngs), flush=True)
    (ROOT / "data" / "extractions.json").write_text(json.dumps(out, indent=1, default=float))
    print("done", len(out))
