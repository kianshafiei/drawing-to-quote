"""Read a part drawing into structured fields, each with a confidence.

OCR (RapidOCR) reads the text; computer vision (OpenCV) classifies the part
family from the main-view outline and counts drawn holes, which is used to
cross-check the OCR'd hole callout.
"""
import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import cv2
import numpy as np

from .generate import MATERIALS, MATERIAL_DE, FINISHES, FINISH_DE

_ENGINE = None


def engine():
    global _ENGINE
    if _ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR
        _ENGINE = RapidOCR()
    return _ENGINE


@dataclass
class Token:
    text: str
    conf: float
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def cy(self):
        return (self.y0 + self.y1) / 2


DIM_RE = re.compile(r"^[Ø@0O]?\d{1,3}(?:[.,]\d{1,2})?$")


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Z0-9]", "", s.upper())


LABEL_KEYS = {
    "title": ["TITLE", "BENENNUNG"], "part_no": ["PARTNO", "ZEICHNUNGSNR"],
    "material": ["MATERIAL", "WERKSTOFF"], "qty": ["QTY", "STUCKZAHL"],
    "finish": ["FINISH", "OBERFLACHE"], "tol": ["GENERALTOLERANCE", "ALLGEMEINTOLERANZ"],
}
MAT_NAMES = {norm(k): k for k in MATERIALS} | {norm(v): k for k, v in MATERIAL_DE.items()}
FIN_NAMES = {norm(k): k for k in FINISHES} | {norm(v): k for k, v in FINISH_DE.items()}


def best_match(s: str, names: dict, cutoff=0.6):
    n = norm(s)
    scored = [(SequenceMatcher(None, n, k).ratio(), v) for k, v in names.items()]
    sc, v = max(scored)
    return (v, sc) if sc >= cutoff else (None, sc)


def deskew(gray):
    """Rotate a scanned sheet so its long straight lines (border, title block) are level."""
    edges = cv2.Canny(gray, 50, 150)
    h, w = gray.shape
    lines = cv2.HoughLinesP(edges, 1, np.pi / 1800, threshold=200, minLineLength=w // 4, maxLineGap=10)
    if lines is None:
        return gray, 0.0
    ang = [np.degrees(np.arctan2(y2 - y1, x2 - x1)) for x1, y1, x2, y2 in lines.reshape(-1, 4)]
    ang = [a for a in ang if abs(a) < 10]
    if not ang:
        return gray, 0.0
    a = float(np.median(ang))
    if abs(a) < 0.05:
        return gray, 0.0
    M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=255), a


def ocr(img, rotated_pass=True) -> list[Token]:
    """OCR the sheet, plus a second pass on the sheet turned 90 degrees so that
    vertical dimension text is read too (only dimension-like tokens are kept from it)."""
    res, _ = engine()(img)
    toks = []
    for box, text, conf in res or []:
        xs, ys = [p[0] for p in box], [p[1] for p in box]
        toks.append(Token(text, float(conf), min(xs), min(ys), max(xs), max(ys)))
    if rotated_pass:
        h = img.shape[0]
        rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        res, _ = engine()(rot)
        for box, text, conf in res or []:
            if not DIM_RE.match(text.replace(" ", "")):
                continue
            pts = [(yr, h - 1 - xr) for xr, yr in box]  # back to the original frame
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            t = Token(text, float(conf), min(xs), min(ys), max(xs), max(ys))
            if not any(_overlap(t, u) for u in toks):
                toks.append(t)
    return toks


def _overlap(a, b):
    return not (a.x1 < b.x0 or b.x1 < a.x0 or a.y1 < b.y0 or b.y1 < a.y0)


def parse_num(s: str):
    s = s.replace(",", ".")
    m = re.search(r"\d+(?:\.\d+)?", s)
    return float(m.group()) if m else None


def reread_cell(gray, lab: Token, right_edge: float, digits: bool = False):
    """Re-read an empty value cell next to a label. RapidOCR's detector skips lone
    short strings (a quantity of "5"), so: find the cell's left border and its
    top/bottom rules, crop the ink inside, and read it with Tesseract in single-line mode."""
    if gray is None:
        return None
    h = lab.y1 - lab.y0
    y0, y1 = int(max(0, lab.y0 - h * 0.7)), int(min(gray.shape[0], lab.y1 + h * 0.7))
    x0, x1 = int(lab.x1 + 5), int(min(gray.shape[1], right_edge))
    if x1 - x0 < 10 or y1 - y0 < 5:
        return None
    cell = cv2.GaussianBlur(gray[y0:y1, x0:x1], (3, 3), 0)
    bw = cv2.adaptiveThreshold(cell, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    ch, cw = bw.shape
    # long horizontal and vertical strokes are table rules
    hl = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((1, max(40, cw // 5)), np.uint8))
    vl = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((max(10, int(ch * 0.6)), 1), np.uint8))
    cols = np.nonzero(vl.sum(0) > 0)[0]
    left = int(cols[0]) + 4 if len(cols) and cols[0] < cw * 0.6 else 0
    ink = cv2.subtract(bw, cv2.dilate(hl | vl, np.ones((5, 5), np.uint8)))
    ink[:, :left] = 0
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab_im, stats, _ = cv2.connectedComponentsWithStats(ink)
    keep = [k for k in range(1, n) if stats[k, cv2.CC_STAT_HEIGHT] >= 8 and stats[k, cv2.CC_STAT_AREA] >= 20]
    if not keep:
        return None
    # glyphs of the value sit on one line: keep components near the median centre line
    cy = np.median([stats[k, 1] + stats[k, 3] / 2 for k in keep])
    keep = [k for k in keep if abs(stats[k, 1] + stats[k, 3] / 2 - cy) < h * 0.6]
    xa = min(stats[k, 0] for k in keep); xb = max(stats[k, 0] + stats[k, 2] for k in keep)
    ya = min(stats[k, 1] for k in keep); yb = max(stats[k, 1] + stats[k, 3] for k in keep)
    pad = 6
    crop = gray[y0 + max(0, ya - pad):y0 + yb + pad, x0 + max(0, xa - pad):x0 + xb + pad]
    crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    crop = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    crop = cv2.copyMakeBorder(crop, 20, 20, 30, 30, cv2.BORDER_CONSTANT, value=255)
    text, conf = tesseract_line(crop, "0123456789.," if digits else None)
    if not text:
        return None
    return Token(text, conf * 0.9, x0 + xa, y0 + ya, x0 + xb, y0 + yb)


def tesseract_line(img, whitelist=None):
    """Single-line recognition with Tesseract (psm 7); returns text and mean confidence 0-1."""
    import subprocess, tempfile, os
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "c.png")
        cv2.imwrite(f, img)
        cmd = ["tesseract", f, "stdout", "--psm", "7"]
        if whitelist:
            cmd += ["-c", f"tessedit_char_whitelist={whitelist}"]
        out = subprocess.run(cmd + ["tsv"], capture_output=True, text=True).stdout
    words = []
    for line in out.splitlines()[1:]:
        cols = line.split("\t")
        if len(cols) == 12 and cols[11].strip() and float(cols[10]) >= 0:
            words.append((cols[11].strip(), float(cols[10]) / 100))
    if not words:
        return "", 0.0
    return " ".join(w for w, _ in words), sum(c for _, c in words) / len(words)


def title_block(toks: list[Token], gray=None):
    """Find label tokens, then take the nearest token to the right on the same row."""
    out = {}
    for key, labels in LABEL_KEYS.items():
        cands = []
        for t in toks:
            n = norm(t.text)
            sc = max(SequenceMatcher(None, n, lab).ratio() for lab in labels)
            if sc > 0.75:
                cands.append((sc, t))
        if not cands:
            out[key] = (None, 0.0)
            continue
        _, lab = max(cands, key=lambda c: c[0])
        h = max(lab.y1 - lab.y0, 10)
        row = [t for t in toks if t is not lab and t.x0 > lab.x1 and abs(t.cy - lab.cy) < h * 0.9]
        if not row:
            v = reread_cell(gray, lab, right_edge=(gray.shape[1] - 40) if gray is not None else 0,
                            digits=key in ("qty", "tol"))
            if v is None:
                out[key] = (None, 0.0)
                continue
            out[key] = (v.text, v.conf * min(1.0, max(c[0] for c in cands)))
            continue
        v = min(row, key=lambda t: t.x0)
        out[key] = (v.text, v.conf * min(1.0, max(c[0] for c in cands)))
    block_top = min((t.y0 for t in toks if norm(t.text) in ("TITLE", "BENENNUNG")), default=1e9)
    block_left = min((t.x0 for t in toks if any(norm(t.text) == l for ls in LABEL_KEYS.values() for l in ls)), default=1e9)
    return out, block_top, block_left


CALLOUT_HOLE = re.compile(r"(\d+)X[Ø@0O]?(\d+(?:[.,]\d+)?)(THRU|DURCH)")
CALLOUT_THREAD = re.compile(r"(\d+)XM(\d+)")


def callouts(toks):
    holes = threads = None
    pocket = False
    hconf = tconf = 1.0
    used = set()
    for t in toks:
        s = unicodedata.normalize("NFKD", t.text).upper().replace(" ", "").replace("×", "X").replace("Ø", "0").replace(",", ".")
        m = CALLOUT_HOLE.search(s.replace("THRU", "THRU").replace("DURCHGEHEND", "DURCH"))
        if m:
            holes = (int(m.group(1)), float(m.group(2))); hconf = t.conf; used.add(id(t))
            continue
        m = CALLOUT_THREAD.search(s)
        if m:
            threads = (int(m.group(1)), int(m.group(2))); tconf = t.conf; used.add(id(t))
            continue
        if "POCKET" in s or "TASCHE" in s:
            pocket = True; used.add(id(t))
            continue
        # looks like a callout but did not parse: send it to review instead of reading "none"
        if holes is None and ("THRU" in s or "DURCH" in s):
            hconf = min(hconf, 0.3)
        if threads is None and re.search(r"\dXM|XM\d", s):
            tconf = min(tconf, 0.3)
    return holes, hconf, threads, tconf, pocket, used


def dimension_tokens(toks, block_top, block_left, used):
    dims = []
    for t in toks:
        if id(t) in used or (t.y0 > block_top - 20 and t.x0 > block_left - 20):
            continue
        s = t.text.replace(" ", "")
        if not DIM_RE.match(s):
            continue
        dia = s[0] in "Ø@O" or (s[0] == "0" and len(s) > 1 and s[1] not in ".,")
        v = parse_num(s[1:] if dia else s)
        if v is None or v <= 0:
            continue
        dims.append((v, dia, t.conf, t.x0, t))
    return dims


def recover_short_numbers(gray, toks, block_top, block_left):
    """Find digit-sized ink blobs that OCR skipped (lone "5" or "8" dimensions) and read
    them with Tesseract. Candidates must look like characters: right height for the
    drawing's text, taller than wide, not inside any token already read."""
    ref = [t.y1 - t.y0 for t in toks if DIM_RE.match(t.text.replace(" ", "")) and t.y1 - t.y0 < t.x1 - t.x0 + 5]
    if not ref:
        return []
    th = float(np.median(ref))
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, _, st, _ = cv2.connectedComponentsWithStats(bw)
    boxes = [(t.x0 - 4, t.y0 - 4, t.x1 + 4, t.y1 + 4) for t in toks]
    inside = lambda x, y: any(a <= x <= c and b <= y <= d for a, b, c, d in boxes)
    cand = []
    for k in range(1, n):
        x, y, w, h, area = st[k]
        if not (0.45 * th <= h <= 0.95 * th) or not (1.1 <= h / max(w, 1) <= 6) or area < 15:
            continue
        if (y > block_top - 20 and x > block_left - 20) or inside(x + w / 2, y + h / 2):
            continue
        cand.append((x, y, w, h))
    cand.sort()
    groups = []
    for c in cand:  # join neighbours on the same line into one string
        g = groups[-1] if groups else None
        if g and c[0] - (g[-1][0] + g[-1][2]) < 0.5 * th and abs(c[1] - g[-1][1]) < 0.3 * th:
            g.append(c)
        else:
            groups.append([c])
    out = []
    for g in groups[:40]:
        if len(g) > 3:
            continue
        xa, ya = min(c[0] for c in g), min(c[1] for c in g)
        xb, yb = max(c[0] + c[2] for c in g), max(c[1] + c[3] for c in g)
        # isolation: nothing else inked right next to the glyphs (rules out hatching, arrows)
        pad = max(4, int(0.15 * th))
        ring = bw[max(0, ya - pad):yb + pad, max(0, xa - pad):xb + pad].copy()
        ring[ya - max(0, ya - pad):ya - max(0, ya - pad) + (yb - ya), xa - max(0, xa - pad):xa - max(0, xa - pad) + (xb - xa)] = 0
        if ring.mean() > 255 * 0.08:
            continue
        crop = gray[max(0, ya - 6):yb + 6, max(0, xa - 6):xb + 6]
        crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        crop = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        crop = cv2.copyMakeBorder(crop, 20, 20, 30, 30, cv2.BORDER_CONSTANT, value=255)
        text, conf = tesseract_line(crop, "0123456789")
        if text.isdigit() and conf > 0.6:
            out.append(Token(text, conf * 0.85, xa, ya, xb, yb))
    return out


# ------------------------------------------------------------- computer vision
MODEL_PATH = __import__("pathlib").Path(__file__).resolve().parent / "family_clf.joblib"
_CLF = None


def view_features(gray: np.ndarray) -> np.ndarray:
    """HOG descriptor of the main-view region (above the title block, left of the callouts)."""
    from skimage.feature import hog
    h, w = gray.shape
    roi = gray[int(h * 0.06):int(h * 0.74), int(w * 0.03):int(w * 0.75)]
    _, bw = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw = cv2.dilate(bw, np.ones((5, 5), np.uint8))
    small = cv2.resize(bw, (192, 128), interpolation=cv2.INTER_AREA)
    return hog(small, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)


def cv_family(gray: np.ndarray):
    """Part family from the drawing's geometry, with its probability."""
    global _CLF
    if _CLF is None:
        import joblib
        _CLF = joblib.load(MODEL_PATH)
    proba = _CLF.predict_proba([view_features(gray)])[0]
    k = int(np.argmax(proba))
    return _CLF.classes_[k], float(proba[k])


def classify_family(cv_fam, dims):
    """Use the vision model, but tell plate from block by the OCR'd thickness ratio."""
    fam, p = cv_fam
    if fam in ("plate", "block"):
        plain = sorted([v for v, d, *_ in dims if not d], reverse=True)
        if len(plain) >= 3:
            by_ratio = "block" if plain[2] / max(1e-9, plain[1]) > 0.25 else "plate"
            if by_ratio != fam:          # the two cues disagree: decide, but send to review
                return (fam if p >= 0.85 else by_ratio), min(p, 0.5)
    return fam, p


def second_read(gray, t: Token) -> list[str]:
    """Read a dimension token again with Tesseract (digits only). Returns the digit
    strings read; vertical text is tried in both rotations."""
    pad = 10
    crop = gray[int(max(0, t.y0 - pad)):int(t.y1 + pad), int(max(0, t.x0 - pad)):int(t.x1 + pad)]
    if crop.size == 0:
        return []
    views = [crop]
    if crop.shape[0] > crop.shape[1] * 1.3:
        views = [cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE), cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)]
    out = []
    for v in views:
        v = cv2.resize(v, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        v = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        v = cv2.copyMakeBorder(v, 25, 25, 35, 35, cv2.BORDER_CONSTANT, value=255)
        text, _ = tesseract_line(v, "0123456789.,")
        out.append(re.sub(r"\D", "", text.split(".")[0].split(",")[0]))
    return out


def agrees(reads: list[str], value: float, diameter: bool) -> bool:
    want = str(int(value)) if value == int(value) else str(value).split(".")[0]
    for r in reads:
        # the diameter sign is not a digit; Tesseract renders it as an extra 0 or 8
        if r == want or (diameter and len(r) == len(want) + 1 and r.endswith(want)):
            return True
    return False


def _longest_run(line: np.ndarray, centre: int, max_gap: int = 4) -> int:
    """Length of the inked run through `centre` in a 1-D 0/1 array, bridging small gaps."""
    if not (0 <= centre < len(line)):
        return 0
    idx = np.nonzero(line)[0]
    if len(idx) == 0:
        return 0
    # nearest inked pixel to the centre, then grow both ways across gaps <= max_gap
    k = int(np.argmin(np.abs(idx - centre)))
    if abs(idx[k] - centre) > max_gap:
        return 0
    lo = hi = k
    while lo > 0 and idx[lo] - idx[lo - 1] <= max_gap:
        lo -= 1
    while hi < len(idx) - 1 and idx[hi + 1] - idx[hi] <= max_gap:
        hi += 1
    return int(idx[hi] - idx[lo])


def dimension_line_px(bw_h: np.ndarray, bw_v: np.ndarray, t: Token) -> tuple[int, int]:
    """Length of the longest straight run of ink beside the text, measured both
    horizontally (rows above/below) and vertically (columns left/right), because
    rotated dimension text is not always detected as rotated."""
    th = min(t.x1 - t.x0, t.y1 - t.y0)
    cx, cy = int((t.x0 + t.x1) / 2), int((t.y0 + t.y1) / 2)
    hb = vb = 0
    for y in range(int(max(0, t.y0 - 1.5 * th)), int(min(bw_h.shape[0], t.y1 + 1.5 * th))):
        if not (t.y0 <= y <= t.y1):
            hb = max(hb, _longest_run(bw_h[y], cx))
    for x in range(int(max(0, t.x0 - 1.5 * th)), int(min(bw_v.shape[1], t.x1 + 1.5 * th))):
        if not (t.x0 <= x <= t.x1):
            vb = max(vb, _longest_run(bw_v[:, x], cy))
    return hb, vb


def scale_consistent(gray, vals):
    """Drawings are to scale, so each dimension line's pixel length should equal its value
    times one common scale. Returns False if a value has no dimension line at all (likely a
    label, e.g. a bore note) or no line that fits the scale; True if all fit; None if it
    can't tell."""
    if len(vals) < 2:
        return None
    _, bw = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # thicken across the line direction so a slightly skewed line stays one run
    bw_h = cv2.dilate(bw, np.ones((3, 1), np.uint8))
    bw_v = cv2.dilate(bw, np.ones((1, 3), np.uint8))
    meas = [(v, dimension_line_px(bw_h, bw_v, tok)) for v, _, tok, _ in vals]
    if any(max(px) == 0 for _, px in meas):
        return False
    v0, px0 = max(meas, key=lambda m: m[0])
    sc = max(px0) / v0
    fits = lambda v, px: abs(px - v * sc) <= max(0.08 * v * sc, 4)
    return all(fits(v, px[0]) or fits(v, px[1]) for v, px in meas)


def envelope(fam, dims, gray=None):
    """Stock envelope (largest, middle, smallest) and a confidence.

    OCR scores turned out not to predict envelope errors, so confidence comes from
    agreement: each chosen dimension is read again by a second engine (Tesseract);
    all agree -> 0.95, any disagreement -> 0.4."""
    plain = sorted([(v, c, t, False) for v, d, c, _, t in dims if not d], key=lambda z: -z[0])
    dia = sorted([(v, c, t, True) for v, d, c, _, t in dims if d], key=lambda z: -z[0])
    try:
        if fam == "flange":
            od = dia[0]
            t = min(plain, key=lambda z: z[0]) if plain else None
            vals = [od, od, t]
        elif fam == "shaft":
            # only diameters left of the length dimension's text belong to the main body
            lx = max((x for v, d, c, x, _ in dims if not d and v == plain[0][0]), default=0)
            left = min(((x, v, c, t) for v, d, c, x, t in dims if d and x < lx), key=lambda z: z[0])
            ln, dd = plain[0], (left[1], left[2], left[3], True)
            vals = [ln, dd, dd]
        else:
            vals = plain[:3]
        if len(vals) < 3 or any(v is None for v in vals):
            return None, 0.0
        env = sorted((v[0] for v in vals), reverse=True)
        if gray is None:
            return env, min(v[1] for v in vals)
        uniq = list({id(z[2]): z for z in vals}.values())
        sc = scale_consistent(gray, uniq)
        if sc is False:                     # values disagree with the geometry on the sheet
            return env, 0.3
        agree = all(agrees(second_read(gray, tok), v, isd) for v, _, tok, isd in uniq)
        if sc and agree:
            return env, 0.97
        return env, 0.85 if (sc or agree) else 0.4
    except (IndexError, ValueError):
        return None, 0.0


@dataclass
class Extraction:
    fields: dict = field(default_factory=dict)   # name -> value
    conf: dict = field(default_factory=dict)     # name -> 0..1
    cv: dict = field(default_factory=dict)


def extract(path) -> Extraction:
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    gray, _ = deskew(gray)
    toks = ocr(gray)
    tb, block_top, block_left = title_block(toks, gray)
    ex = Extraction()
    title, tc = tb["title"]
    ex.fields["title"], ex.conf["title"] = title, tc
    ex.fields["part_no"], ex.conf["part_no"] = tb["part_no"]
    mv, mc = tb["material"]
    ex.fields["material_text"] = mv
    mat, msc = best_match(mv, MAT_NAMES) if mv else (None, 0)
    ex.fields["material"], ex.conf["material"] = mat, mc * msc if mat else 0.0
    fv, fc = tb["finish"]
    fin, fsc = best_match(fv, FIN_NAMES) if fv else (None, 0)
    ex.fields["finish"], ex.conf["finish"] = fin, fc * fsc if fin else 0.0
    qv, qc = tb["qty"]
    q = re.sub(r"\D", "", qv or "")
    ex.fields["qty"], ex.conf["qty"] = (int(q), qc) if q else (None, 0.0)
    tv, tcf = tb["tol"]
    tol = parse_num(tv or "")
    ex.fields["tol"], ex.conf["tol"] = (tol, tcf) if tol in (0.05, 0.1, 0.2) else (tol, tcf * 0.3)

    holes, hconf, threads, tconf, pocket, used = callouts(toks)
    ex.fields["n_holes"], ex.fields["hole_dia"] = holes if holes else (0, 0.0)
    ex.fields["n_threads"], ex.fields["thread"] = threads if threads else (0, 0)
    ex.conf["holes"], ex.conf["threads"] = hconf, tconf
    ex.fields["pocket"] = pocket
    # If the title block doesn't match the layout this reader knows, "no callout found"
    # is not evidence of "no holes": send absences to review instead of trusting them.
    ex.fields["layout_known"] = sum(v is not None for v, _ in tb.values()) >= 4
    if not ex.fields["layout_known"]:
        if not holes:
            ex.conf["holes"] = min(ex.conf["holes"], 0.2)
        if not threads:
            ex.conf["threads"] = min(ex.conf["threads"], 0.2)

    extra = recover_short_numbers(gray, toks, block_top, block_left)
    dims = dimension_tokens(toks + extra, block_top, block_left, used)
    cvf = cv_family(gray)
    ex.cv = {"family": cvf[0], "p": cvf[1]}
    fam, fconf = classify_family(cvf, dims)
    ex.fields["family"], ex.conf["family"] = fam, fconf
    env, ec = envelope(fam, dims, gray)
    ex.fields["envelope"], ex.conf["envelope"] = env, ec
    return ex
