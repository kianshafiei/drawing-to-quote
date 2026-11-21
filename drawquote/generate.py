"""Synthetic mechanical part drawings with exact ground truth.

Each drawing has a title block (English or German labels), a main view with
overall dimensions, hole and thread callouts, and general notes. The same
random part spec is also priced by a simulated shop cost model, which stands
in for a company's quote history.
"""
import json
import math
import random
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
W, H = 2400, 1700

MATERIALS = {  # name: (density kg/dm3, price GBP/kg, machinability factor)
    "Aluminium 6082-T6": (2.70, 4.5, 0.8),
    "Mild Steel S275": (7.85, 1.6, 1.0),
    "Stainless Steel 304": (8.00, 5.5, 1.6),
    "Brass CZ121": (8.50, 7.0, 0.7),
    "Tool Steel O1": (7.85, 6.0, 1.8),
    "Acetal POM": (1.41, 6.5, 0.6),
}
MATERIAL_DE = {"Aluminium 6082-T6": "Aluminium 6082-T6", "Mild Steel S275": "Stahl S275",
               "Stainless Steel 304": "Edelstahl 1.4301", "Brass CZ121": "Messing CuZn39Pb3",
               "Tool Steel O1": "Werkzeugstahl 1.2510", "Acetal POM": "POM"}
FINISHES = ["None", "Anodise Black", "Zinc Plate", "Passivate", "Powder Coat"]
FINISH_DE = {"None": "Keine", "Anodise Black": "Eloxal schwarz", "Zinc Plate": "Verzinkt",
             "Passivate": "Passiviert", "Powder Coat": "Pulverbeschichtet"}
TOLS = [0.05, 0.1, 0.2]
FAMILIES = ["plate", "flange", "shaft", "block"]
LABELS = {
    "en": dict(title="TITLE", part="PART NO", material="MATERIAL", qty="QTY", finish="FINISH",
               tol="GENERAL TOLERANCE", scale="SCALE", drawn="DRAWN", thru="THRU", notes="NOTES",
               edges="BREAK ALL SHARP EDGES", units="ALL DIMENSIONS IN MM"),
    "de": dict(title="BENENNUNG", part="ZEICHNUNGSNR", material="WERKSTOFF", qty="STÜCKZAHL",
               finish="OBERFLÄCHE", tol="ALLGEMEINTOLERANZ", scale="MASSSTAB", drawn="GEZEICHNET",
               thru="DURCHGEHEND", notes="HINWEISE", edges="KANTEN GEBROCHEN", units="MASSE IN MM"),
}
TITLES = {"plate": ["Mounting Plate", "Adapter Plate", "Base Plate", "Cover Plate"],
          "flange": ["Bearing Flange", "Pipe Flange", "Motor Flange"],
          "shaft": ["Drive Shaft", "Stepped Pin", "Spindle"],
          "block": ["Clamp Block", "Spacer Block", "Guide Block"]}
TITLES_DE = {"Mounting Plate": "Montageplatte", "Adapter Plate": "Adapterplatte", "Base Plate": "Grundplatte",
             "Cover Plate": "Deckplatte", "Bearing Flange": "Lagerflansch", "Pipe Flange": "Rohrflansch",
             "Motor Flange": "Motorflansch", "Drive Shaft": "Antriebswelle", "Stepped Pin": "Stufenbolzen",
             "Spindle": "Spindel", "Clamp Block": "Klemmblock", "Spacer Block": "Distanzblock",
             "Guide Block": "Fuehrungsblock"}


@dataclass
class Part:
    id: str
    family: str
    lang: str
    title: str
    part_no: str
    material: str
    qty: int
    finish: str
    tol: float
    length: float      # longest overall size (plate/block length, flange OD, shaft length)
    width: float       # plate/block width, flange OD, shaft max diameter
    thickness: float   # plate/flange/block thickness, shaft max diameter
    n_holes: int = 0
    hole_dia: float = 0.0
    n_threads: int = 0
    thread: int = 0    # metric size, e.g. 8 for M8
    pocket: bool = False
    quote: float = 0.0  # simulated unit price, GBP
    meta: dict = field(default_factory=dict)


def rnd(r, lo, hi, step=5):
    return float(r.randrange(int(lo / step), int(hi / step) + 1) * step)


def random_part(r: random.Random, i: int) -> Part:
    fam = r.choice(FAMILIES)
    lang = "de" if r.random() < 0.3 else "en"
    mat = r.choice(list(MATERIALS))
    title = r.choice(TITLES[fam])
    p = Part(id=f"D{i:04d}", family=fam, lang=lang, title=title,
             part_no=f"{r.choice(['MA','KS','TT','QP'])}-{r.randint(1000, 9999)}-{r.choice('ABC')}",
             material=mat, qty=r.choice([1, 2, 5, 10, 20, 50, 100, 250]),
             finish=r.choice(FINISHES), tol=r.choice(TOLS), length=0, width=0, thickness=0)
    if fam == "plate":
        p.length, p.width = rnd(r, 60, 400), rnd(r, 40, 300)
        if p.width > p.length:
            p.length, p.width = p.width, p.length
        p.thickness = r.choice([3, 5, 6, 8, 10, 12, 15, 20, 25])
        p.n_holes = r.choice([0, 2, 4, 4, 6, 8])
        p.hole_dia = r.choice([5.5, 6.6, 9.0, 11.0, 13.5]) if p.n_holes else 0.0
        p.n_threads = r.choice([0, 0, 2, 4])
    elif fam == "flange":
        od = rnd(r, 60, 300)
        p.length = p.width = od
        p.thickness = r.choice([8, 10, 12, 15, 20, 25, 30])
        p.n_holes = r.choice([3, 4, 6, 8])
        p.hole_dia = r.choice([6.6, 9.0, 11.0, 13.5, 17.5])
        p.n_threads = r.choice([0, 0, 3, 4])
        p.meta["bore"] = rnd(r, 20, od * 0.5)
    elif fam == "shaft":
        p.length = rnd(r, 50, 500)
        d = rnd(r, 10, 80, 2)
        p.width = p.thickness = d
        p.n_threads = r.choice([0, 1, 2])
        p.meta["step_d"] = max(6.0, d - rnd(r, 4, max(4, d * 0.4), 2))
        p.meta["step_l"] = rnd(r, 10, p.length * 0.4)
    else:  # block
        p.length, p.width, p.thickness = rnd(r, 30, 250), rnd(r, 20, 150), rnd(r, 15, 100)
        p.n_holes = r.choice([0, 2, 4])
        p.hole_dia = r.choice([6.6, 9.0, 11.0]) if p.n_holes else 0.0
        p.n_threads = r.choice([0, 2, 4])
        p.pocket = r.random() < 0.5
    if p.n_threads:
        p.thread = r.choice([4, 5, 6, 8, 10, 12])
    if p.lang == "de":
        p.title = TITLES_DE[p.title]
    p.quote = shop_quote(p, r)
    return p


def stock_mass_kg(p: Part) -> float:
    if p.family in ("flange", "shaft"):
        d = p.width if p.family == "shaft" else p.length
        vol = math.pi * (d / 2 + 2) ** 2 * (p.length if p.family == "shaft" else p.thickness + 3)
    else:
        vol = (p.length + 5) * (p.width + 5) * (p.thickness + 3)
    return vol / 1e6 * MATERIALS[p.material][0]


def shop_quote(p: Part, r: random.Random | None = None) -> float:
    """Simulated unit price: material + machining + finish + setup spread over qty."""
    dens, price_kg, mach = MATERIALS[p.material]
    mass = stock_mass_kg(p)
    base_min = {"plate": 12, "flange": 25, "shaft": 20, "block": 18}[p.family]
    minutes = base_min + 0.04 * (p.length + p.width) + 1.5 * p.n_holes + 2.5 * p.n_threads + 20 * p.pocket
    minutes *= mach * {0.05: 1.4, 0.1: 1.15, 0.2: 1.0}[p.tol]
    area_dm2 = 2 * (p.length * p.width + (p.length + p.width) * p.thickness) / 1e4
    finish = {"None": 0, "Anodise Black": 0.9, "Zinc Plate": 0.5, "Passivate": 0.4, "Powder Coat": 0.7}[p.finish] * area_dm2 + (8 if p.finish != "None" else 0) / max(1, p.qty) * 5
    setup = 90 / p.qty
    unit = (mass * price_kg * 1.1 + minutes / 60 * 55 + finish + setup) * 1.25
    noise = math.exp(r.gauss(0, 0.08)) if r else 1.0
    return round(unit * noise, 2)


# ---------------------------------------------------------------- drawing
def fmt(v: float, lang: str) -> str:
    s = f"{v:.2f}" if v != int(v) else f"{v:.0f}"
    return s.replace(".", ",") if lang == "de" else s


def font(size):
    return ImageFont.truetype(FONT, size)


def dim_h(d, x1, x2, y, text, f, ext_from=None):
    d.line([(x1, y), (x2, y)], fill=0, width=2)
    for x in (x1, x2):
        d.line([(x, y - 12), (x, y + 12)], fill=0, width=2)
        if ext_from is not None:
            d.line([(x, ext_from), (x, y)], fill=0, width=1)
    tw = d.textlength(text, font=f)
    d.rectangle([(x1 + x2) / 2 - tw / 2 - 4, y - f.size - 10, (x1 + x2) / 2 + tw / 2 + 4, y - 4], fill=255)
    d.text(((x1 + x2) / 2 - tw / 2, y - f.size - 8), text, font=f, fill=0)


def dim_v(img, x, y1, y2, text, f, ext_from=None):
    d = ImageDraw.Draw(img)
    d.line([(x, y1), (x, y2)], fill=0, width=2)
    for y in (y1, y2):
        d.line([(x - 12, y), (x + 12, y)], fill=0, width=2)
        if ext_from is not None:
            d.line([(ext_from, y), (x, y)], fill=0, width=1)
    tw = int(ImageDraw.Draw(img).textlength(text, font=f)) + 8
    t = Image.new("L", (tw, f.size + 10), 255)
    ImageDraw.Draw(t).text((4, 2), text, font=f, fill=0)
    t = t.rotate(90, expand=True)
    img.paste(t, (int(x - t.width - 6), int((y1 + y2) / 2 - t.height / 2)))


def render(p: Part, r: random.Random) -> Image.Image:
    L = LABELS[p.lang]
    img = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(img)
    d.rectangle([30, 30, W - 30, H - 30], outline=0, width=4)
    fs = r.choice([30, 32, 34, 36])
    f, fsm, fbig = font(fs), font(int(fs * 0.75)), font(int(fs * 1.4))

    # main view area: left 2/3 of sheet
    ax0, ay0, ax1, ay1 = 250, 220, 1500, 1150
    sx = min((ax1 - ax0) / max(p.length, 1), (ay1 - ay0) / max(p.width, 1)) * r.uniform(0.6, 0.8)
    cx, cy = (ax0 + ax1) / 2, (ay0 + ay1) / 2
    if p.family in ("plate", "block"):
        w2, h2 = p.length * sx / 2, p.width * sx / 2
        x1, y1, x2, y2 = cx - w2, cy - h2, cx + w2, cy + h2
        d.rectangle([x1, y1, x2, y2], outline=0, width=4)
        hole_pts = []
        if p.n_holes:
            inset = min(p.length, p.width) * 0.15 * sx
            cols = 2 if p.n_holes <= 4 else p.n_holes // 2
            for k in range(p.n_holes):
                row, col = divmod(k, cols) if p.n_holes > 2 else (0, k)
                hx = x1 + inset + col * ((x2 - x1 - 2 * inset) / max(1, cols - 1))
                hy = (y1 + inset) if row == 0 else (y2 - inset)
                if p.n_holes == 2:
                    hy = cy
                hole_pts.append((hx, hy))
                rr = max(6, p.hole_dia * sx / 2)
                d.ellipse([hx - rr, hy - rr, hx + rr, hy + rr], outline=0, width=3)
        if p.pocket:
            d.rectangle([cx - w2 * 0.4, cy - h2 * 0.35, cx + w2 * 0.4, cy + h2 * 0.35], outline=0, width=2)
        dim_h(d, x1, x2, y2 + 110, fmt(p.length, p.lang), f, ext_from=y2 + 8)
        dim_v(img, x1 - 110, y1, y2, fmt(p.width, p.lang), f, ext_from=x1 - 8)
        # side view for thickness
        tx = min(x2 + 220, 1650)
        tw = max(10, p.thickness * sx)
        d.rectangle([tx, y1, tx + tw, y2], outline=0, width=4)
        dim_h(d, tx, tx + tw, y1 - 60, fmt(p.thickness, p.lang), f, ext_from=y1 - 8)
        callout_xy = hole_pts[0] if hole_pts else (x1 + 30, y1 + 30)
        side_right = tx + tw
    elif p.family == "flange":
        rr = p.length * sx / 2
        d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=0, width=4)
        rb = p.meta["bore"] * sx / 2
        d.ellipse([cx - rb, cy - rb, cx + rb, cy + rb], outline=0, width=3)
        pcd = (rr + rb) / 2
        hole_pts = []
        for k in range(p.n_holes):
            a = 2 * math.pi * k / p.n_holes + 0.3
            hx, hy = cx + pcd * math.cos(a), cy + pcd * math.sin(a)
            hole_pts.append((hx, hy))
            h = max(6, p.hole_dia * sx / 2)
            d.ellipse([hx - h, hy - h, hx + h, hy + h], outline=0, width=3)
        dim_h(d, cx - rr, cx + rr, cy + rr + 110, "Ø" + fmt(p.length, p.lang), f, ext_from=cy + 8)
        tx = min(cx + rr + 220, 1650)
        tw = max(10, p.thickness * sx)
        d.rectangle([tx, cy - rr, tx + tw, cy + rr], outline=0, width=4)
        dim_h(d, tx, tx + tw, cy - rr - 60, fmt(p.thickness, p.lang), f, ext_from=cy - rr - 8)
        d.text((cx - rb + 5, cy - fs / 2), "Ø" + fmt(p.meta["bore"], p.lang), font=fsm, fill=0)
        callout_xy = hole_pts[0]
        side_right = tx + tw
    else:  # shaft, side view
        hl, hr = p.length * sx / 2, p.width * sx / 2
        sl = p.meta["step_l"] * sx
        hr2 = p.meta["step_d"] * sx / 2
        x1, x2 = cx - hl, cx + hl
        d.rectangle([x1, cy - hr, x2 - sl, cy + hr], outline=0, width=4)
        d.rectangle([x2 - sl, cy - hr2, x2, cy + hr2], outline=0, width=4)
        d.line([(x1 - 30, cy), (x2 + 30, cy)], fill=0, width=1)
        dim_h(d, x1, x2, cy + hr + 120, fmt(p.length, p.lang), f, ext_from=cy + 8)
        dim_v(img, x1 - 90, cy - hr, cy + hr, "Ø" + fmt(p.width, p.lang), f)
        dim_v(img, x2 + 150, cy - hr2, cy + hr2, "Ø" + fmt(p.meta["step_d"], p.lang), fsm)
        callout_xy = (x2 - sl / 2, cy)
        side_right = x2 + 200

    # callouts with leader lines
    lines = []
    if p.n_holes:
        lines.append(f"{p.n_holes} x Ø{fmt(p.hole_dia, p.lang)} {L['thru']}")
    if p.n_threads:
        lines.append(f"{p.n_threads} x M{p.thread} {L['thru']}" if p.family != "shaft" else f"{p.n_threads} x M{p.thread} x 15")
    if p.pocket:
        lines.append("POCKET 10 DEEP" if p.lang == "en" else "TASCHE 10 TIEF")
    ly = 140
    for t in lines:
        # keep callout text clear of the side view and its thickness dimension
        lx = min(max(1500, side_right + 90) + r.randint(0, 60), 1820)
        d.line([callout_xy, (lx - 10, ly + fs / 2)], fill=0, width=2)
        d.text((lx, ly), t, font=f, fill=0)
        ly += int(fs * 1.6)

    # notes
    d.text((80, 1260), f"{L['notes']}:", font=fsm, fill=0)
    d.text((80, 1300), f"1. {L['units']}", font=fsm, fill=0)
    d.text((80, 1340), f"2. {L['edges']}", font=fsm, fill=0)

    # title block, bottom right
    bx0, by0, bx1, by1 = 1500, 1250, W - 30, H - 30
    d.rectangle([bx0, by0, bx1, by1], outline=0, width=3)
    rows = [(L["title"], p.title), (L["part"], p.part_no),
            (L["material"], p.material if p.lang == "en" else MATERIAL_DE[p.material]),
            (L["qty"], str(p.qty)), (L["finish"], p.finish if p.lang == "en" else FINISH_DE[p.finish]),
            (L["tol"], "±" + fmt(p.tol, p.lang)), (L["scale"], r.choice(["1:1", "1:2", "2:1"])),
            (L["drawn"], r.choice(["J.Smith", "A.Patel", "M.Weber", "L.Rossi"]))]
    rh = (by1 - by0) / len(rows)
    for k, (lab, val) in enumerate(rows):
        yy = by0 + k * rh
        if k:
            d.line([(bx0, yy), (bx1, yy)], fill=0, width=1)
        d.text((bx0 + 12, yy + rh / 2 - fsm.size / 2), lab, font=fsm, fill=0)
        d.text((bx0 + 330, yy + rh / 2 - fsm.size / 2), val, font=fsm if k else fbig if len(val) < 18 else f, fill=0)
    d.line([(bx0 + 315, by0), (bx0 + 315, by1)], fill=0, width=1)
    return img


def degrade(img: Image.Image, level: int, r: random.Random) -> Image.Image:
    """0 = clean export, 1 = good scan, 2 = poor scan/photo."""
    if level == 0:
        return img
    a = np.array(img).astype(np.float32)
    ang = r.uniform(-0.6, 0.6) if level == 1 else r.uniform(-1.8, 1.8)
    im = Image.fromarray(a.astype(np.uint8)).rotate(ang, fillcolor=255, resample=Image.BICUBIC)
    im = im.filter(ImageFilter.GaussianBlur(0.6 if level == 1 else 1.3))
    a = np.array(im).astype(np.float32)
    rng = np.random.default_rng(r.randint(0, 10**6))
    a += rng.normal(0, 6 if level == 1 else 18, a.shape)
    if level == 2:
        a = a * rng.uniform(0.85, 1.0) + 20  # washed-out contrast
        sp = rng.random(a.shape)
        a[sp < 0.002] = 0
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if level == 2:
        im = im.resize((int(W * 0.7), int(H * 0.7)), Image.BILINEAR)
        import io
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=40)
        im = Image.open(buf).convert("L")
    return im


def make_history(n: int, seed: int = 1):
    """Past jobs with structured data and prices: what a shop's quote history holds."""
    r = random.Random(seed)
    return [random_part(r, 100000 + i) for i in range(n)]


def make_drawings(out: Path, n: int, seed: int = 7, levels=(0, 1, 2)):
    out.mkdir(parents=True, exist_ok=True)
    r = random.Random(seed)
    parts = []
    for i in range(n):
        p = random_part(r, i)
        base = render(p, r)
        for lv in levels:
            degrade(base, lv, r).save(out / f"{p.id}_n{lv}.png", optimize=True)
        parts.append(asdict(p))
    (out / "truth.json").write_text(json.dumps(parts, indent=1))
    return parts


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    make_drawings(Path(__file__).resolve().parent.parent / "data" / "drawings", n)
    print("done", n)
