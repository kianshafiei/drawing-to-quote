"""Engineering checks a quoting engineer would run before pricing a drawing.

Each rule returns an Issue. "error" means the drawing (or its extraction) is
contradictory and must go back to the customer or a reviewer; "warn" means it
can be priced but deserves a second look.
"""
from dataclasses import dataclass

REQUIRED = ["material", "qty", "family", "envelope", "tol"]
STEELS = {"Mild Steel S275", "Tool Steel O1"}
SOFT = {"Aluminium 6082-T6", "Brass CZ121", "Acetal POM"}  # need deeper thread engagement


@dataclass(frozen=True)
class Issue:
    code: str
    severity: str
    message: str


def check(f: dict) -> list[Issue]:
    out = []
    for k in REQUIRED:
        if f.get(k) in (None, "", []):
            out.append(Issue(f"missing_{k}", "error", f"{k} not found on the drawing"))
    mat, fin = f.get("material"), f.get("finish")
    if mat and fin:
        if fin == "Anodise Black" and mat != "Aluminium 6082-T6":
            out.append(Issue("finish_material", "error", f"anodising is only possible on aluminium, not {mat}"))
        if fin == "Passivate" and mat != "Stainless Steel 304":
            out.append(Issue("finish_material", "error", f"passivation applies to stainless steel, not {mat}"))
        if fin == "Zinc Plate" and mat not in STEELS:
            out.append(Issue("finish_material", "error", f"zinc plating is for carbon steels, not {mat}"))
        if fin == "Powder Coat" and mat == "Acetal POM":
            out.append(Issue("finish_material", "error", "POM cannot take a powder-coat cure temperature"))
    if mat == "Acetal POM" and f.get("tol") == 0.05:
        out.append(Issue("tol_material", "warn", "±0.05 on POM is hard to hold (thermal movement); confirm with customer"))
    env, thread, fam = f.get("envelope"), f.get("thread") or 0, f.get("family")
    if env and thread and f.get("n_threads"):
        if fam == "shaft":
            if env[1] < 1.2 * thread:
                out.append(Issue("thread_fit", "error", f"M{thread} does not fit a Ø{env[1]:g} shaft"))
        else:
            need = (1.5 if mat in SOFT else 1.0) * thread
            if env[2] < need:
                out.append(Issue("thread_engagement", "warn",
                                 f"M{thread} in {env[2]:g} mm wall gives under {need:g} mm engagement"))
    hd = f.get("hole_dia") or 0
    if env and hd and f.get("n_holes") and fam in ("plate", "block") and hd > 0.5 * env[1]:
        out.append(Issue("hole_size", "warn", f"Ø{hd:g} holes are large for a {env[1]:g} mm wide part"))
    return out


def codes(f: dict) -> set[str]:
    """Design-rule issue codes only (ignores missing-field flags)."""
    return {i.code for i in check(f) if not i.code.startswith("missing_")}
