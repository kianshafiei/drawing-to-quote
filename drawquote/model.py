"""Similar-job retrieval and quote estimation from structured drawing fields."""
import math

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .generate import FAMILIES, FINISHES, MATERIALS

MATS = list(MATERIALS)
NUM = ["log_L", "log_W", "log_T", "log_mass", "n_holes", "n_threads", "thread", "pocket", "tol", "log_qty"]


def part_fields(p: dict) -> dict:
    """Ground-truth part (generator dict) -> the same field layout extraction produces."""
    return {"family": p["family"], "material": p["material"], "finish": p["finish"], "qty": p["qty"],
            "tol": p["tol"], "n_holes": p["n_holes"], "hole_dia": p["hole_dia"], "n_threads": p["n_threads"],
            "thread": p["thread"], "pocket": p["pocket"],
            "envelope": sorted([p["length"], p["width"], p["thickness"]], reverse=True)}


def stock_mass(f: dict) -> float:
    """Billet mass from the envelope; NaN when material or size is unknown."""
    env, mat = f.get("envelope"), f.get("material")
    if not env or mat not in MATERIALS:
        return float("nan")
    a, b, c = env
    if f.get("family") == "shaft":
        vol = math.pi * (b / 2 + 2) ** 2 * a
    elif f.get("family") == "flange":
        vol = math.pi * (a / 2 + 2) ** 2 * (c + 3)
    else:
        vol = (a + 5) * (b + 5) * (c + 3)
    return vol / 1e6 * MATERIALS[mat][0]


def featurize(f: dict) -> np.ndarray:
    """Numeric vector; unknown values become NaN (the GBM handles them natively)."""
    nan = float("nan")
    env = f.get("envelope") or [nan] * 3
    m = stock_mass(f)
    num = [math.log(env[0]) if env[0] == env[0] else nan, math.log(env[1]) if env[1] == env[1] else nan,
           math.log(env[2]) if env[2] == env[2] else nan, math.log(m) if m == m else nan,
           f.get("n_holes") or 0, f.get("n_threads") or 0, f.get("thread") or 0, float(bool(f.get("pocket"))),
           f["tol"] if f.get("tol") in (0.05, 0.1, 0.2) else nan,
           math.log(f["qty"]) if f.get("qty") else nan]
    oh = lambda v, cats: [1.0 if v == c else (nan if v is None else 0.0) for c in cats]
    return np.array(num + oh(f.get("family"), FAMILIES) + oh(f.get("material"), MATS) + oh(f.get("finish"), FINISHES))


class QuoteModel:
    """Gradient-boosted model on log unit price."""

    def fit(self, fields, prices):
        import lightgbm as lgb
        X = np.vstack([featurize(f) for f in fields])
        self.m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=10,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
        self.m.fit(X, np.log(prices))
        return self

    def predict(self, f) -> float:
        return float(np.exp(self.m.predict(featurize(f)[None])[0]))


class PerKgBaseline:
    """Rule of thumb: median price per kg of billet, per material and quantity band."""

    @staticmethod
    def band(q):
        return 0 if q is None else (0 if q < 5 else 1 if q < 50 else 2)

    def fit(self, fields, prices):
        self.rate = {}
        for f, p in zip(fields, prices):
            self.rate.setdefault((f["material"], self.band(f["qty"])), []).append(p / stock_mass(f))
        self.rate = {k: float(np.median(v)) for k, v in self.rate.items()}
        self.fallback = float(np.median(list(self.rate.values())))
        return self

    def predict(self, f) -> float:
        m = stock_mass(f)
        r = self.rate.get((f.get("material"), self.band(f.get("qty"))), self.fallback)
        return r * m if m == m else float("nan")


class SimilarJobs:
    """Nearest past jobs on scaled features; missing values are filled with history means."""

    def fit(self, fields, prices, ids):
        X = np.vstack([featurize(f) for f in fields])
        self.mean = np.nanmean(X, axis=0)
        self.sc = StandardScaler().fit(X)
        w = np.ones(X.shape[1])
        w[len(NUM):len(NUM) + len(FAMILIES)] = 2.0           # same kind of part matters most
        w[len(NUM) + len(FAMILIES):len(NUM) + len(FAMILIES) + len(MATS)] = 1.5
        self.w = w
        self.nn = NearestNeighbors(n_neighbors=10).fit(self.sc.transform(X) * w)
        self.fields, self.prices, self.ids = fields, np.asarray(prices), ids
        return self

    def query(self, f, k=5):
        x = featurize(f)
        x = np.where(np.isnan(x), self.mean, x)
        d, idx = self.nn.kneighbors(self.sc.transform(x[None]) * self.w, n_neighbors=k)
        return [(self.ids[i], self.fields[i], float(self.prices[i]), float(dd)) for i, dd in zip(idx[0], d[0])]

    def predict(self, f, k=5) -> float:
        hits = self.query(f, k)
        # geometric mean of the neighbours' prices
        return float(np.exp(np.mean([math.log(p) for _, _, p, _ in hits])))
