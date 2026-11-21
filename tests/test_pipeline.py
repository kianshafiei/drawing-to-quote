import random

import pytest

from drawquote.generate import make_history, random_part, render
from drawquote.model import PerKgBaseline, QuoteModel, SimilarJobs, part_fields
from drawquote.validate import check, codes


def test_generator_is_deterministic():
    a = random_part(random.Random(5), 0)
    b = random_part(random.Random(5), 0)
    assert a == b


def test_history_and_test_set_do_not_overlap():
    hist = {p.id for p in make_history(200)}
    test = {random_part(random.Random(7), i).id for i in range(120)}
    assert not hist & test


def test_rules_catch_impossible_finish():
    f = {"material": "Mild Steel S275", "finish": "Anodise Black", "qty": 5, "family": "plate",
         "envelope": [100, 50, 10], "tol": 0.1}
    assert "finish_material" in codes(f)


def test_clean_part_has_no_issues():
    f = {"material": "Aluminium 6082-T6", "finish": "Anodise Black", "qty": 5, "family": "plate",
         "envelope": [100, 50, 10], "tol": 0.1, "n_threads": 4, "thread": 5}
    assert check(f) == []


def test_missing_fields_are_reported():
    assert {i.code for i in check({})} >= {"missing_material", "missing_envelope"}


def test_quote_model_beats_rule_of_thumb():
    hist = make_history(1200)
    f = [part_fields(vars(p)) for p in hist]
    y = [p.quote for p in hist]
    gbm, kg = QuoteModel().fit(f[:1000], y[:1000]), PerKgBaseline().fit(f[:1000], y[:1000])
    err = lambda m: sorted(abs(m.predict(x) - t) / t for x, t in zip(f[1000:], y[1000:]))[100]
    assert err(gbm) < err(kg)


def test_similar_jobs_prefers_same_family():
    hist = make_history(500)
    knn = SimilarJobs().fit([part_fields(vars(p)) for p in hist], [p.quote for p in hist], [p.id for p in hist])
    q = part_fields(vars(hist[0]))
    hits = knn.query(q, 5)
    assert hits[0][0] == hist[0].id
    assert sum(h[1]["family"] == q["family"] for h in hits) >= 4


@pytest.mark.slow
def test_reads_a_clean_drawing_or_flags_it(tmp_path):
    """Title-block fields must be read exactly. OCR of dimensions can vary with library
    versions, so the size must be either correct or sent to review, never silently wrong."""
    from drawquote.extract import extract
    r = random.Random(11)
    p = random_part(r, 0)
    render(p, r).save(tmp_path / "d.png")
    ex = extract(tmp_path / "d.png")
    assert ex.fields["material"] == p.material
    assert ex.fields["qty"] == p.qty
    truth = sorted([p.length, p.width, p.thickness], reverse=True)
    env = ex.fields["envelope"]
    assert env == truth or env is None or ex.conf["envelope"] < 0.5
