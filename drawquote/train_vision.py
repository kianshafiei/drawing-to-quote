"""Train the part-family classifier on rendered drawings at all scan-quality levels."""
import random
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .extract import MODEL_PATH, view_features
from .generate import degrade, random_part, render


def dataset(n, seed):
    r = random.Random(seed)
    X, y, lv = [], [], []
    for i in range(n):
        p = random_part(r, i)
        img = render(p, r)
        for level in (0, 1, 2):
            X.append(view_features(np.array(degrade(img, level, r))))
            y.append(p.family)
            lv.append(level)
    return np.array(X), np.array(y), np.array(lv)


def main(n_train=600, n_test=200):
    Xtr, ytr, _ = dataset(n_train, seed=101)
    Xte, yte, lte = dataset(n_test, seed=202)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.05))
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    print("held-out accuracy", round(accuracy_score(yte, pred), 3))
    for level in (0, 1, 2):
        m = lte == level
        print(" level", level, round(accuracy_score(yte[m], pred[m]), 3))
    print(clf.classes_)
    print(confusion_matrix(yte, pred, labels=clf.classes_))
    joblib.dump(clf, MODEL_PATH)


if __name__ == "__main__":
    main(*map(int, sys.argv[1:]))
