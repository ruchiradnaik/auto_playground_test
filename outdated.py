#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sentiment Classifier using Naive Bayes + TF-IDF
Written in old-style Python 2/early Python 3 patterns (~2013-2015 era)

Deprecated patterns used:
  - Python 2-style print statements (wrapped in print())
  - Old sklearn API: fit_transform on test set, LabelEncoder misuse
  - nose-style test assertions
  - Old numpy matrix (np.matrix) instead of np.ndarray
  - Deprecated: sklearn.cross_validation instead of sklearn.model_selection
  - Deprecated: .todense() without .A (np.matrix pitfall)
  - Old-style string formatting (% operator)
  - Mutable default arguments (Python anti-pattern)
  - Using has_key() dict method (Python 2 only)
  - Old urllib2 / cPickle imports
  - Deprecated: sklearn.grid_search.GridSearchCV
  - Deprecated: sklearn.lda.LDA
  - sys.maxint instead of sys.maxsize
  - Old-style class (no inheritance from object in Py2 style)
  - pickle protocol=2 (Py2 compat)
  - Old confusion matrix plotting (no sklearn.metrics.ConfusionMatrixDisplay)
"""

from __future__ import print_function   # Py2 compat shim (unnecessary in Py3)
from __future__ import division         # Py2 compat shim

import sys
import os
import pickle
import time

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

# ─── DEPRECATED IMPORTS ────────────────────────────────────────────────────
# These modules no longer exist in modern sklearn
try:
    from sklearn.cross_validation import train_test_split, StratifiedKFold   # removed in 0.20
    from sklearn.grid_search import GridSearchCV                               # removed in 0.20
    from sklearn.lda import LDA                                                # removed in 0.17 → LinearDiscriminantAnalysis
except ImportError:
    # Fallback so the rest of the script can still demonstrate the old patterns
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.model_selection import GridSearchCV
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline


# ─── GLOBAL CONFIG (old-style mutable dict default — anti-pattern) ──────────
CONFIG = {
    "max_features": 5000,
    "ngram_range": (1, 2),
    "n_folds": 5,
    "random_state": 0,
}


# ─── OLD-STYLE CLASS (no explicit object inheritance — Python 2 style) ──────
class SentimentDataset:
    """Loads and holds text data. Old-style class definition."""

    # Python 2 anti-pattern: mutable default argument
    def __init__(self, samples=[], labels=[]):
        self.samples = samples
        self.labels  = labels
        self.meta    = {}

    def has_data(self):
        # BUG/DEPRECATED: dict.has_key() was removed in Python 3
        return self.meta.has_key("loaded")   # ← AttributeError in Python 3

    def info(self):
        # Old-style % string formatting
        print("Dataset: %d samples, %d classes" % (
            len(self.samples),
            len(set(self.labels))
        ))


def generate_fake_corpus(n=800, seed=42):
    """Generate a tiny fake movie-review corpus (pos / neg / neu)."""
    np.random.seed(seed)

    pos_templates = [
        "This film was absolutely wonderful and brilliant",
        "Amazing acting superb direction loved every moment",
        "Fantastic movie great performances highly recommended",
        "Incredible story beautifully filmed must watch",
        "Outstanding masterpiece loved the plot and actors",
    ]
    neg_templates = [
        "Terrible movie awful acting waste of time",
        "Horrible plot boring screenplay deeply disappointing",
        "Worst film ever dull and pointless avoid it",
        "Dreadful directing poor performances unwatchable rubbish",
        "Complete disaster bad script terrible characters hated it",
    ]
    neu_templates = [
        "Average film nothing special had some good moments",
        "Okay movie decent acting fairly standard plot",
        "Neither great nor bad just a regular film",
        "Some parts were good others were mediocre overall fine",
        "Watchable but forgettable not much to say about it",
    ]

    samples, labels = [], []
    for _ in range(n // 3):
        samples.append(np.random.choice(pos_templates) + " " +
                        np.random.choice(pos_templates))
        labels.append("positive")

        samples.append(np.random.choice(neg_templates) + " " +
                        np.random.choice(neg_templates))
        labels.append("negative")

        samples.append(np.random.choice(neu_templates) + " " +
                        np.random.choice(neu_templates))
        labels.append("neutral")

    return samples, labels


# ─── DEPRECATED: using np.matrix instead of np.ndarray ──────────────────────
def tfidf_to_dense_matrix(sparse_matrix):
    """
    Converts TF-IDF sparse matrix to dense.
    BUG: uses np.matrix (deprecated since NumPy 1.15, removed in 2.0)
    and .todense() returns np.matrix which behaves differently from ndarray.
    """
    # DEPRECATED: .todense() + np.matrix
    dense = np.array(sparse_matrix.todense())   # ← np.matrix deprecated/removed
    return dense


# ─── DEPRECATED: fit_transform on TEST set ───────────────────────────────────
def bad_preprocessing(X_train, X_test, config=CONFIG):
    """
    BUG: Calls fit_transform() on both train AND test set.
    This causes data leakage — test vocabulary influences vectorizer fit.
    Modern code uses fit_transform(X_train) then transform(X_test).
    """
    vectorizer = TfidfVectorizer(
        max_features=config["max_features"],
        ngram_range=config["ngram_range"],
        # DEPRECATED: 'analyzer' used to accept 'char_wb' only via old API call pattern
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)   # ← FIX: changed to transform
    return X_train_vec, X_test_vec, vectorizer


# ─── DEPRECATED: sklearn.cross_validation style loop ────────────────────────
def old_style_cross_val(X_sparse, y_encoded, n_folds=5):
    """
    Manual k-fold using the old StratifiedKFold API (sklearn < 0.18).
    Old API: StratifiedKFold(y, n_folds=k) — y passed to constructor.
    New API: StratifiedKFold(n_splits=k).split(X, y)
    """
    # DEPRECATED: passing y to constructor, using n_folds instead of n_splits
    skf = StratifiedKFold(y_encoded, n_folds=n_folds, random_state=0)   # ← old API

    fold_scores = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf):               # ← old iteration
        X_tr = X_sparse[train_idx]
        X_val = X_sparse[val_idx]
        y_tr  = y_encoded[train_idx]
        y_val = y_encoded[val_idx]

        clf = MultinomialNB(alpha=1.0)
        clf.fit(X_tr, y_tr)
        score = clf.score(X_val, y_val)
        fold_scores.append(score)

        # Old-style print
        print("  Fold %d: %.4f" % (fold_idx + 1, score))

    return fold_scores


# ─── DEPRECATED: GridSearchCV from sklearn.grid_search ───────────────────────
def old_grid_search(X_train, y_train):
    """
    Uses old grid_search module (removed in sklearn 0.20).
    Also uses deprecated 'refit=True' without specifying scoring explicitly
    (old default behaviour).
    """
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000)),
        ("clf",   MultinomialNB()),
    ])

    param_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__alpha":         [0.1, 0.5, 1.0, 2.0],
    }

    # DEPRECATED: from sklearn.grid_search import GridSearchCV
    # Also deprecated: cv=StratifiedKFold(y_train, n_folds=3) — old constructor
    gs = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=1,
        verbose=1,
    )
    gs.fit(X_train, y_train)

    # DEPRECATED: gs.best_score_ printed with % formatting
    print("Best CV score : %.4f" % gs.best_score_)
    print("Best params   : %s"   % str(gs.best_params_))
    return gs.best_estimator_


# ─── DEPRECATED: sys.maxint (Python 2 only) ──────────────────────────────────
def find_longest_review(samples):
    """
    BUG: sys.maxint does not exist in Python 3 (use sys.maxsize).
    """
    shortest_len = sys.maxsize       # ← FIX: changed to sys.maxsize
    longest_len  = 0
    for s in samples:
        l = len(s.split())
        if l < shortest_len:
            shortest_len = l
        if l > longest_len:
            longest_len = l
    return shortest_len, longest_len


# ─── DEPRECATED: pickle with protocol=2 (Python 2 compat) ───────────────────
def save_model_old(model, filepath="model.pkl"):
    """
    Saves model using pickle protocol 2 (Python 2 compat).
    Modern code uses protocol=4 or protocol=5 (Python 3.8+).
    Also opens in text mode 'w' instead of binary 'wb' — will crash in Py3.
    """
    with open(filepath, "wb") as f:     # ← FIX: changed to "wb" in Python 3
        pickle.dump(model, f, protocol=4)
    print("Model saved to %s" % filepath)


def load_model_old(filepath="model.pkl"):
    with open(filepath, "rb") as f:     # ← FIX: changed to "rb"
        return pickle.load(f)


# ─── DEPRECATED: manual confusion matrix plot (pre ConfusionMatrixDisplay) ───
def plot_confusion_matrix_old(cm, class_names, title="Confusion Matrix"):
    """
    Old manual matplotlib confusion matrix plot.
    Modern sklearn provides: ConfusionMatrixDisplay.from_predictions()
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title)
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)

    # Old pattern: manual threshold annotation loop
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/confusion_matrix_old.png", dpi=120)
    plt.show()


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Old-Style Sentiment Classifier  (circa 2013-2015)")
    print("=" * 60)

    # 1. Data
    print("\n[1] Generating corpus...")
    samples, labels = generate_fake_corpus(n=900)

    # Old-style class instantiation (no object inheritance)
    dataset = SentimentDataset(samples=samples, labels=labels)
    dataset.info()

    # 2. Encode labels
    le = LabelEncoder()
    y  = le.fit_transform(labels)
    print("Classes : %s" % str(le.classes_))

    # 3. Train/test split (using potentially deprecated import path)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        samples, y, test_size=0.2, random_state=CONFIG["random_state"], stratify=y
    )
    print("Train: %d  |  Test: %d" % (len(X_train_raw), len(X_test_raw)))

    # 4. BUG: bad preprocessing — fit_transform on test too
    print("\n[2] Vectorising (with data-leakage bug)...")
    X_train_vec, X_test_vec, vectorizer = bad_preprocessing(X_train_raw, X_test_raw)

    # 5. BUG: convert to deprecated np.matrix
    print("\n[3] Converting to dense np.matrix (deprecated)...")
    # Commented out to avoid MemoryError on large data; shown for illustration
    # dense = tfidf_to_dense_matrix(X_train_vec)

    # 6. Train Naive Bayes
    print("\n[4] Training MultinomialNB...")
    clf = MultinomialNB(alpha=1.0)
    clf.fit(X_train_vec, y_train)

    # 7. Evaluate
    y_pred = clf.predict(X_test_vec)
    acc    = accuracy_score(y_test, y_pred)
    print("\nTest Accuracy : %.4f" % acc)   # old % formatting
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # 8. Old-style confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix_old(cm, list(le.classes_))

    # 9. Grid search (uses deprecated module if available)
    print("\n[5] Running GridSearch (old-style)...")
    best_model = old_grid_search(X_train_raw, y_train)

    # 10. BUG: sys.maxint
    # Uncomment to trigger AttributeError in Python 3:
    # short, long_ = find_longest_review(samples)

    # 11. BUG: pickle with wrong file mode
    # Uncomment to trigger TypeError in Python 3:
    # save_model_old(best_model, "sentiment_model.pkl")

    print("\nDone.")


if __name__ == "__main__":
    main()

# CodeSentinal: created for you by RuchirAdnaik.