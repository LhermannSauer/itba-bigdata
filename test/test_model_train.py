import pytest
from unittest.mock import MagicMock
from scipy.sparse import csr_matrix

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))


import src.model_train as mt

def test_tfidf_vectorizer_config():
    """Check the TF-IDF vectorizer configuration."""
    vec = mt.TfidfVectorizer(max_features=50_000, ngram_range=(1, 2))

    assert vec.max_features == 50_000
    assert vec.ngram_range == (1, 2)


def test_run_experiment_mocked():
    """Run the experiment with a fake model and fake data."""
    # Fake classifier
    class DummyModel:
        def fit(self, X, y):
            return self
        def predict(self, X):
            return [y[0] for _ in range(X.shape[0])]

    dummy = DummyModel()

    # Fake data
    X_train = csr_matrix([[1, 0], [0, 1]])
    X_test = csr_matrix([[1, 0]])
    y_train = ["pos", "neg"]
    y_test = ["pos"]

    model, f1 = mt.run_experiment(
        model_name="dummy",
        classifier=dummy,
        params={"test": 1},
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )

    assert isinstance(model, DummyModel)
    assert f1 >= 0  # simple correctness check