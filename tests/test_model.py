"""Basic tests for the credit risk model and encoders."""

import joblib
import numpy as np
import pandas as pd
import pytest

FEATURE_COLS = [
    "Age",
    "Sex",
    "Job",
    "Housing",
    "Saving accounts",
    "Checking account",
    "Credit amount",
    "Duration",
]

ENCODER_COLS = ["Sex", "Housing", "Saving accounts", "Checking account"]


@pytest.fixture(scope="module")
def model():
    return joblib.load("extra_trees_credit_model.pkl")


@pytest.fixture(scope="module")
def encoders():
    return {col: joblib.load(f"{col}_encoder.pkl") for col in ENCODER_COLS}


def test_model_loads(model):
    """Model file exists and is a classifier with predict/predict_proba."""
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_encoders_load(encoders):
    """All four categorical encoders load successfully."""
    assert set(encoders.keys()) == set(ENCODER_COLS)
    for col, enc in encoders.items():
        assert hasattr(enc, "transform")
        assert len(enc.classes_) > 0


def test_encoder_known_labels(encoders):
    """Encoders accept values the app uses in the UI."""
    # These must match the options in app.py
    sample = {
        "Sex": "male",
        "Housing": "own",
        "Saving accounts": "little",
        "Checking account": "little",
    }
    for col, value in sample.items():
        encoded = encoders[col].transform([value])
        assert encoded.shape == (1,)
        assert encoded[0] >= 0


def test_prediction_shape_and_range(model, encoders):
    """A single row prediction returns a valid class and probability."""
    row = pd.DataFrame(
        {
            "Age": [30],
            "Sex": [encoders["Sex"].transform(["male"])[0]],
            "Job": [1],
            "Housing": [encoders["Housing"].transform(["own"])[0]],
            "Saving accounts": [encoders["Saving accounts"].transform(["little"])[0]],
            "Checking account": [encoders["Checking account"].transform(["little"])[0]],
            "Credit amount": [2000],
            "Duration": [12],
        }
    )

    pred = model.predict(row)
    proba = model.predict_proba(row)

    assert pred.shape == (1,)
    assert pred[0] in (0, 1)

    assert proba.shape == (1, 2)
    assert np.isclose(proba[0].sum(), 1.0)
    assert (proba[0] >= 0).all() and (proba[0] <= 1).all()


def test_feature_column_order(model):
    """Model expects the same feature order the app uses."""
    # n_features_in_ is set on fitted sklearn models
    assert model.n_features_in_ == len(FEATURE_COLS)