# tests/test_skops_model.py
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pytest


def test_prediction_shape(model, test_data):
    """
    Make sure model returns exactly one prediction per row.
    """
    X_test, _ = test_data
    preds = model.predict(X_test)
    assert preds.shape[0] == X_test.shape[0]


def train_prediction_shape(model, train_data):
    """
    Make sure model returns exactly one prediction per row.
    """
    X_train, _ = train_data
    preds = model.predict(X_train)
    assert preds.shape[0] == X_train.shape[0]


def test_regression_quality_on_train_data(model, train_data, cli_options):
    """
    Validate model quality against configurable R² and MAE thresholds.
    """
    X_train, y_train = train_data
    preds = model.predict(X_train)

    r2 = r2_score(y_train, preds)
    mae = mean_absolute_error(y_train, preds)
    rmse = np.sqrt(mean_squared_error(y_train, preds))

    r2_min = cli_options["r2_min"]
    mae_max = cli_options["mae_max"]
    rmse_max = cli_options["rmse_max"]

    print(f"Model results on train data → R²={r2:.3f}, MAE={mae:.3f}, RMSE={rmse:.3f}")

    assert r2 >= r2_min, (
        f"R²={r2:.3f} below threshold {r2_min:.3f}"
    )
    assert mae <= mae_max, (
        f"MAE={mae:.2f} exceeds threshold {mae_max:.2f}"
    )

    assert rmse <= rmse_max, f"RMSE={rmse:.2f} exceeds threshold {rmse_max:.2f}"



def train_prediction_shape(model, train_data):
    """
    Make sure model returns exactly one prediction per row.
    """
    X_train, _ = train_data
    preds = model.predict(X_train)
    assert preds.shape[0] == X_train.shape[0]


def test_regression_quality(model, test_data, cli_options):
    """
    Validate model quality against configurable R² and MAE thresholds.
    """
    X_test, y_test = test_data
    preds = model.predict(X_test)

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    r2_min = cli_options["r2_min"]
    mae_max = cli_options["mae_max"]
    rmse_max = cli_options["rmse_max"]

    print(f"Model results on test data → R²={r2:.3f}, MAE={mae:.3f}, RMSE={rmse:.3f}")

    assert r2 >= r2_min, (
        f"R²={r2:.3f} below threshold {r2_min:.3f}"
    )
    assert mae <= mae_max, (
        f"MAE={mae:.2f} exceeds threshold {mae_max:.2f}"
    )

    assert rmse <= rmse_max, f"RMSE={rmse:.2f} exceeds threshold {rmse_max:.2f}"

def test_r2_difference_between_train_and_test(model, train_data, test_data, cli_options):
    """
    Validate that the R² score difference between train and test sets is within a specified maximum.
    """
    # Get training data
    X_train, y_train = train_data
    preds_train = model.predict(X_train)
    r2_train = r2_score(y_train, preds_train)

    # Get test data
    X_test, y_test = test_data
    preds_test = model.predict(X_test)
    r2_test = r2_score(y_test, preds_test)

    # Calculate the difference
    r2_difference = abs(r2_train - r2_test)

    print(f"R² on train data: {r2_train:.3f}, R² on test data: {r2_test:.3f}, Difference: {r2_difference:.3f}")

    # Get the maximum allowed difference from cli_options
    r2_difference_max = cli_options["r2_difference_max"]

    # Assert that the difference is within the specified maximum
    assert r2_difference <= r2_difference_max, (
        f"R² difference {r2_difference:.3f} exceeds the maximum allowed {r2_difference_max:.3f}"
    )
