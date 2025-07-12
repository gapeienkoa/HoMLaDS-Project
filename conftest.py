# conftest.py
import pathlib
import pandas as pd
import pytest
from skops.io import load as skops_load, get_untrusted_types
import numpy as np
import subprocess
import sys


def pytest_addoption(parser):
    """
    Register custom CLI options:
      --model-path  path/to/model.skops
      --data-dir    directory with X_test.csv and y_test.csv
      --r2-min      minimal acceptable R² (float)
      --mae-max     maximal acceptable MAE (float)
    """
    parser.addoption(
        "--model-path",
        action="store",
        default="models/price_model.skops",
        help="Path to SKOPS model file",
    )
    parser.addoption(
        "--data-dir",
        action="store",
        default="data",
        help="Directory containing X_test.csv and y_test.csv",
    )
    parser.addoption(
        "--r2-min",
        action="store",
        type=float,
        default=0.6,
        help="Minimal acceptable R² score",
    )
    parser.addoption(
        "--mae-max",
        action="store",
        type=float,
        default=64_000.0,
        help="Max acceptable MAE (same units as target)",
    )
    parser.addoption(
        "--rmse-max",
        action="store",
        type=float,
        default=126_000.0,
        help="Max acceptable RMSE (same units as target)",
    )
    parser.addoption(
        "--r2-difference-max",
        action="store",
        type=float,
        default=0.15,
        help="Max acceptable R² difference between train and test sets",
    )


# ---------- fixtures ---------- #

@pytest.fixture(scope="session")
def cli_options(request):
    """A namespace-like dict of CLI parameters, easier to pass around."""
    return {
        "model_path": pathlib.Path(request.config.getoption("--model-path")),
        "data_dir": pathlib.Path(request.config.getoption("--data-dir")),
        "r2_min": request.config.getoption("--r2-min"),
        "mae_max": request.config.getoption("--mae-max"),
        "rmse_max": request.config.getoption("--rmse-max"),
        "r2_difference_max": request.config.getoption("--r2-difference-max"),
    }


@pytest.fixture(scope="session")
def model(cli_options):
    """Load the model specified by --model-path (SKOPS format)."""
    path = cli_options["model_path"]
    if not path.exists():
        # attempt to train the model if training script is available
        train_script = pathlib.Path("train_model.py")
        if train_script.exists():
            subprocess.run([sys.executable, str(train_script)], check=True)
    if not path.exists():
        pytest.fail(f"Model file not found: {path}")
    untrusted = get_untrusted_types(file=path)
    return skops_load(path, trusted=untrusted)


@pytest.fixture(scope="session")
def test_data(cli_options):
    """
    Load X_test and y_test saved as CSV.

    * X_test.csv — feature matrix (can include headers; order must match training)
    * y_test.csv — single-column target vector
    """
    data_dir = cli_options["data_dir"]
    X_path = data_dir / "X_test.csv"
    y_path = data_dir / "y_test.csv"

    if not X_path.exists() or not y_path.exists():
        pytest.fail(f"Test data not found in {data_dir}. "
                    "Expected X_test.csv and y_test.csv")

    X_test = pd.read_csv(X_path)
    y_test = pd.read_csv(y_path).squeeze("columns")  # Series

    return X_test, y_test


@pytest.fixture(scope="session")
def train_data(cli_options):
    """
    Load X_train and y_train saved as CSV.

    * X_train.csv — feature matrix (can include headers; order must match training)
    * y_train.csv — single-column target vector
    """
    data_dir = cli_options["data_dir"]
    X_path = data_dir / "X_train.csv"
    y_path = data_dir / "y_train.csv"

    if not X_path.exists() or not y_path.exists():
        pytest.fail(f"Test data not found in {data_dir}. "
                    "Expected X_train.csv and y_train.csv")

    X_train = pd.read_csv(X_path)
    y_train = pd.read_csv(y_path).squeeze("columns")  # Series

    return X_train, y_train