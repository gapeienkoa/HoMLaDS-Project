import pathlib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    StackingRegressor,
)
from sklearn.linear_model import Ridge
from skops.io import dump as skops_dump
from sklearn.model_selection import cross_val_score

DATA_DIR = pathlib.Path("./data")
MODEL_PATH = pathlib.Path("./models/price_model.skops")

NUM_FEATURES = ["0", "1", "2", "3", "4"]


def load_data(data_dir: pathlib.Path):
    X_train = pd.read_csv(data_dir / "X_train.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv").squeeze("columns")
    return X_train, y_train


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    cat_features = [c for c in X.columns if c not in NUM_FEATURES]
    numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_transformer, NUM_FEATURES),
            ("cat", cat_transformer, cat_features),
        ]
    )
    rf = RandomForestRegressor(
        n_estimators=200,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    extra = ExtraTreesRegressor(
        n_estimators=200,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    stack = StackingRegressor(
        estimators=[("rf", rf), ("extra", extra)],
        final_estimator=Ridge(alpha=1.0),
    )
    model = Pipeline(
        [
            ("preprocess", preprocessor),
            ("regressor", stack),
        ]
    )
    return model


def main(data_dir: pathlib.Path = DATA_DIR, model_path: pathlib.Path = MODEL_PATH):
    X_train, y_train = load_data(data_dir)
    pipeline = build_pipeline(X_train)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="r2")
    print(
        f"Cross-validation R²: mean={cv_scores.mean():.3f}, std={cv_scores.std():.3f}"
    )
    pipeline.fit(X_train, y_train)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    skops_dump(pipeline, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
