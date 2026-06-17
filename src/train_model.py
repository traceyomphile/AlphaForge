import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.config import FEATURE_COLUMNS, RANDOM_STATE, TRAIN_SIZE, VAL_SIZE


def chronological_split(
    df: pd.DataFrame,
    train_size: float = TRAIN_SIZE,
    val_size: float = VAL_SIZE,
):
    """
    Chronological train/validation/test split.

    No shuffling. This is time-series data.
    """

    if len(df) < 100:
        raise ValueError(
            "Dataset has fewer than 100 usable rows. "
            "You can still experiment, but your backtest will be statistical confetti."
        )

    train_end = int(len(df) * train_size)
    val_end = int(len(df) * (train_size + val_size))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


def split_features_target(df: pd.DataFrame):
    """
    Split dataframe into X and y.
    """

    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")

    X = df[FEATURE_COLUMNS].copy()
    y = df["target"].astype(int).copy()

    return X, y


def scale_data(X_train, X_val, X_test):
    """
    Fit StandardScaler on training data only.
    Apply it to validation and test data.
    No future leakage.
    """

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return scaler, X_train_scaled, X_val_scaled, X_test_scaled


def train_baseline_models(X_train_scaled, y_train):
    """
    Train simple baseline ML models.

    Version 1 models:
    - Logistic Regression
    - Random Forest
    """

    if y_train.nunique() < 2:
        raise ValueError(
            "Training data contains fewer than 2 classes. "
            "The model cannot learn a classification problem from one class."
        )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    trained_models = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model

    return trained_models