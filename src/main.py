import joblib
import numpy as np
import pandas as pd

from src.backtest import format_backtest_report, run_backtest, save_equity_curve_chart
from src.config import (
    AUTO_DOWNLOAD_DATA,
    BEST_MODEL_PATH,
    DEFAULT_LABEL_THRESHOLD,
    FEATURE_COLUMNS,
    FEATURES_PATH,
    FORCE_DOWNLOAD_DATA,
    LABEL_TO_SIGNAL,
    MODEL_REPORT_PATH,
    PAIR,
    PREDICTIONS_PATH,
    RAW_DATA_PATH,
    SCALER_PATH,
    ensure_directories,
)
from src.data_loader import load_price_data
from src.download_data import ensure_eurusd_data_exists
from src.evaluate_model import (
    always_hold_baseline,
    evaluate_models,
    format_evaluation_report,
    random_baseline,
    select_best_model,
    sma_crossover_baseline,
)
from src.feature_engineering import create_features
from src.labelling import create_labels
from src.train_model import (
    chronological_split,
    scale_data,
    split_features_target,
    train_baseline_models,
)
from src.config import BACKTEST_REPORT_PATH


def get_prediction_confidence(model, X_scaled) -> np.ndarray:
    """
    Return max predicted probability if the model supports predict_proba.
    Otherwise return NaN.
    """

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_scaled)
        return probabilities.max(axis=1)

    return np.full(shape=(X_scaled.shape[0],), fill_value=np.nan)


def assign_split_labels(full_df, train_df, val_df, test_df):
    """
    Add train/validation/test/latest labels to the full dataframe.
    """

    split_series = pd.Series("unlabeled", index=full_df.index)

    split_series.loc[train_df.index] = "train"
    split_series.loc[val_df.index] = "validation"
    split_series.loc[test_df.index] = "test"

    return split_series


def main():
    ensure_directories()

    if AUTO_DOWNLOAD_DATA:
        ensure_eurusd_data_exists(
            raw_data_path=RAW_DATA_PATH,
            force_download=FORCE_DOWNLOAD_DATA,
        )

    print("Loading raw EUR/USD data...")
    raw_df = load_price_data(str(RAW_DATA_PATH))

    print("Creating features...")
    features_df = create_features(raw_df)

    print("Creating labels...")
    labeled_all_df = create_labels(
        features_df,
        threshold=DEFAULT_LABEL_THRESHOLD,
        drop_unlabeled=False,
    )

    labeled_trainable_df = labeled_all_df.dropna(subset=["target"]).copy()
    labeled_trainable_df["target"] = labeled_trainable_df["target"].astype(int)

    print("Splitting data chronologically...")
    train_df, val_df, test_df = chronological_split(labeled_trainable_df)

    X_train, y_train = split_features_target(train_df)
    X_val, y_val = split_features_target(val_df)
    X_test, y_test = split_features_target(test_df)

    print("Scaling data...")
    scaler, X_train_scaled, X_val_scaled, X_test_scaled = scale_data(
        X_train,
        X_val,
        X_test,
    )

    print("Training baseline models...")
    models = train_baseline_models(X_train_scaled, y_train)

    print("Evaluating models...")
    validation_results = evaluate_models(models, X_val_scaled, y_val)
    test_results = evaluate_models(models, X_test_scaled, y_test)

    best_model_name = select_best_model(validation_results)
    best_model = models[best_model_name]

    print(f"Best model selected: {best_model_name}")

    best_test_predictions = best_model.predict(X_test_scaled)

    baseline_results = {
        "Always HOLD": always_hold_baseline(y_test),
        "Random BUY/HOLD/SELL": random_baseline(y_test),
        "Simple MA5/MA20 Crossover": sma_crossover_baseline(test_df),
    }

    print("Saving model report...")
    model_report = format_evaluation_report(
        best_model_name=best_model_name,
        validation_results=validation_results,
        test_results=test_results,
        baseline_results=baseline_results,
        y_test=y_test,
        best_test_predictions=best_test_predictions,
    )

    MODEL_REPORT_PATH.write_text(model_report, encoding="utf-8")

    print("Generating predictions for all available processed rows...")

    all_features = labeled_all_df[FEATURE_COLUMNS].copy()
    all_features_scaled = scaler.transform(all_features)

    all_predictions = best_model.predict(all_features_scaled)
    all_confidence = get_prediction_confidence(best_model, all_features_scaled)

    predictions_df = labeled_all_df.copy()
    predictions_df["pair"] = PAIR
    predictions_df["prediction"] = all_predictions.astype(int)
    predictions_df["signal"] = predictions_df["prediction"].map(LABEL_TO_SIGNAL)
    predictions_df["confidence"] = all_confidence
    predictions_df["actual_signal"] = predictions_df["target"].map(LABEL_TO_SIGNAL)

    split_labels = assign_split_labels(
        full_df=predictions_df,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )

    predictions_df["split"] = split_labels

    print("Saving processed features and predictions...")
    labeled_all_df.to_csv(FEATURES_PATH, index=False)
    predictions_df.to_csv(PREDICTIONS_PATH, index=False)

    print("Running backtest on test set only...")
    test_predictions_df = predictions_df[predictions_df["split"] == "test"].copy()

    backtest_df, backtest_metrics = run_backtest(test_predictions_df)

    print("Saving backtest report and equity curve...")
    backtest_report = format_backtest_report(backtest_metrics)

    BACKTEST_REPORT_PATH.write_text(backtest_report, encoding="utf-8")

    save_equity_curve_chart(backtest_df)

    print("Saving best model and scaler...")
    joblib.dump(best_model, BEST_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print("")
    print("Pipeline completed successfully.")
    print(f"Best model: {best_model_name}")
    print(f"Processed features saved to: {FEATURES_PATH}")
    print(f"Predictions saved to: {PREDICTIONS_PATH}")
    print(f"Model report saved to: {MODEL_REPORT_PATH}")
    print(f"Backtest report saved to: {BACKTEST_REPORT_PATH}")
    print(f"Best model saved to: {BEST_MODEL_PATH}")
    print(f"Scaler saved to: {SCALER_PATH}")


if __name__ == "__main__":
    main()