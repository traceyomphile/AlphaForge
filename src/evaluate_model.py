import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.config import LABEL_NAMES, LABEL_ORDER, LABEL_TO_SIGNAL, RANDOM_STATE


def evaluate_predictions(y_true, y_pred) -> dict:
    """
    Evaluate predictions using standard classification metrics.
    """

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        average="macro",
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)

    report = classification_report(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        target_names=LABEL_NAMES,
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "confusion_matrix": cm,
        "classification_report": report,
    }


def evaluate_models(models: dict, X_scaled, y_true) -> dict:
    """
    Evaluate multiple trained models.
    """

    results = {}

    for name, model in models.items():
        y_pred = model.predict(X_scaled)
        results[name] = evaluate_predictions(y_true, y_pred)

    return results


def always_hold_baseline(y_true):
    """
    Baseline that always predicts HOLD.
    """

    y_pred = np.zeros(len(y_true), dtype=int)
    return evaluate_predictions(y_true, y_pred)


def random_baseline(y_true):
    """
    Random BUY/HOLD/SELL baseline.
    """

    rng = np.random.default_rng(RANDOM_STATE)
    y_pred = rng.choice(LABEL_ORDER, size=len(y_true))
    return evaluate_predictions(y_true, y_pred)


def sma_crossover_baseline(test_df: pd.DataFrame):
    """
    Simple moving average crossover baseline.

    BUY if MA5 > MA20.
    SELL if MA5 < MA20.
    HOLD if equal.
    """

    y_true = test_df["target"].astype(int)

    y_pred = np.where(
        test_df["ma5"] > test_df["ma20"],
        1,
        np.where(test_df["ma5"] < test_df["ma20"], -1, 0),
    )

    return evaluate_predictions(y_true, y_pred)


def select_best_model(validation_results: dict) -> str | None:
    """
    Select best model using validation macro F1.
    Accuracy is used as tie-breaker.
    """

    best_name = None
    best_score = -1
    best_accuracy = -1

    for name, result in validation_results.items():
        macro_f1 = result["macro_f1"]
        accuracy = result["accuracy"]

        if macro_f1 > best_score or (
            macro_f1 == best_score and accuracy > best_accuracy
        ):
            best_name = name
            best_score = macro_f1
            best_accuracy = accuracy

    return best_name


def prediction_distribution(y_pred) -> dict:
    """
    Count prediction labels.
    """

    values, counts = np.unique(y_pred, return_counts=True)

    distribution = {}

    for value, count in zip(values, counts):
        distribution[LABEL_TO_SIGNAL.get(int(value), str(value))] = int(count)

    return distribution


def class_distribution(y_true) -> dict:
    """
    Count actual labels.
    """

    values, counts = np.unique(y_true, return_counts=True)

    distribution = {}

    for value, count in zip(values, counts):
        distribution[LABEL_TO_SIGNAL.get(int(value), str(value))] = int(count)

    return distribution


def format_evaluation_report(
    best_model_name: str | None,
    validation_results: dict,
    test_results: dict,
    baseline_results: dict,
    y_test,
    best_test_predictions,
) -> str:
    """
    Create readable model evaluation report.
    """

    best_test = test_results[best_model_name]

    lines = []

    lines.append("AI Forex Trading Research System - Version 1")
    lines.append("Model Evaluation Report")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"best_model: {best_model_name}")
    lines.append(f"test_accuracy: {best_test['accuracy']:.6f}")
    lines.append(f"test_macro_precision: {best_test['macro_precision']:.6f}")
    lines.append(f"test_macro_recall: {best_test['macro_recall']:.6f}")
    lines.append(f"test_macro_f1: {best_test['macro_f1']:.6f}")
    lines.append("")
    lines.append("Class distribution in test set:")
    lines.append(str(class_distribution(y_test)))
    lines.append("")
    lines.append("Prediction distribution for best model:")
    lines.append(str(prediction_distribution(best_test_predictions)))
    lines.append("")
    lines.append("=" * 60)
    lines.append("Validation Results")
    lines.append("=" * 60)

    for model_name, result in validation_results.items():
        lines.append("")
        lines.append(f"Model: {model_name}")
        lines.append(f"Accuracy: {result['accuracy']:.6f}")
        lines.append(f"Macro Precision: {result['macro_precision']:.6f}")
        lines.append(f"Macro Recall: {result['macro_recall']:.6f}")
        lines.append(f"Macro F1: {result['macro_f1']:.6f}")
        lines.append("Confusion Matrix rows/cols = SELL, HOLD, BUY:")
        lines.append(str(result["confusion_matrix"]))

    lines.append("")
    lines.append("=" * 60)
    lines.append("Test Results")
    lines.append("=" * 60)

    for model_name, result in test_results.items():
        lines.append("")
        lines.append(f"Model: {model_name}")
        lines.append(f"Accuracy: {result['accuracy']:.6f}")
        lines.append(f"Macro Precision: {result['macro_precision']:.6f}")
        lines.append(f"Macro Recall: {result['macro_recall']:.6f}")
        lines.append(f"Macro F1: {result['macro_f1']:.6f}")
        lines.append("Confusion Matrix rows/cols = SELL, HOLD, BUY:")
        lines.append(str(result["confusion_matrix"]))
        lines.append("")
        lines.append(result["classification_report"])

    lines.append("")
    lines.append("=" * 60)
    lines.append("Baseline Comparisons")
    lines.append("=" * 60)

    for baseline_name, result in baseline_results.items():
        lines.append("")
        lines.append(f"Baseline: {baseline_name}")
        lines.append(f"Accuracy: {result['accuracy']:.6f}")
        lines.append(f"Macro Precision: {result['macro_precision']:.6f}")
        lines.append(f"Macro Recall: {result['macro_recall']:.6f}")
        lines.append(f"Macro F1: {result['macro_f1']:.6f}")
        lines.append("Confusion Matrix rows/cols = SELL, HOLD, BUY:")
        lines.append(str(result["confusion_matrix"]))

    lines.append("")
    lines.append("Warning:")
    lines.append(
        "Good classification metrics do not guarantee profitable trading. "
        "Bad metrics are usually honest. Good metrics are often suspicious. Check leakage."
    )

    return "\n".join(lines)