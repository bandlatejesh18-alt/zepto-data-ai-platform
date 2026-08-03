from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
)

from utils import (
    PLOTS_DIR,
)


def evaluate_model(
    model,
    X_test,
    y_test,
    model_name,
):
    """
    Evaluate a trained classification model.

    Args:
        model:
            Trained pipeline.

        X_test:
            Testing features.

        y_test:
            Testing labels.

        model_name:
            Name of the classifier.

    Returns:
        dict:
            Complete evaluation results.
    """

    predictions = model.predict(
        X_test,
    )

    probabilities = model.predict_proba(
        X_test,
    )[
        :,
        1,
    ]

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
    )

    recall = recall_score(
        y_test,
        predictions,
    )

    f1 = f1_score(
        y_test,
        predictions,
    )

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities,
    )

    roc_auc = auc(
        fpr,
        tpr,
    )

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "AUC": roc_auc,
        "Predictions": predictions,
        "FPR": fpr,
        "TPR": tpr,
    }


def plot_confusion_matrix(
    y_test,
    predictions,
    model_name,
):
    """
    Plot and save the confusion matrix.

    Args:
        y_test:
            True labels.

        predictions:
            Predicted labels.

        model_name:
            Model name.
    """

    cm = confusion_matrix(
        y_test,
        predictions,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
    )

    plt.figure(
        figsize=(6, 6),
    )

    display.plot(
        cmap="Blues",
    )

    plt.title(
        f"{model_name} Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR /
        f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
    )

    plt.close()


def plot_roc_curve(
    fpr,
    tpr,
    auc_score,
    model_name,
):
    """
    Plot and save the ROC curve.

    Args:
        fpr:
            False Positive Rate.

        tpr:
            True Positive Rate.

        auc_score:
            Area Under Curve.

        model_name:
            Model name.
    """

    plt.figure(
        figsize=(8, 6),
    )

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"AUC = {auc_score:.3f}",
    )

    plt.plot(
        [
            0,
            1,
        ],
        [
            0,
            1,
        ],
        linestyle="--",
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        f"{model_name} ROC Curve"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR /
        f"{model_name.lower().replace(' ', '_')}_roc_curve.png"
    )

    plt.close()


def compare_models(
    results,
):
    """
    Create a comparison table for all classifiers.

    Args:
        results:
            List of evaluation dictionaries.

    Returns:
        pd.DataFrame
    """

    comparison = pd.DataFrame(
        [
            {
                "Model": result["Model"],
                "Accuracy": result["Accuracy"],
                "Precision": result["Precision"],
                "Recall": result["Recall"],
                "F1 Score": result["F1 Score"],
                "AUC": result["AUC"],
            }
            for result in results
        ]
    ).round(4)

    print("\nModel Comparison")
    print("-" * 80)

    print(comparison)

    return comparison


def evaluate_all_models(
    models,
    X_test,
    y_test,
):
    """
    Evaluate all trained models.

    Args:
        models:
            Dictionary containing trained models.

        X_test:
            Testing features.

        y_test:
            Testing labels.

    Returns:
        pd.DataFrame
    """

    results = []

    for model_name, model in models.items():

        print(
            f"\nEvaluating {model_name}..."
        )

        result = evaluate_model(
            model,
            X_test,
            y_test,
            model_name,
        )

        plot_confusion_matrix(
            y_test,
            result["Predictions"],
            model_name,
        )

        plot_roc_curve(
            result["FPR"],
            result["TPR"],
            result["AUC"],
            model_name,
        )

        results.append(
            result,
        )

    comparison = compare_models(
        results,
    )

    return comparison


def save_comparison_table(
    comparison_df: pd.DataFrame,
) -> None:
    """
    Save model comparison table.

    Args:
        comparison_df:
            Comparison dataframe.
    """

    output_path = (
        PLOTS_DIR.parent /
        "tables" /
        "classification_metrics.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_df.to_csv(
        output_path,
        index=False,
    )

    print(
        "\nComparison table saved to:"
    )

    print(output_path)