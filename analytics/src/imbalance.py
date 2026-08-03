import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)

from preprocessing import (
    create_preprocessor,
)

from utils import (
    PLOTS_DIR,
)


def print_class_distribution(
    y_train,
):
    """
    Print training class balance.
    """

    print("\nTraining Class Distribution")

    print("-" * 50)

    print(
        y_train.value_counts()
    )

    print("\nPercentage")

    print(
        (
            y_train.value_counts(
                normalize=True
            ) * 100
        ).round(2)
    )


def evaluate_pipeline(
    pipeline,
    X_train,
    X_test,
    y_train,
    y_test,
):
    """
    Train and evaluate a pipeline.

    Args:
        pipeline:
            Machine learning pipeline.

        X_train:
            Training features.

        X_test:
            Testing features.

        y_train:
            Training labels.

        y_test:
            Testing labels.

    Returns:
        tuple:
            Precision,
            Recall,
            F1 Score.
    """

    pipeline.fit(
        X_train,
        y_train,
    )

    predictions = pipeline.predict(
        X_test,
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

    return (
        precision,
        recall,
        f1,
    )


def compare_imbalance_methods(
    X_train,
    X_test,
    y_train,
    y_test,
):
    """
    Compare imbalance handling methods.

    Args:
        X_train:
            Training features.

        X_test:
            Testing features.

        y_train:
            Training labels.

        y_test:
            Testing labels.

    Returns:
        pd.DataFrame
    """

    preprocessor = create_preprocessor()

    # ------------------------------------------
    # Baseline
    # ------------------------------------------

    baseline_pipeline = ImbPipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                LogisticRegression(
                    random_state=42,
                    max_iter=1000,
                ),
            ),
        ]
    )

    baseline_precision, baseline_recall, baseline_f1 = evaluate_pipeline(
        baseline_pipeline,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # ------------------------------------------
    # Class Weight
    # ------------------------------------------

    balanced_pipeline = ImbPipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    random_state=42,
                    max_iter=1000,
                ),
            ),
        ]
    )

    balanced_precision, balanced_recall, balanced_f1 = evaluate_pipeline(
        balanced_pipeline,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # ------------------------------------------
    # SMOTE
    # ------------------------------------------

    smote_pipeline = ImbPipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "smote",
                SMOTE(
                    random_state=42,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    random_state=42,
                    max_iter=1000,
                ),
            ),
        ]
    )

    smote_precision, smote_recall, smote_f1 = evaluate_pipeline(
        smote_pipeline,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    comparison = pd.DataFrame(
        {
            "Method": [
                "Baseline",
                "Class Weight",
                "SMOTE",
            ],
            "Precision": [
                baseline_precision,
                balanced_precision,
                smote_precision,
            ],
            "Recall": [
                baseline_recall,
                balanced_recall,
                smote_recall,
            ],
            "F1 Score": [
                baseline_f1,
                balanced_f1,
                smote_f1,
            ],
        }
    ).round(4)

    print("\nImbalance Handling Comparison")
    print("-" * 70)

    print(comparison)

    return comparison


def save_results(
    comparison_df,
):
    """
    Save imbalance comparison table.

    Args:
        comparison_df:
            Comparison dataframe.
    """

    output_path = (
        PLOTS_DIR.parent /
        "tables" /
        "imbalance_comparison.csv"
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
        "\nResults saved to:"
    )

    print(output_path)