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
    load_cleaned_dataset,
    split_dataset,
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


def compare_imbalance_methods(
    X_train,
    X_test,
    y_train,
    y_test,
):
    """
    Compare different imbalance handling strategies.

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
        pd.DataFrame:
            Comparison of Precision,
            Recall and F1 Score.
    """

    # --------------------------------------------------
    # Preprocess Data
    # --------------------------------------------------

    preprocessor = create_preprocessor()

    X_train_processed = preprocessor.fit_transform(
        X_train,
    )

    X_test_processed = preprocessor.transform(
        X_test,
    )

    # --------------------------------------------------
    # Baseline Logistic Regression
    # --------------------------------------------------

    baseline_model = LogisticRegression(
        random_state=42,
        max_iter=1000,
    )

    baseline_model.fit(
        X_train_processed,
        y_train,
    )

    baseline_predictions = baseline_model.predict(
        X_test_processed,
    )

    baseline_precision = precision_score(
        y_test,
        baseline_predictions,
    )

    baseline_recall = recall_score(
        y_test,
        baseline_predictions,
    )

    baseline_f1 = f1_score(
        y_test,
        baseline_predictions,
    )

    # --------------------------------------------------
    # Logistic Regression with Class Weight
    # --------------------------------------------------

    balanced_model = LogisticRegression(
        class_weight="balanced",
        random_state=42,
        max_iter=1000,
    )

    balanced_model.fit(
        X_train_processed,
        y_train,
    )

    balanced_predictions = balanced_model.predict(
        X_test_processed,
    )

    balanced_precision = precision_score(
        y_test,
        balanced_predictions,
    )

    balanced_recall = recall_score(
        y_test,
        balanced_predictions,
    )

    balanced_f1 = f1_score(
        y_test,
        balanced_predictions,
    )

    # --------------------------------------------------
    # SMOTE Oversampling
    # (Training data only)
    # --------------------------------------------------

    smote = SMOTE(
        random_state=42,
    )

    X_train_smote, y_train_smote = smote.fit_resample(
        X_train_processed,
        y_train,
    )

    smote_model = LogisticRegression(
        random_state=42,
        max_iter=1000,
    )

    smote_model.fit(
        X_train_smote,
        y_train_smote,
    )

    smote_predictions = smote_model.predict(
        X_test_processed,
    )

    smote_precision = precision_score(
        y_test,
        smote_predictions,
    )

    smote_recall = recall_score(
        y_test,
        smote_predictions,
    )

    smote_f1 = f1_score(
        y_test,
        smote_predictions,
    )

    # --------------------------------------------------
    # Comparison Table
    # --------------------------------------------------

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


def main():
    """
    Execute imbalance handling comparison.
    """

    print("Loading cleaned dataset...")

    df = load_cleaned_dataset()

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_dataset(
        df,
    )

    print_class_distribution(
        y_train,
    )

    comparison = compare_imbalance_methods(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    save_results(
        comparison,
    )

    print("\nImbalance handling completed successfully!")


if __name__ == "__main__":
    main()