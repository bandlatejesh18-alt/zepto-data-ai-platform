from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree,
)
from sklearn.ensemble import RandomForestClassifier
from preprocessing import (
    load_cleaned_dataset,
    split_dataset,
    create_preprocessor,
)

from utils import (
    PLOTS_DIR,
)


def train_logistic_regression(
    preprocessor,
):
    """
    Create a Logistic Regression pipeline.

    Args:
        preprocessor:
            ColumnTransformer.

    Returns:
        Pipeline:
            Logistic Regression pipeline.
    """

    pipeline = Pipeline(
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

    return pipeline


def train_decision_tree(
    preprocessor,
):
    """
    Create a Decision Tree pipeline.

    Args:
        preprocessor:
            ColumnTransformer.

    Returns:
        Pipeline:
            Decision Tree pipeline.
    """

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                DecisionTreeClassifier(
                    random_state=42,
                ),
            ),
        ]
    )

    return pipeline


def train_random_forest(
    preprocessor,
):
    """
    Create a Random Forest pipeline.

    Args:
        preprocessor:
            ColumnTransformer.

    Returns:
        Pipeline:
            Random Forest pipeline.
    """

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=42,
                ),
            ),
        ]
    )

    return pipeline


def fit_models(
    X_train,
    y_train,
    preprocessor,
):
    """
    Train all classification models.

    Args:
        X_train:
            Training features.

        y_train:
            Training labels.

        preprocessor:
            ColumnTransformer used for preprocessing.

    Returns:
        tuple:
            Trained Logistic Regression pipeline,
            Decision Tree pipeline,
            Random Forest pipeline.
    """

    logistic_model = train_logistic_regression(
        preprocessor,
    )

    decision_tree_model = train_decision_tree(
        preprocessor,
    )

    random_forest_model = train_random_forest(
        preprocessor,
    )

    print("\nTraining Logistic Regression...")

    logistic_model.fit(
        X_train,
        y_train,
    )

    print("Completed")

    print("\nTraining Decision Tree...")

    decision_tree_model.fit(
        X_train,
        y_train,
    )

    print("Completed")

    print("\nTraining Random Forest...")

    random_forest_model.fit(
        X_train,
        y_train,
    )

    print("Completed")

    return {
        "Logistic Regression": logistic_model,
        "Decision Tree": decision_tree_model,
        "Random Forest": random_forest_model,
    }


def visualize_decision_tree(
    decision_tree_pipeline,
):
    """
    Plot the trained Decision Tree.

    Args:
        decision_tree_pipeline:
            Trained pipeline.
    """

    classifier = (
        decision_tree_pipeline.named_steps[
            "classifier"
        ]
    )

    preprocessor = (
        decision_tree_pipeline.named_steps[
            "preprocessor"
        ]
    )

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    plt.figure(
        figsize=(22, 12),
    )

    plot_tree(
        classifier,
        feature_names=feature_names,
        class_names=[
            "Not Survived",
            "Survived",
        ],
        filled=True,
        rounded=True,
        fontsize=8,
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR /
        "decision_tree.png"
    )

    plt.close()
    

def main():
    """
    Execute the complete
    classification training pipeline.
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

    preprocessor = create_preprocessor()

    models = fit_models(
        X_train,
        y_train,
        preprocessor,
    )

    visualize_decision_tree(
        models["Decision Tree"],
    )

    print("\nTraining Completed Successfully!")

    print(
        f"\nDecision Tree saved to:\n{PLOTS_DIR}"
    )


if __name__ == "__main__":
    main()