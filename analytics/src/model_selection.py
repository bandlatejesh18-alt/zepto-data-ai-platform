import json

from pathlib import Path

from sklearn.ensemble import (
    RandomForestClassifier,
)
from sklearn.model_selection import (
    GridSearchCV,
)
from sklearn.pipeline import Pipeline

from preprocessing import (
    load_cleaned_dataset,
    split_dataset,
    create_preprocessor,
)
from utils import (
    MODELS_DIR,
)


def train_grid_search(
    X_train,
    y_train,
):
    """
    Perform GridSearchCV for
    Random Forest hyperparameter tuning.

    Args:
        X_train:
            Training features.

        y_train:
            Training labels.

    Returns:
        GridSearchCV:
            Fitted GridSearchCV object.
    """

    print("\nCreating preprocessing pipeline...")

    preprocessor = create_preprocessor()

    print("Creating Random Forest pipeline...")

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                RandomForestClassifier(
                    bootstrap=True,
                    oob_score=True,
                    random_state=42,
                ),
            ),
        ]
    )

    print("Creating parameter grid...")

    parameter_grid = {
        "classifier__n_estimators": [
            100,
            200,
        ],
        "classifier__max_depth": [
            None,
            10,
            20,
        ],
        "classifier__max_features": [
            "sqrt",
            "log2",
        ],
    }

    print("Running GridSearchCV...")

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )

    grid_search.fit(
        X_train,
        y_train,
    )

    print("Grid Search Completed Successfully!")

    return grid_search


def display_best_model(
    grid_search,
):
    """
    Display the best hyperparameters
    and Out-of-Bag (OOB) score.

    Args:
        grid_search:
            Fitted GridSearchCV object.
    """

    print("\nBest Hyperparameters")
    print("-" * 60)

    for parameter, value in grid_search.best_params_.items():

        print(
            f"{parameter}: {value}"
        )

    best_pipeline = grid_search.best_estimator_

    best_classifier = (
        best_pipeline.named_steps[
            "classifier"
        ]
    )

    print("\nBest Cross Validation Accuracy")
    print("-" * 60)

    print(
        f"{grid_search.best_score_:.4f}"
    )

    print("\nOut-of-Bag (OOB) Score")
    print("-" * 60)

    print(
        f"{best_classifier.oob_score_:.4f}"
    )


def save_best_parameters(
    grid_search,
):
    """
    Save the best Random Forest
    hyperparameters and OOB score
    to a JSON file.

    Args:
        grid_search:
            Fitted GridSearchCV object.
    """

    best_pipeline = grid_search.best_estimator_

    best_classifier = (
        best_pipeline.named_steps[
            "classifier"
        ]
    )

    results = {
        "best_parameters": grid_search.best_params_,
        "best_cross_validation_accuracy": round(
            grid_search.best_score_,
            4,
        ),
        "oob_score": round(
            best_classifier.oob_score_,
            4,
        ),
    }

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        MODELS_DIR /
        "best_random_forest.json"
    )

    with open(
        output_path,
        "w",
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
        )

    print(
        "\nBest model parameters saved to:"
    )

    print(output_path)


def main():
    """
    Execute Random Forest
    hyperparameter tuning.
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

    grid_search = train_grid_search(
        X_train,
        y_train,
    )

    display_best_model(
        grid_search,
    )

    save_best_parameters(
        grid_search,
    )

    print("\nHyperparameter tuning completed successfully!")


if __name__ == "__main__":
    main()