import joblib

from classification import (
    fit_models,
)
from preprocessing import (
    load_cleaned_dataset,
    split_dataset,
    create_preprocessor,
)
from utils import (
    MODELS_DIR,
)


def save_pipeline(
    pipeline,
):
    """
    Save the complete trained pipeline.

    Args:
        pipeline:
            Trained sklearn Pipeline.
    """

    output_path = (
        MODELS_DIR /
        "best_pipeline.joblib"
    )

    joblib.dump(
        pipeline,
        output_path,
    )

    print("\nPipeline saved to:")

    print(
        output_path,
    )
    

def load_pipeline():
    """
    Load the saved pipeline.

    Returns:
        Pipeline:
            Loaded sklearn pipeline.
    """

    model_path = (
        MODELS_DIR /
        "best_pipeline.joblib"
    )

    pipeline = joblib.load(
        model_path,
    )

    print(
        "\nPipeline loaded successfully!"
    )

    return pipeline


def verify_pipeline(
    pipeline,
    X_test,
    y_test,
):
    """
    Verify that the loaded pipeline
    predicts correctly on raw data.

    Args:
        pipeline:
            Loaded sklearn pipeline.

        X_test:
            Raw testing features.

        y_test:
            True testing labels.
    """

    predictions = pipeline.predict(
        X_test.iloc[:5],
    )

    comparison = X_test[
        [
            "pclass",
            "sex",
            "age",
            "sibsp",
            "parch",
            "fare",
            "embarked",
        ]
    ].iloc[:5].copy()

    comparison["Actual"] = (
        y_test.iloc[:5].values
    )

    comparison["Predicted"] = (
        predictions
    )

    print("\nPrediction Verification")
    print("-" * 60)

    print(comparison)


def main():
    """
    Save and reload the
    best classification pipeline.
    """

    print(
        "Loading cleaned dataset..."
    )

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

    best_pipeline = models[
        "Random Forest"
    ]

    save_pipeline(
        best_pipeline,
    )

    loaded_pipeline = load_pipeline()

    verify_pipeline(
        loaded_pipeline,
        X_test,
        y_test,
    )

    print(
        "\nModel persistence completed successfully!"
    )


if __name__ == "__main__":
    main()