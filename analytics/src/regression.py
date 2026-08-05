import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import (
    ColumnTransformer,
)
from sklearn.impute import (
    SimpleImputer,
)
from sklearn.linear_model import (
    LinearRegression,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    train_test_split,
)
from sklearn.pipeline import (
    Pipeline,
)
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)
from preprocessing import (
    load_cleaned_dataset,
)
from utils import (
    PLOTS_DIR,
)


def load_regression_data():
    """
    Load the cleaned Titanic dataset
    and prepare features and target
    for multivariate linear regression.

    Returns:
        tuple:
            X_train,
            X_test,
            y_train,
            y_test
    """

    print("Loading cleaned dataset...")

    df = load_cleaned_dataset()

    X = df[
        [
            "pclass",
            "sex",
            "age",
            "sibsp",
            "parch",
            "embarked",
        ]
    ]

    y = df[
        "fare"
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    print("\nRegression Dataset Split")
    print("-" * 50)

    print(
        f"Training Samples : {len(X_train)}"
    )

    print(
        f"Testing Samples  : {len(X_test)}"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


def create_regression_preprocessor():
    """
    Create the preprocessing pipeline
    for the regression model.

    Returns:
        ColumnTransformer:
            Regression preprocessing pipeline.
    """

    numeric_features = [
        "age",
        "sibsp",
        "parch",
        "pclass",
    ]

    categorical_features = [
        "sex",
        "embarked",
    ]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    return preprocessor


def create_regression_pipeline():
    """
    Create the complete regression
    pipeline consisting of the
    preprocessor and Linear Regression.

    Returns:
        Pipeline:
            Complete regression pipeline.
    """

    print("\nCreating regression pipeline...")

    preprocessor = create_regression_preprocessor()

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "regressor",
                LinearRegression(),
            ),
        ]
    )

    return pipeline


def train_regression_model(
    X_train,
    X_test,
    y_train,
):
    """
    Train the regression model
    and generate predictions.

    Args:
        X_train:
            Training features.

        X_test:
            Testing features.

        y_train:
            Training target.

    Returns:
        tuple:
            Trained pipeline,
            Predicted values.
    """

    pipeline = create_regression_pipeline()

    print("\nTraining Linear Regression Model...")

    pipeline.fit(
        X_train,
        y_train,
    )

    predictions = pipeline.predict(
        X_test,
    )

    print("Training Completed Successfully!")

    return (
        pipeline,
        predictions,
    )


def evaluate_regression(
    X_test,
    y_test,
    predictions,
):
    """
    Evaluate the regression model.

    Args:
        X_test:
            Testing features.

        y_test:
            True target values.

        predictions:
            Predicted target values.

    Returns:
        dict:
            Regression metrics.
    """

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    rmse = (
        mean_squared_error(
            y_test,
            predictions,
        )
        ** 0.5
    )

    r2 = r2_score(
        y_test,
        predictions,
    )

    number_of_samples = len(
        y_test,
    )

    number_of_features = X_test.shape[
        1
    ]

    adjusted_r2 = 1 - (
        (1 - r2)
        * (number_of_samples - 1)
        / (number_of_samples - number_of_features - 1)
    )

    results = {
        "MAE": mae,
        "RMSE": rmse,
        "R²": r2,
        "Adjusted R²": adjusted_r2,
    }

    print("\nRegression Metrics")
    print("-" * 60)

    for metric, value in results.items():

        print(
            f"{metric:<20}: {value:.4f}"
        )

    return results


def plot_residuals(
    y_test,
    predictions,
):
    """
    Plot and save the residual plot.

    Args:
        y_test:
            Actual fare values.

        predictions:
            Predicted fare values.
    """

    residuals = (
        y_test -
        predictions
    )

    plt.figure(
        figsize=(8, 6),
    )

    plt.scatter(
        predictions,
        residuals,
        alpha=0.7,
    )

    plt.axhline(
        y=0,
        color="red",
        linestyle="--",
        linewidth=2,
    )

    plt.xlabel(
        "Predicted Fare"
    )

    plt.ylabel(
        "Residuals"
    )

    plt.title(
        "Residual Plot"
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR /
        "residual_plot.png"
    )

    plt.close()

    print("\nResidual plot saved to:")

    print(
        PLOTS_DIR /
        "residual_plot.png"
    )


def save_regression_results(
    results,
):
    """
    Save regression evaluation
    metrics to a CSV file.

    Args:
        results:
            Dictionary containing
            regression metrics.
    """

    results_df = pd.DataFrame(
        {
            "Metric": list(
                results.keys(),
            ),
            "Value": list(
                results.values(),
            ),
        }
    )

    output_path = (
        PLOTS_DIR.parent /
        "tables" /
        "regression_results.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print("\nRegression results saved to:")

    print(
        output_path
    )


def main():
    """
    Execute the complete
    regression pipeline.
    """

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = load_regression_data()

    pipeline, predictions = train_regression_model(
        X_train,
        X_test,
        y_train,
    )

    regression_results = evaluate_regression(
        X_test,
        y_test,
        predictions,
    )

    plot_residuals(
        y_test,
        predictions,
    )

    save_regression_results(
        regression_results,
    )

    print("\nRegression analysis completed successfully!")


if __name__ == "__main__":
    main()