from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from utils import (
    PROCESSED_DATA_DIR,
)


def load_cleaned_dataset() -> pd.DataFrame:
    """
    Load the cleaned Titanic dataset.

    Returns:
        pd.DataFrame:
            Cleaned dataframe.
    """

    dataset_path = (
        PROCESSED_DATA_DIR /
        "titanic_cleaned.csv"
    )

    return pd.read_csv(dataset_path)


def split_dataset(
    df: pd.DataFrame,
):
    """
    Split the dataset into
    training and testing sets
    using stratified sampling.

    Args:
        df (pd.DataFrame):
            Cleaned dataframe.

    Returns:
        tuple:
            X_train,
            X_test,
            y_train,
            y_test
    """

    X = df.drop(
        columns=[
            "survived",
        ]
    )

    y = df[
        "survived"
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("\nDataset Split")
    print("-" * 50)

    print(f"Training Samples : {len(X_train)}")
    print(f"Testing Samples  : {len(X_test)}")

    print("\nClass Distribution")

    print("\nTraining")

    print(
        y_train.value_counts(
            normalize=True,
        ) * 100
    )

    print("\nTesting")

    print(
        y_test.value_counts(
            normalize=True,
        ) * 100
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


def create_preprocessor():
    """
    Create the preprocessing pipeline.

    Returns:
        ColumnTransformer:
            Complete preprocessing pipeline.
    """

    # ------------------------------------------
    # Numeric Features
    # ------------------------------------------

    numeric_features = [
        "age",
        "sibsp",
        "parch",
        "fare",
        "pclass",
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

    # ------------------------------------------
    # Categorical Features
    # ------------------------------------------

    categorical_features = [
        "sex",
        "embarked",
    ]

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

    # ------------------------------------------
    # Combine Pipelines
    # ------------------------------------------

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
        ],
        remainder="drop",
    )

    return preprocessor


def main():
    """
    Execute preprocessing steps.
    """

    print("Loading cleaned dataset...")

    df = load_cleaned_dataset()

    X_train, X_test, y_train, y_test = split_dataset(df)

    preprocessor = create_preprocessor()

    print("\nPreprocessing Pipeline")
    print("-" * 50)

    print(preprocessor)


if __name__ == "__main__":
    main()