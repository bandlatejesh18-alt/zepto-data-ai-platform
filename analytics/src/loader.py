import pandas as pd
import seaborn as sns

from utils import RAW_DATA_DIR


def load_dataset() -> pd.DataFrame:
    """
    Load the Titanic dataset using Seaborn.

    Returns:
        pd.DataFrame: Titanic dataset.
    """

    print("Loading Titanic dataset...")

    df = sns.load_dataset("titanic")

    return df


def dataset_overview(
    df: pd.DataFrame,
) -> None:
    """
    Display basic information about the dataset.

    Args:
        df (pd.DataFrame): Titanic dataframe.
    """

    print("\nDataset Shape")
    print("-" * 40)
    print(df.shape)

    print("\nDataset Information")
    print("-" * 40)
    df.info()

    print("\nStatistical Summary")
    print("-" * 40)
    print(df.describe(include="all"))


def save_raw_dataset(
    df: pd.DataFrame,
) -> None:
    """
    Save the raw Titanic dataset.

    Args:
        df (pd.DataFrame): Titanic dataframe.
    """

    file_path = RAW_DATA_DIR / "titanic.csv"

    df.to_csv(
        file_path,
        index=False,
    )

    print(f"\nRaw dataset saved to:\n{file_path}")


def main():
    """
    Execute dataset loading pipeline.
    """

    df = load_dataset()

    dataset_overview(df)

    save_raw_dataset(df)


if __name__ == "__main__":
    main()