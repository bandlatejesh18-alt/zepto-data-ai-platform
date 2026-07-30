import pandas as pd

from utils import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    RATING_MAP,
    GBP_TO_INR,
)


def load_raw_data() -> pd.DataFrame:
    """
    Load the raw scraped dataset.

    Returns:
        pd.DataFrame: Raw books dataset.
    """
    file_path = RAW_DATA_DIR / "books_raw.csv"

    return pd.read_csv(
        file_path,
        encoding="utf-8",
    )


def clean_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the price column and convert it to float.

    Args:
        df (pd.DataFrame): Raw dataframe.

    Returns:
        pd.DataFrame: Updated dataframe.
    """

    df["price_gbp"] = (
        df["price"]
        .str.replace("Â£", "", regex=False)
        .astype(float)
    )

    df.drop(columns=["price"], inplace=True)

    return df


def clean_rating(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert star ratings to numeric values.

    Args:
        df (pd.DataFrame): Dataframe.

    Returns:
        pd.DataFrame: Updated dataframe.
    """

    df["rating"] = df["star_rating"].map(RATING_MAP)

    df.drop(columns=["star_rating"], inplace=True)

    return df


def clean_availability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert availability text into boolean values.

    Args:
        df (pd.DataFrame): Dataframe.

    Returns:
        pd.DataFrame: Updated dataframe.
    """

    df["in_stock"] = df["availability"].str.contains(
        "In stock",
        case=False,
        na=False,
    )

    df.drop(columns=["availability"], inplace=True)

    return df


def add_price_inr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add INR price column.

    Args:
        df (pd.DataFrame): Dataframe.

    Returns:
        pd.DataFrame: Updated dataframe.
    """

    df["price_inr"] = (
        df["price_gbp"] * GBP_TO_INR
    ).round(2)

    return df


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder dataframe columns for better readability.

    Returns:
        pd.DataFrame: Reordered dataframe.
    """

    column_order = [
        "title",
        "category",
        "rating",
        "in_stock",
        "price_gbp",
        "price_inr",
    ]

    return df[column_order]


def validate_dataset(df: pd.DataFrame) -> None:
    """
    Print basic data quality checks.

    Args:
        df (pd.DataFrame): Cleaned dataframe.
    """

    print("\nData Validation")
    print("-" * 30)

    print(f"Rows           : {len(df)}")
    print(f"Columns        : {len(df.columns)}")
    print(f"Duplicate Rows : {df.duplicated().sum()}")

    print("\nMissing Values")
    print(df.isnull().sum())


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform the complete cleaning pipeline.

    Args:
        df (pd.DataFrame): Raw dataframe.

    Returns:
        pd.DataFrame: Cleaned dataframe.
    """

    df = clean_price(df)
    df = clean_rating(df)
    df = clean_availability(df)
    df = add_price_inr(df)
    df = reorder_columns(df)

    return df


def main():
    """
    Execute the complete data cleaning pipeline.
    """

    print("Loading raw dataset...")

    df = load_raw_data()

    print("Cleaning dataset...")

    df = clean_dataset(df)

    validate_dataset(df)

    output_path = (
        PROCESSED_DATA_DIR / "books_processed.csv"
    )

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    print("\nCleaning completed successfully!")
    print(f"Books processed : {len(df)}")
    print(f"Dataset saved to: {output_path}")


if __name__ == "__main__":
    main()