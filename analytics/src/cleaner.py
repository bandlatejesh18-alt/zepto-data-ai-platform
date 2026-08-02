import pandas as pd

from utils import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
)


def load_raw_dataset() -> pd.DataFrame:
    """
    Load the raw Titanic dataset.

    Returns:
        pd.DataFrame: Raw Titanic dataframe.
    """

    file_path = RAW_DATA_DIR / "titanic.csv"

    return pd.read_csv(file_path)


def calculate_missing_percentage(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate the missing percentage for every column
    containing missing values.

    Args:
        df (pd.DataFrame): Raw dataframe.

    Returns:
        pd.DataFrame: Missing value report.
    """

    missing_values = df.isnull().sum()

    missing_percentage = (
        missing_values / len(df)
    ) * 100

    report = pd.DataFrame(
        {
            "Missing Values": missing_values,
            "Missing Percentage": missing_percentage,
        }
    )

    report = report[
        report["Missing Values"] > 0
    ]

    report = report.sort_values(
        by="Missing Percentage",
        ascending=False,
    )

    print("\nMissing Value Report")
    print("-" * 60)
    print(report.round(2))

    return report


def handle_missing_values(
    df: pd.DataFrame,
    report: pd.DataFrame,
) -> pd.DataFrame:
    """
    Handle missing values according to the assignment.

    Rules
    -----
    < 5%
        Drop rows.

    5% - 30%
        Impute values.

    > 30%
        Drop the column.

    Args:
        df (pd.DataFrame): Raw dataframe.
        report (pd.DataFrame): Missing value report.

    Returns:
        pd.DataFrame: Cleaned dataframe.
    """

    for column in report.index:

        percentage = float(report.loc[
            column,
            "Missing Percentage",
        ])

        print(
            f"\nColumn : {column}"
        )

        print(
            f"Missing Percentage : {percentage:.2f}%"
        )

        # --------------------------------------------------
        # Less than 5%
        # --------------------------------------------------

        if percentage < 5:

            print(
                "Strategy : Drop Rows"
            )

            df = df.dropna(
                subset=[column]
            )

        # --------------------------------------------------
        # Between 5% and 30%
        # --------------------------------------------------

        elif percentage <= 30:

            if pd.api.types.is_numeric_dtype(
                df[column]
            ):

                print(
                    "Strategy : Median Imputation"
                )

                df[column] = df[column].fillna(
                    df[column].median()
                )

            else:

                print(
                    "Strategy : Mode Imputation"
                )

                df[column] = df[column].fillna(
                    df[column].mode()[0]
                )

        # --------------------------------------------------
        # Greater than 30%
        # --------------------------------------------------

        else:

            print(
                "Strategy : Drop Column"
            )

            df = df.drop(
                columns=[column]
            )

    return df


def save_processed_dataset(
    df: pd.DataFrame,
) -> None:
    """
    Save the cleaned dataset.

    Args:
        df (pd.DataFrame): Cleaned dataframe.
    """

    file_path = (
        PROCESSED_DATA_DIR /
        "titanic_cleaned.csv"
    )

    df.to_csv(
        file_path,
        index=False,
    )

    print(
        f"\nProcessed dataset saved to:\n{file_path}"
    )


def main():
    """
    Execute the complete cleaning pipeline.
    """

    print("Loading raw dataset...")

    df = load_raw_dataset()

    report = calculate_missing_percentage(
        df
    )

    print("\nHandling missing values...")

    cleaned_df = handle_missing_values(
        df,
        report,
    )

    save_processed_dataset(
        cleaned_df
    )

    print("\nCleaning Completed Successfully")
    print("-" * 50)
    print(
        f"Rows    : {cleaned_df.shape[0]}"
    )
    print(
        f"Columns : {cleaned_df.shape[1]}"
    )


if __name__ == "__main__":
    main()