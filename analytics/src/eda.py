import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import StandardScaler

from utils import (
    PROCESSED_DATA_DIR,
    PLOTS_DIR,
)


def load_cleaned_dataset() -> pd.DataFrame:
    """
    Load the cleaned Titanic dataset.

    Returns:
        pd.DataFrame: Cleaned dataframe.
    """

    file_path = PROCESSED_DATA_DIR / "titanic_cleaned.csv"

    return pd.read_csv(file_path)


def dataset_summary(
    df: pd.DataFrame,
) -> None:
    """
    Display dataset summary.

    Args:
        df (pd.DataFrame): Cleaned dataframe.
    """

    print("\nDataset Shape")
    print("-" * 50)
    print(df.shape)

    print("\nDataset Information")
    print("-" * 50)
    df.info()

    print("\nStatistical Summary")
    print("-" * 50)
    print(df.describe(include="all"))


def plot_histograms(
    df: pd.DataFrame,
) -> None:
    """
    Plot histograms for Age and Fare.

    Args:
        df (pd.DataFrame): Cleaned dataframe.
    """

    columns = [
        "age",
        "fare",
    ]

    for column in columns:

        plt.figure(figsize=(8, 5))

        sns.histplot(
            data=df,
            x=column,
            kde=True,
        )

        plt.title(f"{column.title()} Distribution")

        plt.xlabel(column.title())

        plt.ylabel("Frequency")

        plt.tight_layout()

        plt.savefig(
            PLOTS_DIR /
            f"{column}_histogram.png"
        )

        plt.close()


def plot_boxplots(
    df: pd.DataFrame,
) -> None:
    """
    Plot boxplots for Age and Fare.

    Args:
        df (pd.DataFrame): Cleaned dataframe.
    """

    columns = [
        "age",
        "fare",
    ]

    for column in columns:

        plt.figure(figsize=(8, 5))

        sns.boxplot(
            data=df,
            x=column,
        )

        plt.title(f"{column.title()} Box Plot")

        plt.tight_layout()

        plt.savefig(
            PLOTS_DIR /
            f"{column}_boxplot.png"
        )

        plt.close()


def detect_outliers(
    df: pd.DataFrame,
) -> None:
    """
    Detect outliers using the IQR rule.

    Args:
        df (pd.DataFrame): Cleaned dataframe.
    """

    columns = [
        "age",
        "fare",
    ]

    print("\nOutlier Report")
    print("-" * 50)

    for column in columns:

        q1 = df[column].quantile(0.25)

        q3 = df[column].quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - (1.5 * iqr)

        upper_bound = q3 + (1.5 * iqr)

        outliers = df[
            (df[column] < lower_bound)
            |
            (df[column] > upper_bound)
        ]

        print(f"{column.title()} Outliers : {len(outliers)}")


def fare_statistics(
    df: pd.DataFrame,
) -> None:
    """
    Compute mean, median and mode for Fare.

    Args:
        df (pd.DataFrame): Cleaned dataframe.
    """

    mean = df["fare"].mean()

    median = df["fare"].median()

    mode = df["fare"].mode()[0]

    print("\nFare Statistics")
    print("-" * 50)

    print(f"Mean   : {mean:.2f}")

    print(f"Median : {median:.2f}")

    print(f"Mode   : {mode:.2f}")

    if mean > median > mode:

        skewness = "Right Skewed"

    elif mean < median < mode:

        skewness = "Left Skewed"

    else:

        skewness = "Approximately Symmetric"

    print(f"Distribution : {skewness}")


def survival_by_sex(
    df: pd.DataFrame,
) -> None:
    """
    Compute survival rate by sex.

    Args:
        df (pd.DataFrame): Cleaned dataframe.
    """

    print("\nSurvival Rate by Sex")
    print("-" * 50)

    for gender in df["sex"].unique():

        total = df[
            df["sex"] == gender
        ]

        survived = df[
            (df["sex"] == gender)
            &
            (df["survived"] == 1)
        ]

        survival_rate = (
            len(survived) / len(total)
        ) * 100

        print(
            f"{gender:<10}: {survival_rate:.2f}%"
        )

    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=df,
        x="sex",
        y="survived",
        estimator="mean",
        errorbar=None,
    )

    plt.title("Survival Rate by Sex")

    plt.ylabel("Survival Rate")

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR /
        "survival_by_sex.png"
    )

    plt.close()


def survival_by_pclass(
    df: pd.DataFrame,
) -> None:
    """
    Compute survival rate by passenger class.

    Args:
        df (pd.DataFrame): Cleaned dataframe.
    """

    print("\nSurvival Rate by Passenger Class")
    print("-" * 50)

    for passenger_class in sorted(
        df["pclass"].unique()
    ):

        total = df[
            df["pclass"] == passenger_class
        ]

        survived = df[
            (df["pclass"] == passenger_class)
            &
            (df["survived"] == 1)
        ]

        survival_rate = (
            len(survived) / len(total)
        ) * 100

        print(
            f"Class {passenger_class}: "
            f"{survival_rate:.2f}%"
        )

    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=df,
        x="pclass",
        y="survived",
        estimator="mean",
        errorbar=None,
    )

    plt.title("Survival Rate by Passenger Class")

    plt.ylabel("Survival Rate")

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR /
        "survival_by_pclass.png"
    )

    plt.close()


def survival_by_sex_and_pclass(
    df: pd.DataFrame,
) -> None:
    """
    Compute survival rate by
    sex and passenger class.

    Args:
        df (pd.DataFrame): Cleaned dataframe.
    """

    print("\nSurvival Rate by Sex and Passenger Class")
    print("-" * 50)

    grouped = (
        df.groupby(
            ["sex", "pclass"]
        )["survived"]
        .mean()
        * 100
    )

    print(grouped)

    plt.figure(figsize=(9, 6))

    sns.barplot(
        data=df,
        x="pclass",
        y="survived",
        hue="sex",
        estimator="mean",
        errorbar=None,
    )

    plt.title(
        "Survival Rate by Sex and Passenger Class"
    )

    plt.ylabel("Survival Rate")

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR /
        "survival_by_sex_pclass.png"
    )

    plt.close()


def correlation_analysis(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute correlation matrix.

    Args:
        df (pd.DataFrame): Cleaned dataframe.

    Returns:
        pd.DataFrame: Correlation matrix.
    """

    columns = [
        "survived",
        "pclass",
        "age",
        "sibsp",
        "parch",
        "fare",
    ]

    correlation_matrix = (
        df[columns]
        .corr()
    )

    print("\nCorrelation Matrix")
    print("-" * 50)

    print(correlation_matrix)

    return correlation_matrix


def plot_heatmap(
    correlation_matrix: pd.DataFrame,
) -> None:
    """
    Plot correlation heatmap.

    Args:
        correlation_matrix (pd.DataFrame):
            Correlation matrix.
    """

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        linewidths=0.5,
        fmt=".2f",
    )

    plt.title(
        "Correlation Heatmap"
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR /
        "correlation_heatmap.png"
    )

    plt.close()


def multivariate_analysis(
    df: pd.DataFrame,
) -> None:
    """
    Create multivariate visualizations.

    Args:
        df (pd.DataFrame): Cleaned dataframe.
    """

    # --------------------------------------------------
    # Chart 1
    # Survival by Sex
    # --------------------------------------------------

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x="sex",
        hue="survived",
    )

    plt.title("Survival by Sex")

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR /
        "multivariate_chart_1.png"
    )

    plt.close()

    # --------------------------------------------------
    # Chart 2
    # Survival by Passenger Class
    # --------------------------------------------------

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x="pclass",
        hue="survived",
    )

    plt.title("Survival by Passenger Class")

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR /
        "multivariate_chart_2.png"
    )

    plt.close()

    # --------------------------------------------------
    # Chart 3
    # Age vs Fare
    # --------------------------------------------------

    plt.figure(figsize=(8, 6))

    sns.scatterplot(
        data=df,
        x="age",
        y="fare",
        hue="survived",
    )

    plt.title("Age vs Fare")

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR /
        "multivariate_chart_3.png"
    )

    plt.close()

    # --------------------------------------------------
    # Chart 4
    # Age Distribution by Sex and Survival
    # --------------------------------------------------

    plt.figure(figsize=(8, 6))

    sns.boxplot(
        data=df,
        x="sex",
        y="age",
        hue="survived",
    )

    plt.title(
        "Age Distribution by Sex and Survival"
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR /
        "multivariate_chart_4.png"
    )

    plt.close()


def standardization_check(
    df: pd.DataFrame,
) -> None:
    """
    Standardize Age and Fare using Z-score normalization.

    This is only an exploratory data analysis (EDA) check.
    The standardized values are NOT used in the modeling
    pipeline.

    Args:
        df (pd.DataFrame): Cleaned dataframe.
    """

    scaler = StandardScaler()

    standardized_values = scaler.fit_transform(
        df[
            [
                "age",
                "fare",
            ]
        ]
    )

    standardized_df = pd.DataFrame(
        standardized_values,
        columns=[
            "age",
            "fare",
        ],
    )

    # --------------------------------------------------
    # Before / After Statistics
    # --------------------------------------------------

    print("\nStandardization Check")
    print("-" * 60)

    print("\nBefore Standardization")
    print(
        df[
            [
                "age",
                "fare",
            ]
        ].agg(
            [
                "mean",
                "std",
            ]
        )
    )

    print("\nAfter Standardization")
    print(
        standardized_df.agg(
            [
                "mean",
                "std",
            ]
        )
    )

    # --------------------------------------------------
    # Age Distribution
    # --------------------------------------------------

    plt.figure(figsize=(8, 5))

    sns.histplot(
        df["age"],
        kde=True,
        stat="density",
        label="Before",
        alpha=0.5,
    )

    sns.histplot(
        standardized_df["age"],
        kde=True,
        stat="density",
        label="After",
        alpha=0.5,
    )

    plt.title(
        "Age Distribution: Before vs After Standardization"
    )

    plt.xlabel("Age")

    plt.ylabel("Density")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR /
        "age_standardization_comparison.png"
    )

    plt.close()

    # --------------------------------------------------
    # Fare Distribution
    # --------------------------------------------------

    plt.figure(figsize=(8, 5))

    sns.histplot(
        df["fare"],
        kde=True,
        stat="density",
        label="Before",
        alpha=0.5,
    )

    sns.histplot(
        standardized_df["fare"],
        kde=True,
        stat="density",
        label="After",
        alpha=0.5,
    )

    plt.title(
        "Fare Distribution: Before vs After Standardization"
    )

    plt.xlabel("Fare")

    plt.ylabel("Density")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR /
        "fare_standardization_comparison.png"
    )

    plt.close()


def main():
    """
    Execute the complete
    exploratory data analysis pipeline.
    """

    print("Loading cleaned dataset...")

    df = load_cleaned_dataset()

    dataset_summary(df)

    print("\nGenerating Histograms...")

    plot_histograms(df)

    print("Generating Boxplots...")

    plot_boxplots(df)

    detect_outliers(df)

    fare_statistics(df)

    survival_by_sex(df)

    survival_by_pclass(df)

    survival_by_sex_and_pclass(df)

    correlation_matrix = correlation_analysis(df)

    plot_heatmap(
        correlation_matrix
    )

    multivariate_analysis(df)

    standardization_check(df)

    print("\nEDA Completed Successfully!")

    print(
        f"\nPlots saved to:\n{PLOTS_DIR}"
    )


if __name__ == "__main__":
    main()