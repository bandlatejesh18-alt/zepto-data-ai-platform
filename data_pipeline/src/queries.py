import sqlite3

import pandas as pd

from utils import (
    DATABASE_DIR,
    SQL_RESULTS_DIR,
)


def create_connection() -> sqlite3.Connection:
    """
    Create and return a SQLite database connection.

    Returns:
        sqlite3.Connection: SQLite connection object.
    """

    db_path = DATABASE_DIR / "books.db"

    return sqlite3.connect(db_path)


def execute_query(
    connection: sqlite3.Connection,
    query: str,
    output_file: str,
    title: str,
) -> pd.DataFrame:
    """
    Execute a SQL query, print the result, and save it as a CSV.

    Args:
        connection (sqlite3.Connection): SQLite connection.
        query (str): SQL query.
        output_file (str): Output CSV filename.
        title (str): Query title.

    Returns:
        pd.DataFrame: Query result.
    """

    print(f"\n{title}")
    print("-" * len(title))

    df = pd.read_sql(query, connection)

    print(df)

    output_path = SQL_RESULTS_DIR / output_file

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    print(f"\nSaved: {output_path}")

    return df


def main():
    """
    Execute all required SQL queries.
    """

    connection = create_connection()

    # --------------------------------------------------
    # Query 1
    # SELECT + WHERE
    # --------------------------------------------------

    query_1 = """
    SELECT
        title,
        price_gbp
    FROM books
    WHERE price_gbp > 50;
    """

    execute_query(
        connection,
        query_1,
        "query_1_price_above_50.csv",
        "Query 1 - Books costing more than £50",
    )

    # --------------------------------------------------
    # Query 2
    # ORDER BY + LIMIT
    # --------------------------------------------------

    query_2 = """
    SELECT
        title,
        price_gbp
    FROM books
    ORDER BY price_gbp DESC
    LIMIT 10;
    """

    execute_query(
        connection,
        query_2,
        "query_2_top_10_expensive.csv",
        "Query 2 - Top 10 Most Expensive Books",
    )

    # --------------------------------------------------
    # Query 3
    # DISTINCT
    # --------------------------------------------------

    query_3 = """
    SELECT DISTINCT
        category_name
    FROM categories
    ORDER BY category_name;
    """

    execute_query(
        connection,
        query_3,
        "query_3_categories.csv",
        "Query 3 - Distinct Categories",
    )

    # --------------------------------------------------
    # Query 4
    # IN
    # --------------------------------------------------

    query_4 = """
    SELECT
        title,
        rating
    FROM books
    WHERE rating IN (4,5)
    ORDER BY rating DESC;
    """

    execute_query(
        connection,
        query_4,
        "query_4_high_rating.csv",
        "Query 4 - Books Rated 4 or 5",
    )

    # --------------------------------------------------
    # Query 5
    # BETWEEN
    # --------------------------------------------------

    query_5 = """
    SELECT
        title,
        price_gbp
    FROM books
    WHERE price_gbp BETWEEN 20 AND 30;
    """

    execute_query(
        connection,
        query_5,
        "query_5_price_range.csv",
        "Query 5 - Books Between £20 and £30",
    )

    # --------------------------------------------------
    # Query 6
    # JOIN
    # --------------------------------------------------

    query_6 = """
    SELECT

        b.title,

        c.category_name,

        b.rating,

        b.price_gbp

    FROM books b

    JOIN categories c

    ON b.category_id = c.category_id

    ORDER BY
        b.rating DESC,
        b.price_gbp DESC

    LIMIT 10;
    """

    join_df = execute_query(
        connection,
        query_6,
        "query_6_join.csv",
        "Query 6 - Top Rated Books with Categories",
    )

    # --------------------------------------------------
    # Requirement 6
    # Read SQL results into pandas
    # --------------------------------------------------

    print("\nReading SQL Results into Pandas...")
    print("-" * 35)

    # Read Query 2 into a DataFrame
    top_books_df = pd.read_sql(
        query_2,
        connection,
    )

    # Read Query 4 into a DataFrame
    high_rating_df = pd.read_sql(
        query_4,
        connection,
    )

    # Read base tables for pandas merge
    books_df = pd.read_sql(
        "SELECT * FROM books;",
        connection,
    )

    categories_df = pd.read_sql(
        "SELECT * FROM categories;",
        connection,
    )

    print(f"Top Books Shape        : {top_books_df.shape}")
    print(f"High Rating Shape      : {high_rating_df.shape}")
    print(f"Books Shape            : {books_df.shape}")
    print(f"Categories Shape       : {categories_df.shape}")

    print("\nTop Books DataFrame")
    print("-" * 25)
    print(top_books_df.head())

    print("\nHigh Rating Books DataFrame")
    print("-" * 30)
    print(high_rating_df.head())

    # --------------------------------------------------
    # Requirement 6
    # Reproduce JOIN using Pandas Merge
    # --------------------------------------------------

    merged_df = books_df.merge(
        categories_df,
        on="category_id",
        how="inner",
    )

    merged_df = merged_df[
        [
            "title",
            "category_name",
            "rating",
            "price_gbp",
        ]
    ]

    merged_df = merged_df.sort_values(
        by=["rating", "price_gbp"],
        ascending=[False, False],
    ).head(10)

    print("\nPandas Merge Result")
    print("-" * 25)
    print(merged_df)

    merged_df.to_csv(
        SQL_RESULTS_DIR / "query_6_pandas_merge.csv",
        index=False,
    )

    # --------------------------------------------------
    # Compare SQL JOIN vs Pandas Merge
    # --------------------------------------------------

    print("\nChecking SQL JOIN vs Pandas Merge...")
    print("-" * 35)

    if join_df.reset_index(drop=True).equals(
        merged_df.reset_index(drop=True)
    ):
        print("✓ SQL JOIN and Pandas Merge produce identical results.")
    else:
        print("✗ SQL JOIN and Pandas Merge results do not match.")

    connection.close()

    print("\nAll query results saved successfully!")


if __name__ == "__main__":
    main()