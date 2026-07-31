import sqlite3

import pandas as pd

from utils import (
    PROCESSED_DATA_DIR,
    DATABASE_DIR,
)


def create_connection() -> sqlite3.Connection:
    """
    Create and return a SQLite database connection.

    Returns:
        sqlite3.Connection: SQLite connection object.
    """

    db_path = DATABASE_DIR / "books.db"

    connection = sqlite3.connect(db_path)

    return connection


def create_tables(connection: sqlite3.Connection) -> None:
    """
    Drop existing tables and create fresh normalized tables.

    Args:
        connection (sqlite3.Connection): SQLite connection.
    """

    cursor = connection.cursor()

    cursor.execute("DROP TABLE IF EXISTS books;")
    cursor.execute("DROP TABLE IF EXISTS categories;")

    cursor.execute(
        """
        CREATE TABLE categories (

            category_id INTEGER PRIMARY KEY AUTOINCREMENT,

            category_name TEXT UNIQUE NOT NULL

        );
        """
    )

    cursor.execute(
        """
        CREATE TABLE books (

            book_id INTEGER PRIMARY KEY AUTOINCREMENT,

            title TEXT NOT NULL,

            rating INTEGER NOT NULL,

            in_stock BOOLEAN NOT NULL,

            price_gbp REAL NOT NULL,

            price_inr REAL NOT NULL,

            category_id INTEGER NOT NULL,

            FOREIGN KEY(category_id)
            REFERENCES categories(category_id)

        );
        """
    )

    connection.commit()


def load_processed_data() -> pd.DataFrame:
    """
    Load the processed dataset.

    Returns:
        pd.DataFrame: Processed dataframe.
    """

    file_path = PROCESSED_DATA_DIR / "books_processed.csv"

    return pd.read_csv(
        file_path,
        encoding="utf-8",
    )


def insert_categories(
    connection: sqlite3.Connection,
    df: pd.DataFrame,
) -> None:
    """
    Insert unique categories into the categories table.

    Args:
        connection (sqlite3.Connection): SQLite connection.
        df (pd.DataFrame): Processed dataframe.
    """

    categories_df = (
        df[["category"]]
        .drop_duplicates()
        .sort_values("category")
        .reset_index(drop=True)
    )

    categories_df.rename(
        columns={"category": "category_name"},
        inplace=True,
    )

    categories_df.to_sql(
        "categories",
        connection,
        if_exists="append",
        index=False,
    )

    connection.commit()


def get_category_mapping(
    connection: sqlite3.Connection,
) -> dict:
    """
    Retrieve category name to category ID mapping.

    Args:
        connection (sqlite3.Connection): SQLite connection.

    Returns:
        dict: Category mapping dictionary.
    """

    query = """
    SELECT
        category_id,
        category_name
    FROM categories;
    """

    category_df = pd.read_sql(query, connection)

    return dict(
        zip(
            category_df["category_name"],
            category_df["category_id"],
        )
    )


def prepare_books_dataframe(
    df: pd.DataFrame,
    category_mapping: dict,
) -> pd.DataFrame:
    """
    Replace category names with category IDs.

    Args:
        df (pd.DataFrame): Processed dataframe.
        category_mapping (dict): Category lookup.

    Returns:
        pd.DataFrame: Updated dataframe.
    """

    books_df = df.copy()

    books_df["category_id"] = books_df["category"].map(
        category_mapping
    )

    books_df.drop(
        columns=["category"],
        inplace=True,
    )

    return books_df


def insert_books(
    connection: sqlite3.Connection,
    books_df: pd.DataFrame,
) -> None:
    """
    Insert books into the books table.

    Args:
        connection (sqlite3.Connection): SQLite connection.
        books_df (pd.DataFrame): Books dataframe.
    """

    books_df.to_sql(
        "books",
        connection,
        if_exists="append",
        index=False,
    )

    connection.commit()


def verify_data(connection: sqlite3.Connection) -> None:
    """
    Verify inserted data.

    Args:
        connection (sqlite3.Connection): SQLite connection.
    """

    cursor = connection.cursor()

    cursor.execute(
        "SELECT COUNT(*) FROM categories;"
    )

    total_categories = cursor.fetchone()[0]

    cursor.execute(
        "SELECT COUNT(*) FROM books;"
    )

    total_books = cursor.fetchone()[0]

    print("\nDatabase Verification")
    print("-" * 30)
    print(f"Categories : {total_categories}")
    print(f"Books      : {total_books}")


def close_connection(
    connection: sqlite3.Connection,
) -> None:
    """
    Close database connection.

    Args:
        connection (sqlite3.Connection): SQLite connection.
    """

    connection.close()


def main():
    """
    Execute the complete database loading pipeline.
    """

    print("Connecting to database...")

    connection = create_connection()

    print("Creating database tables...")

    create_tables(connection)

    print("Loading processed dataset...")

    df = load_processed_data()

    print("Inserting categories...")

    insert_categories(connection, df)

    category_mapping = get_category_mapping(connection)

    books_df = prepare_books_dataframe(
        df,
        category_mapping,
    )

    print("Inserting books...")

    insert_books(
        connection,
        books_df,
    )

    verify_data(connection)

    close_connection(connection)

    print("\nDatabase created successfully!")
    print("Database saved in data/database/books.db")


if __name__ == "__main__":
    main()