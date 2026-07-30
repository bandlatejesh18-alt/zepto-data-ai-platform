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


def create_table(connection: sqlite3.Connection) -> None:
    """
    Drop the existing books table (if any) and create a fresh one.

    Args:
        connection (sqlite3.Connection): SQLite connection.
    """

    cursor = connection.cursor()

    cursor.execute(
        """
        DROP TABLE IF EXISTS books;
        """
    )

    cursor.execute(
        """
        CREATE TABLE books (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            title TEXT NOT NULL,

            category TEXT NOT NULL,

            rating INTEGER NOT NULL,

            in_stock BOOLEAN NOT NULL,

            price_gbp REAL NOT NULL,

            price_inr REAL NOT NULL

        );
        """
    )

    connection.commit()


def load_processed_data() -> pd.DataFrame:
    """
    Load the cleaned dataset.

    Returns:
        pd.DataFrame: Processed dataset.
    """

    file_path = PROCESSED_DATA_DIR / "books_processed.csv"

    return pd.read_csv(
        file_path,
        encoding="utf-8",
    )


def insert_books(
    connection: sqlite3.Connection,
    df: pd.DataFrame,
) -> None:
    """
    Insert all books into the database.

    Args:
        connection (sqlite3.Connection): SQLite connection.
        df (pd.DataFrame): Processed dataframe.
    """

    df.to_sql(
        name="books",
        con=connection,
        if_exists="append",
        index=False,
    )

    connection.commit()


def verify_data(connection: sqlite3.Connection) -> None:
    """
    Verify that data was inserted successfully.

    Args:
        connection (sqlite3.Connection): SQLite connection.
    """

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM books;
        """
    )

    total_books = cursor.fetchone()[0]

    print("\nDatabase Verification")
    print("-" * 30)
    print(f"Total Books : {total_books}")


def close_connection(connection: sqlite3.Connection) -> None:
    """
    Close the SQLite connection.

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

    print("Creating books table...")

    create_table(connection)

    print("Loading processed dataset...")

    df = load_processed_data()

    print("Inserting records into database...")

    insert_books(connection, df)

    verify_data(connection)

    close_connection(connection)

    print("\nDatabase created successfully!")
    print("Database saved in data/database/books.db")


if __name__ == "__main__":
    main()