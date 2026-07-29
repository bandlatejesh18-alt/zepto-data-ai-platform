"""
Scraper for BooksToScrape.

This module extracts book information from the first
five catalogue pages and stores the raw dataset.
"""

from urllib.parse import urljoin

import pandas as pd

from utils import (
    BASE_URL,
    CATALOGUE_URL,
    RAW_DATA_DIR,
    get_soup,
)


def extract_category(book_url: str) -> str:
    """
    Extract the category from an individual book page.

    Args:
        book_url (str): URL of the individual book page.

    Returns:
        str: Category name (e.g., Travel, Mystery).
    """

    soup = get_soup(book_url)

    breadcrumb = soup.select("ul.breadcrumb li a")

    if len(breadcrumb) >= 3:
        return breadcrumb[2].text.strip()

    return "Unknown"


def extract_book_data(book, page_url: str) -> dict:
    """
    Extract information from a single book card.

    Args:
        book: BeautifulSoup Tag representing one book.
        page_url (str): URL of the listing page containing the book.

    Returns:
        dict: Dictionary containing raw book information.
    """

    title = book.h3.a["title"]

    price = book.select_one(".price_color").text.strip()

    availability = (
        book.select_one(".instock.availability")
        .text.strip()
    )

    star_rating = book.p["class"][1]

    relative_url = book.h3.a["href"]

    # Resolve relative URL against the current page URL
    book_url = urljoin(page_url, relative_url)

    category = extract_category(book_url)

    return {
        "title": title,
        "price": price,
        "star_rating": star_rating,
        "availability": availability,
        "category": category,
    }


def scrape_listing_page(page_number: int) -> list:
    """
    Scrape one catalogue page.

    Args:
        page_number (int): Catalogue page number.

    Returns:
        list: List of dictionaries containing raw book data.
    """

    if page_number == 1:
        page_url = BASE_URL
    else:
        page_url = urljoin(
            BASE_URL,
            f"catalogue/page-{page_number}.html"
        )

    soup = get_soup(page_url)

    books = []

    for book in soup.select("article.product_pod"):
        books.append(
            extract_book_data(book, page_url)
        )

    return books


def scrape_books(num_pages: int = 5) -> pd.DataFrame:
    """
    Scrape multiple catalogue pages.

    Args:
        num_pages (int): Number of pages to scrape.

    Returns:
        pd.DataFrame: Raw books dataset.
    """

    all_books = []

    for page in range(1, num_pages + 1):

        print(f"Scraping page {page}...")

        all_books.extend(
            scrape_listing_page(page)
        )

    return pd.DataFrame(all_books)


def main():
    """
    Run the complete scraping pipeline.
    """

    books_df = scrape_books(num_pages=5)

    output_file = RAW_DATA_DIR / "books_raw.csv"

    books_df.to_csv(
        output_file,
        index=False,
    )

    print("\nScraping completed successfully!")

    print(f"Books scraped: {len(books_df)}")

    print(f"Dataset saved to: {output_file}")


if __name__ == "__main__":
    main()