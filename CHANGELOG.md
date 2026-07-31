# Changelog

All notable changes to this project will be documented in this file.

---

## [0.1.0] - Data Pipeline Module Completed

### Added

- Created the initial project structure.
- Implemented a complete ETL pipeline.
- Scraped 100 books from BooksToScrape using Requests and BeautifulSoup.
- Cleaned and transformed the scraped data.
- Converted prices from GBP to INR using the fixed project conversion rate.
- Designed a normalized SQLite database with `books` and `categories` tables.
- Implemented SQL queries demonstrating SELECT, WHERE, ORDER BY, LIMIT, DISTINCT, IN, BETWEEN, and JOIN.
- Demonstrated SQL queries using `pd.read_sql()`.
- Reproduced SQL JOIN results using `pd.merge()` and verified equivalent outputs.
- Added a single pipeline entry point (`main.py`).
- Added module documentation.

---