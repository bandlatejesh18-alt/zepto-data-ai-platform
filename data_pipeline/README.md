# 📚 Data Pipeline Module

## Overview

This module implements a complete ETL (Extract, Transform, Load) pipeline using the BooksToScrape website as a sample product catalog.

The goal is to demonstrate the complete data engineering workflow:

- Extract product information from a website
- Clean and transform the data
- Perform feature engineering
- Store the data in a normalized relational database
- Query the database using SQL
- Analyze the results using Pandas

Although the source dataset contains books instead of grocery products, the pipeline architecture is identical to what would be used for retail catalog data such as Zepto.

---

# ETL Pipeline

```
BooksToScrape Website
        │
        ▼
scraper.py
        │
        ▼
books_raw.csv
        │
        ▼
cleaner.py
        │
        ▼
books_processed.csv
        │
        ▼
database.py
        │
        ▼
SQLite Database
        │
        ▼
queries.py
        │
        ▼
SQL Results + Pandas Analysis
```

---

# Project Structure

```
data_pipeline/

│

├── data/
│   ├── raw/
│   │   └── books_raw.csv
│   │
│   ├── processed/
│   │   └── books_processed.csv
│   │
│   └── database/
│       └── books.db
│
├── outputs/
│   └── sql_results/
│
├── notebooks/
│
├── src/
│   ├── scraper.py
│   ├── cleaner.py
│   ├── database.py
│   ├── queries.py
│   ├── main.py
│   └── utils.py
│
└── README.md
```

---

# Technologies Used

- Python
- Requests
- BeautifulSoup
- Pandas
- SQLite3

---

# Dataset

Website:

https://books.toscrape.com/

Books Scraped:

- First 5 catalogue pages
- 100 books collected

Captured Fields:

- Title
- Price (GBP)
- Star Rating
- Availability
- Category

---

# Data Cleaning

The following preprocessing steps were performed.

### Price

Original

```
£51.77
```

Converted to

```
51.77
```

Stored as

```
price_gbp (float)
```

---

### Rating

Original

```
One
Two
Three
Four
Five
```

Converted to

```
1
2
3
4
5
```

Stored as

```
rating (integer)
```

---

### Availability

Original

```
In stock (22 available)
```

Converted to

```
True
```

Stored as

```
in_stock (boolean)
```

---

### Currency Conversion

Project-defined conversion rate

```
1 GBP = 105.50 INR
```

Used to create

```
price_inr
```

No external API was used.

---

### Missing Values

Numeric parsing failures are handled using median imputation.

Rows containing unrecoverable categorical parsing errors are removed to maintain dataset integrity.

---

# Database Design

The database follows a normalized relational design.

## Categories

| Column | Type |
|---------|------|
| category_id | INTEGER PRIMARY KEY |
| category_name | TEXT UNIQUE |

---

## Books

| Column | Type |
|---------|------|
| book_id | INTEGER PRIMARY KEY |
| title | TEXT |
| rating | INTEGER |
| in_stock | BOOLEAN |
| price_gbp | REAL |
| price_inr | REAL |
| category_id | INTEGER (Foreign Key) |

Relationship

```
Categories (1)
        │
        │
        ▼
Books (Many)
```

---

# SQL Queries

The following SQL concepts are demonstrated.

- SELECT
- WHERE
- ORDER BY
- LIMIT
- DISTINCT
- IN
- BETWEEN
- JOIN

All query outputs are saved inside

```
outputs/sql_results/
```

---

# Pandas Analysis

The module demonstrates

- pd.read_sql()
- DataFrame operations
- pd.merge()

The SQL JOIN result is reproduced using Pandas merge and both outputs are verified to be identical.

---

# How to Run

Install dependencies

```bash
pip install -r requirements.txt
```

Run the complete ETL pipeline

```bash
python src/main.py
```

---

# Output Files

Raw Dataset

```
data/raw/books_raw.csv
```

Processed Dataset

```
data/processed/books_processed.csv
```

SQLite Database

```
data/database/books.db
```

SQL Results

```
outputs/sql_results/
```

---

# Design Decisions

- Scraped the first five catalogue pages to obtain 100 books.
- Used a fixed exchange rate of 1 GBP = 105.50 INR as specified in the assignment.
- Implemented a normalized SQLite schema using separate Books and Categories tables.
- Used Pandas for efficient CSV processing and SQL interoperability.
- Saved SQL query outputs as CSV files for reproducibility.

---

# Future Improvements

- Add logging
- Add automated unit tests
- Schedule incremental scraping
- Support multiple e-commerce websites
- Deploy as an automated ETL workflow