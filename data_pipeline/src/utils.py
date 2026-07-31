"""
Utility functions for the Data Pipeline module.
"""

from pathlib import Path

import requests
from bs4 import BeautifulSoup

# -----------------------------
# Project Paths
# -----------------------------

MODULE_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = MODULE_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = MODULE_ROOT / "data" / "processed"
DATABASE_DIR = MODULE_ROOT / "data" / "database"
OUTPUTS_DIR = MODULE_ROOT / "outputs"
SQL_RESULTS_DIR = OUTPUTS_DIR / "sql_results"


RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
SQL_RESULTS_DIR.mkdir(parents=True,exist_ok=True)


# -----------------------------
# URLs
# -----------------------------

BASE_URL = "https://books.toscrape.com/"
CATALOGUE_URL = BASE_URL + "catalogue/"

# -----------------------------
# Constants
# -----------------------------

GBP_TO_INR = 105.50

RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
}


def get_soup(url: str) -> BeautifulSoup:
    """
    Download a webpage and return a BeautifulSoup object.
    """

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    return BeautifulSoup(response.text, "lxml")