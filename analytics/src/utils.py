from pathlib import Path

# --------------------------------------------------
# Project Root
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------
# Data Directories
# --------------------------------------------------

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


# --------------------------------------------------
# Output Directories
# --------------------------------------------------

PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"

TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"


# --------------------------------------------------
# Model Directory
# --------------------------------------------------

MODELS_DIR = PROJECT_ROOT / "models"


# --------------------------------------------------
# Automatically Create Directories
# --------------------------------------------------

RAW_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PROCESSED_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PLOTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TABLES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)