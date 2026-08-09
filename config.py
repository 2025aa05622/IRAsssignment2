"""
config.py

Central configuration file for the End-to-End Information Retrieval System.
Modify values here instead of changing them throughout the project.
"""

from pathlib import Path

# -------------------------------------------------------
# Project Directories
# -------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

INDEX_DIR = BASE_DIR / "index"
DATABASE_DIR = BASE_DIR / "database"
ASSETS_DIR = BASE_DIR / "assets"
REPORT_DIR = BASE_DIR / "report"

# -------------------------------------------------------
# Database
# -------------------------------------------------------

DATABASE_NAME = "ir.db"
DATABASE_PATH = DATABASE_DIR / DATABASE_NAME

# -------------------------------------------------------
# Crawling Configuration
# -------------------------------------------------------

DEFAULT_MAX_DEPTH = 2
DEFAULT_MAX_PAGES = 100
REQUEST_TIMEOUT = 50
USER_AGENT = (
    "Mozilla/5.0 "
    "(Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 "
    "(KHTML, like Gecko) "
    "Chrome/120.0 Safari/537.36"
)

# -------------------------------------------------------
# Text Processing
# -------------------------------------------------------

DEFAULT_LANGUAGE = "english"

REMOVE_STOPWORDS = True
ENABLE_STEMMING = True
ENABLE_LEMMATIZATION = True

MIN_WORD_LENGTH = 2

# -------------------------------------------------------
# Search Configuration
# -------------------------------------------------------

DEFAULT_TOP_K = 10

FUZZY_MATCH_THRESHOLD = 80

# -------------------------------------------------------
# Recommendation
# -------------------------------------------------------

TOP_RECOMMENDATIONS = 10

# -------------------------------------------------------
# Evaluation
# -------------------------------------------------------

DEFAULT_K = 10

# -------------------------------------------------------
# Visualization
# -------------------------------------------------------

MAX_WORDCLOUD_WORDS = 200

TOP_TERMS_CHART = 20

# -------------------------------------------------------
# Logging
# -------------------------------------------------------

LOG_LEVEL = "INFO"

# -------------------------------------------------------
# Create directories automatically
# -------------------------------------------------------

DIRECTORIES = [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    INDEX_DIR,
    DATABASE_DIR,
    ASSETS_DIR,
    REPORT_DIR,
]

for directory in DIRECTORIES:
    directory.mkdir(parents=True, exist_ok=True)
