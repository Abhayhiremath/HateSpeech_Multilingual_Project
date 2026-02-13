# streamlit_app/utils/text_cleaner.py

import os
import sys

# ---------------------------------------------------
# FIX: Add project root to Python path automatically
# ---------------------------------------------------

# current file: streamlit_app/utils/text_cleaner.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))          
STREAMLIT_APP_DIR = os.path.dirname(CURRENT_DIR)                  # streamlit_app/
PROJECT_ROOT = os.path.dirname(STREAMLIT_APP_DIR)                 # main project folder

# add root to Python path
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# ---------------------------------------------------
# Import clean_text function from preprocessing folder
# ---------------------------------------------------
try:
    from preprocessing.clean_text import basic_clean
except Exception as e:
    print("❌ Could not import basic_clean:", e)
    print("🔍 Checked PATHS:", sys.path)
    raise e

# ---------------------------------------------------
# Streamlit cleaner wrapper
# ---------------------------------------------------
def app_clean(s: str) -> str:
    return basic_clean(str(s))
