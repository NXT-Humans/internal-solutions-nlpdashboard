"""
config.py

Loads environment variables and defines application settings.
"""

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    MODEL_PATH = os.getenv("MODEL_PATH", "models/bert-base-uncased")
