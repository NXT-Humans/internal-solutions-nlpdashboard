"""
data_cleaner.py

Applies a basic cleaning to each paragraph (trimming whitespace, 
removing extra spaces). Could be extended for lemmatization or spaCy usage.
"""

import re
from typing import List

class DataCleaner:
    def __init__(self, logs_ref):
        self.logs = logs_ref

    def clean_paragraphs(self, paragraphs: List[str]) -> List[str]:
        cleaned = []
        for p in paragraphs:
            text = p.strip()
            text = re.sub(r"\s+", " ", text)
            if text:
                cleaned.append(text)
        return cleaned