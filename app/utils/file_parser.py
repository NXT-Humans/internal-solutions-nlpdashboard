"""
file_parser.py

Walks through the unzipped folder, parses each file type, 
and returns a dictionary of {filename: [paragraphs]}.
"""

import os
import json
from app.utils.text_extractor import extract_text_from_file

def parse_zip_file(temp_folder: str, logs_ref: dict) -> dict:
    logs_ref["steps"].append("Parsing extracted files.")
    file_texts = {}

    for root, dirs, files in os.walk(temp_folder):
        for filename in files:
            if filename.endswith(".zip"):
                continue
            file_path = os.path.join(root, filename)
            logs_ref["steps"].append(f"Extracting text from: {filename}")
            paragraphs = extract_text_from_file(file_path, logs_ref)
            if paragraphs:
                file_texts[filename] = paragraphs

    logs_ref["steps"].append("Finished parsing unzipped files.")
    return file_texts
