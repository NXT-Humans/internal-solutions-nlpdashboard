"""
text_extractor.py

Extract text from PDFs, DOCX, TXT, CSV, JSON and maintain a text cache.
"""

import os
import json
import fitz  # PyMuPDF
import pandas as pd
from docx import Document
from typing import List, Dict
import pickle

def load_text_cache(temp_folder: str, logs_ref: Dict) -> Dict[str, List[str]]:
    """
    Load cached text from the temp folder. If a cache file exists, load it,
    otherwise return an empty dict.
    
    Args:
        temp_folder: Path to temporary folder containing extracted files
        logs_ref: Dictionary for logging steps and errors
        
    Returns:
        Dict[str, List[str]]: Mapping of filenames to their paragraphs
    """
    cache_path = os.path.join(temp_folder, "text_cache.pkl")
    
    if os.path.exists(cache_path):
        logs_ref["steps"].append(f"Loading text cache from {cache_path}")
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logs_ref["errors"].append(f"Error loading text cache: {str(e)}")
            return {}
    else:
        logs_ref["steps"].append("No text cache found, creating new cache")
        return {}

def save_text_cache(temp_folder: str, texts_by_doc: Dict[str, List[str]], logs_ref: Dict) -> None:
    """
    Save extracted text to a cache file in the temp folder.
    
    Args:
        temp_folder: Path to temporary folder
        texts_by_doc: Dictionary mapping filenames to their paragraphs
        logs_ref: Dictionary for logging steps and errors
    """
    cache_path = os.path.join(temp_folder, "text_cache.pkl")
    logs_ref["steps"].append(f"Saving text cache to {cache_path}")
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(texts_by_doc, f)
    except Exception as e:
        logs_ref["errors"].append(f"Error saving text cache: {str(e)}")

def extract_text_from_file(file_path: str, logs_ref: Dict) -> List[str]:
    """
    Extract text content from various file types.
    
    Args:
        file_path: Path to the file to extract text from
        logs_ref: Dictionary for logging steps and errors
        
    Returns:
        List[str]: List of extracted paragraphs
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path, logs_ref)
    elif ext == ".docx":
        return extract_text_from_docx(file_path, logs_ref)
    elif ext == ".txt":
        return extract_text_from_txt(file_path, logs_ref)
    elif ext == ".csv":
        return extract_text_from_csv(file_path, logs_ref)
    elif ext == ".json":
        return extract_text_from_json(file_path, logs_ref)
    else:
        logs_ref["steps"].append(f"Skipping unsupported file type: {ext}")
        return []

def extract_text_from_pdf(pdf_path: str, logs_ref: Dict) -> List[str]:
    logs_ref["steps"].append(f"Parsing PDF: {pdf_path}")
    paragraphs = []
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text = page.get_text()
            blocks = text.split("\n\n")
            for block in blocks:
                block_stripped = block.strip()
                if block_stripped:
                    paragraphs.append(block_stripped)
    except Exception as e:
        logs_ref["errors"].append(f"Error parsing PDF: {str(e)}")
    return paragraphs

def extract_text_from_docx(docx_path: str, logs_ref: Dict) -> List[str]:
    logs_ref["steps"].append(f"Parsing DOCX: {docx_path}")
    paragraphs = []
    try:
        doc = Document(docx_path)
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                paragraphs.append(text)
    except Exception as e:
        logs_ref["errors"].append(f"Error parsing DOCX: {str(e)}")
    return paragraphs

def extract_text_from_txt(txt_path: str, logs_ref: Dict) -> List[str]:
    logs_ref["steps"].append(f"Parsing TXT: {txt_path}")
    paragraphs = []
    try:
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            blocks = content.split("\n\n")
            for block in blocks:
                strip_block = block.strip()
                if strip_block:
                    paragraphs.append(strip_block)
    except Exception as e:
        logs_ref["errors"].append(f"Error parsing TXT: {str(e)}")
    return paragraphs

def extract_text_from_csv(csv_path: str, logs_ref: Dict) -> List[str]:
    logs_ref["steps"].append(f"Parsing CSV: {csv_path}")
    paragraphs = []
    try:
        df = pd.read_csv(csv_path, encoding="utf-8", engine="python", on_bad_lines='skip')
        for index, row in df.iterrows():
            row_text = " ".join(str(val) for val in row if pd.notna(val))
            row_text = row_text.strip()
            if row_text:
                paragraphs.append(row_text)
    except Exception as e:
        logs_ref["errors"].append(f"Error parsing CSV: {str(e)}")
    return paragraphs

def extract_text_from_json(json_path: str, logs_ref: Dict) -> List[str]:
    logs_ref["steps"].append(f"Parsing JSON: {json_path}")
    paragraphs = []
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    paragraphs.append(item.strip())
                elif isinstance(item, dict):
                    combined = " ".join(str(v) for v in item.values() if v).strip()
                    if combined:
                        paragraphs.append(combined)
        elif isinstance(data, dict):
            combined = " ".join(str(v) for v in data.values() if v).strip()
            if combined:
                paragraphs.append(combined)
    except Exception as e:
        logs_ref["errors"].append(f"Error parsing JSON: {str(e)}")
    return paragraphs