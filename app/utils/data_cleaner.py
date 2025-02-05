"""
data_cleaner.py - Enhanced text cleaning with regex patterns
"""

import re
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class DataCleaner:
    def __init__(self, logs_ref: Dict):
        self.logs = logs_ref
        
        # Compile regex patterns for efficiency
        self.patterns = {
            'citations': re.compile(r'\[\d+\]|\(\w+\s+et\s+al\.\s*,\s*\d{4}\)'),
            'urls': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            'emails': re.compile(r'\S+@\S+\.\S+'),
            'multiple_spaces': re.compile(r'\s+'),
            'special_chars': re.compile(r'[^\w\s.!?;:,\-]'),
            'section_headers': re.compile(r'^(?:INTRODUCTION|METHODOLOGY|RESULTS|DISCUSSION|CONCLUSION)s?:?\s*$', re.I)
        }

    def clean_paragraphs(self, paragraphs: List[str]) -> List[str]:
        """Clean list of paragraphs using regex patterns"""
        cleaned = []
        for para in paragraphs:
            try:
                cleaned_para = self._clean_text(para)
                if cleaned_para:
                    cleaned.append(cleaned_para)
            except Exception as e:
                self.logs["errors"].append(f"Error cleaning paragraph: {str(e)}")
                logger.error(f"Error cleaning paragraph: {e}")
        return cleaned

    def _clean_text(self, text: str) -> str:
        """Apply cleaning patterns to text"""
        if not text or not text.strip():
            return ""

        # Remove citations
        text = self.patterns['citations'].sub('', text)
        
        # Remove URLs
        text = self.patterns['urls'].sub('', text)
        
        # Remove email addresses
        text = self.patterns['emails'].sub('', text)
        
        # Remove special characters but keep basic punctuation
        text = self.patterns['special_chars'].sub('', text)
        
        # Normalize whitespace
        text = self.patterns['multiple_spaces'].sub(' ', text)
        
        # Clean up punctuation spacing
        text = re.sub(r'\s*([.,!?;:])\s*', r'\1 ', text)
        
        # Remove section headers if they appear alone
        if self.patterns['section_headers'].match(text.strip()):
            return ""
            
        return text.strip()

    def is_valid_paragraph(self, text: str) -> bool:
        """Check if paragraph is valid after cleaning"""
        if not text or not text.strip():
            return False
            
        # Check minimum length (e.g., at least 20 chars)
        if len(text.strip()) < 20:
            return False
            
        # Check if it's just a section header
        if self.patterns['section_headers'].match(text.strip()):
            return False
            
        return True