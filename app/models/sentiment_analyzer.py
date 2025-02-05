"""
sentiment_analyzer.py - Transformer-Based Sentiment Analyzer

This module uses DistilBERT finetuned on SST-2 to analyze sentiment.
It supports chunking of long texts and returns polarity and label information.
"""

__all__ = ["SentimentAnalyzer"]

import logging
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self, logs_ref: Dict) -> None:
        """
        Initialize the Sentiment Analyzer with DistilBERT finetuned on SST-2.

        Args:
            logs_ref: Dictionary for logging initialization steps and errors.
        """
        self.logs = logs_ref
        if "steps" not in self.logs or not isinstance(self.logs["steps"], list):
            self.logs["steps"] = []
        if "errors" not in self.logs or not isinstance(self.logs["errors"], list):
            self.logs["errors"] = []
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.classifier = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                truncation=True
            )
            self.max_tokens = 512
            self.logs["steps"].append("SentimentAnalyzer initialized")
        except Exception as e:
            error_msg = f"Error initializing SentimentAnalyzer: {str(e)}"
            self.logs["errors"].append(error_msg)
            logger.error(error_msg)
            raise

    def analyze(self, paragraphs: List[str]) -> List[Dict]:
        """
        Analyze sentiment for each paragraph, using chunking if necessary.

        Args:
            paragraphs: List of paragraph strings.

        Returns:
            A list of dictionaries with keys "polarity" and "label".
        """
        results = []
        for paragraph in paragraphs:
            polarity, label = self._analyze_chunked(paragraph)
            results.append({"polarity": polarity, "label": label})
        return results

    def _analyze_chunked(self, text: str) -> Tuple[float, str]:
        """
        Chunk a paragraph if it exceeds the token limit and average sentiment scores.

        Args:
            text: The paragraph text.

        Returns:
            A tuple (average polarity, majority label).
        """
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) <= self.max_tokens:
            return self._analyze_one_chunk(text)
        chunk_size = 256
        polarities = []
        labels = []
        for i in range(0, len(tokens), chunk_size):
            subset = tokens[i: i + chunk_size]
            chunk_text = self.tokenizer.convert_tokens_to_string(subset)
            pol, lab = self._analyze_one_chunk(chunk_text)
            polarities.append(pol)
            labels.append(lab)
        avg_polarity = sum(polarities) / len(polarities)
        final_label = max(set(labels), key=labels.count)
        return (avg_polarity, final_label)

    def _analyze_one_chunk(self, text_chunk: str) -> Tuple[float, str]:
        """
        Analyze sentiment for a text chunk.

        Args:
            text_chunk: The text chunk.

        Returns:
            A tuple (polarity, label).
        """
        try:
            result = self.classifier(text_chunk)
            if result and isinstance(result, list):
                first = result[0]
                label = first["label"]
                score = first["score"]
                polarity = score if label.upper() == "POSITIVE" else -score
                return (polarity, label)
        except Exception as ex:
            logger.error(f"Sentiment error: {ex}", exc_info=True)
            self.logs["errors"].append(f"Sentiment error: {str(ex)}")
        return (0.0, "NEUTRAL")
