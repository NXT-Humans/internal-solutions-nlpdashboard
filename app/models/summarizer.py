"""
summarizer.py - Enhanced Transformer-Based Research Content Summarizer

This module uses the BART model to generate structured summaries with four distinct sections:
    1) Overview
    2) Key Findings
    3) Primary Challenges
    4) Strategic Solutions

It employs a zero-shot classification pipeline to automatically filter out metadata so that summaries are built solely from research content.
"""

__all__ = ["Summarizer"]

import re
from typing import Dict, List, Any, Union
import logging
import torch
from transformers import pipeline, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Summarizer:
    def __init__(self, logs_reference: Dict[str, Any]) -> None:
        """
        Initialize the Summarizer with the BART model and a zero-shot classifier for content detection.

        Args:
            logs_reference: A dictionary for logging initialization steps and errors.
        """
        self.logs = logs_reference
        if "steps" not in self.logs or not isinstance(self.logs["steps"], list):
            self.logs["steps"] = []
        if "errors" not in self.logs or not isinstance(self.logs["errors"], list):
            self.logs["errors"] = []
        bart_model_identifier = "facebook/bart-large-cnn"
        mnli_model_identifier = "facebook/bart-large-mnli"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(bart_model_identifier, force_download=False)
            self.model_maximum_length = min(self.tokenizer.model_max_length, 512)
            self.summarization_pipeline = pipeline(
                "summarization",
                model=bart_model_identifier,
                tokenizer=self.tokenizer,
                device=-1,
                truncation=True,
                max_length=self.model_maximum_length
            )
            self.classifier = pipeline("zero-shot-classification", model=mnli_model_identifier)
            self.candidate_labels = ["research", "metadata", "boilerplate", "other"]
            self.logs["steps"].append("Initialized BART summarization and zero-shot classification pipelines")
        except Exception as initialization_exception:
            error_message = f"Error initializing summarizer: {str(initialization_exception)}"
            self.logs["errors"].append(error_message)
            logger.error(error_message)
            raise RuntimeError("Summarizer initialization failed") from initialization_exception

    def summarize_paragraphs(self, paragraphs: List[str], document_type: str) -> Dict[str, str]:
        """
        Generate a structured summary from a list of paragraphs with clear section differentiation.

        Args:
            paragraphs: A list of paragraphs.
            document_type: A string indicating the document type (for example, "academic").

        Returns:
            A dictionary with keys "overview", "findings", "challenges", and "solutions".
        """
        if not paragraphs:
            return self._format_empty_summary()

        cleaned_paragraphs = [self._clean_text(paragraph) for paragraph in paragraphs if paragraph.strip()]
        if not cleaned_paragraphs:
            return self._format_empty_summary()

        # Filter out non-research content using zero-shot classification.
        research_paragraphs = [
            paragraph for paragraph in cleaned_paragraphs
            if self._detect_content_type(paragraph) == "research"
        ]
        if not research_paragraphs:
            research_paragraphs = cleaned_paragraphs

        # Generate an overview using a custom prompt.
        overview_prompt = "Generate a concise overview focusing on the key contributions and methodology: "
        overview_input = overview_prompt + " ".join(research_paragraphs[:3])
        try:
            overview_result = self.summarization_pipeline(
                overview_input,
                max_length=self.model_maximum_length,
                min_length=30,
                truncation=True
            )
            overview = overview_result[0]["summary_text"]
        except Exception as overview_exception:
            overview = "Error generating overview."
            self.logs["errors"].append(f"Overview generation error: {str(overview_exception)}")
            logger.error(f"Overview generation error: {str(overview_exception)}")

        # Extract key findings based on numerical metrics.
        key_findings_list = []
        numerical_metrics_pattern = r'(\d+(?:\.\d+)?%|\d+(?:\.\d+)?x|\d+\s*(?:thousand|million|billion)|\d+(?:\.\d+)?)'
        for paragraph in research_paragraphs:
            sentences = paragraph.split('.')
            for sentence in sentences:
                if re.search(numerical_metrics_pattern, sentence):
                    key_findings_list.append(sentence.strip())

        # Extract primary challenges using flexible patterns.
        primary_challenges_list = []
        challenges_patterns = [
            r'(?:challenge|limitation|constraint|difficulty|problem).*?(?:\.|$)',
            r'(?:lack|missing|required).*?(?:\.|$)'
        ]
        for paragraph in research_paragraphs:
            for pattern in challenges_patterns:
                for match in re.finditer(pattern, paragraph, re.IGNORECASE):
                    matched_text = match.group(0).strip()
                    if len(matched_text) > 20:
                        primary_challenges_list.append(matched_text)

        # Extract strategic solutions using flexible patterns.
        strategic_solutions_list = []
        solutions_patterns = [
            r'(?:propose|present|introduce|develop).*?(?:\.|$)',
            r'(?:solution|approach|method|technique).*?(?:\.|$)',
            r'(?:implement|apply|utilize|leverage).*?(?:\.|$)'
        ]
        for paragraph in research_paragraphs:
            for pattern in solutions_patterns:
                for match in re.finditer(pattern, paragraph, re.IGNORECASE):
                    matched_text = match.group(0).strip()
                    if len(matched_text) > 20:
                        strategic_solutions_list.append(matched_text)

        structured_summary = {
            "overview": self._format_section("Overview", overview),
            "findings": self._format_section("Key Findings", key_findings_list[:5]),
            "challenges": self._format_section("Primary Challenges", primary_challenges_list[:3]),
            "solutions": self._format_section("Strategic Solutions", strategic_solutions_list[:3])
        }
        return structured_summary

    def _detect_content_type(self, text: str) -> str:
        """
        Automatically detect the content type of a paragraph using zero-shot classification.

        Args:
            text: The paragraph text.

        Returns:
            A string indicating the content type.
        """
        try:
            classification_result = self.classifier(text, candidate_labels=self.candidate_labels)
            return classification_result["labels"][0].lower()
        except Exception as detection_exception:
            logger.error(f"Error in content type detection: {str(detection_exception)}")
            return "other"

    def _format_section(self, title: str, content: Union[str, List[str]]) -> str:
        """
        Format a summary section with a title and content.

        Args:
            title: The title of the section.
            content: The content of the section as a string or list of strings.

        Returns:
            A formatted string with the section title and content.
        """
        if isinstance(content, list):
            content = "\n".join(f"- {line}" for line in content)
        return f"{title}:\n{content}"

    def _format_empty_summary(self) -> Dict[str, str]:
        """
        Return a default empty summary structure.

        Returns:
            A dictionary with default empty summary values.
        """
        return {
            "overview": "No content provided for analysis.",
            "findings": "No content provided for analysis.",
            "challenges": "No content provided for analysis.",
            "solutions": "No content provided for analysis."
        }

    def _clean_text(self, text: str) -> str:
        """
        Clean text by removing citations and normalizing whitespace.

        Args:
            text: The raw text.

        Returns:
            The cleaned text.
        """
        try:
            text_without_citations = re.sub(r'\[\d+\]', '', text)
            normalized_whitespace = re.sub(r'\s+', ' ', text_without_citations)
            normalized_punctuation = re.sub(r'\s*([.,!?:;])\s*', r'\1 ', normalized_whitespace)
            return normalized_punctuation.strip()
        except Exception as cleaning_exception:
            logger.error(f"Error cleaning text: {str(cleaning_exception)}")
            return text if isinstance(text, str) else ""
