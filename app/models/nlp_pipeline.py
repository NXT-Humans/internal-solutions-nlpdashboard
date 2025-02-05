"""
nlp_pipeline.py - Transformer-Based NLP Pipeline with Contextual Modal Integration

This module integrates text summarization, topic extraction, sentiment analysis, and word cloud generation.
It builds a unified text mapping, processes documents through multiple stages, and compiles a comprehensive JSON result.
It also uses a simple content analyzer to determine the document type for adaptive processing.
"""

from typing import Dict, List, Any
from datetime import datetime
import numpy as np
import torch
from app.models.summarizer import Summarizer
from app.models.topic_modeler import TopicModeler
from app.models.sentiment_analyzer import SentimentAnalyzer
from sklearn.metrics.pairwise import cosine_similarity
import re


class ContentAnalyzer:
    def analyze_structure(self, paragraphs: List[str]) -> str:
        """
        Determine the document type based on key patterns using basic keyword matching.

        Args:
            paragraphs: A list of paragraphs.

        Returns:
            A string indicating the document type.
        """
        combined_text = " ".join(paragraphs).lower()
        if "study" in combined_text or "research" in combined_text or "methodology" in combined_text:
            return "academic"
        if "comment by" in combined_text or "upvote" in combined_text or "posted" in combined_text:
            return "social media"
        if re.search(r'\b[A-Z][a-zA-Z]+:', combined_text):
            return "conversation"
        return "general"


class NLPPipeline:
    def __init__(self, logs_reference: Dict[str, Any]) -> None:
        """
        Initialize the NLP pipeline components.

        Args:
            logs_reference: A dictionary for logging.
        """
        self.logs = logs_reference
        if "steps" not in self.logs or not isinstance(self.logs["steps"], list):
            self.logs["steps"] = []
        if "errors" not in self.logs or not isinstance(self.logs["errors"], list):
            self.logs["errors"] = []
        self.summarizer = Summarizer(logs_reference)
        self.topic_modeler = TopicModeler(logs_reference)
        self.sentiment_analyzer = SentimentAnalyzer(logs_reference)
        self.content_analyzer = ContentAnalyzer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logs["stage_details"] = {
            "preprocessing": {"status": "pending"},
            "topic_extraction": {"status": "pending"},
            "sentiment_analysis": {"status": "pending"},
            "summary_generation": {"status": "pending"},
            "visualization_preparation": {"status": "pending"}
        }
        self.logs["files_processed"] = {}
        self.logs["steps"].append("NLPPipeline initialized")

    def process(self, texts_by_document: Dict[str, List[str]], **options) -> Dict[str, Any]:
        """
        Process documents through the pipeline.

        Args:
            texts_by_document: A dictionary mapping document names to lists of paragraphs.
            options: Additional processing options.

        Returns:
            A dictionary with the full analysis results.
        """
        try:
            if not self._validate_input(texts_by_document):
                raise ValueError("Invalid input document format")

            # Build a unified text mapping and corpus.
            text_mapping = {}
            unified_corpus = []
            for document_name, paragraphs in texts_by_document.items():
                for index, paragraph in enumerate(paragraphs):
                    if isinstance(paragraph, str) and paragraph.strip():
                        paragraph_identifier = f"{document_name}__paragraph{index}"
                        text_mapping[paragraph_identifier] = {
                            "document": document_name,
                            "index": index,
                            "text": paragraph.strip()
                        }
                        unified_corpus.append(paragraph.strip())

            # Determine the document type for each document.
            document_types = {
                document: self.content_analyzer.analyze_structure(paragraphs)
                for document, paragraphs in texts_by_document.items()
            }

            # Process stages with robust error tracking.
            stage_results = {}
            processing_stages = [
                ("preprocessing", self._preprocess_documents),
                ("topic_extraction", self._process_topics),
                ("sentiment_analysis", self._process_sentiment),
                ("summary_generation", self._process_summary),
                ("visualization_preparation", self._prepare_visualizations)
            ]
            for stage_name, stage_function in processing_stages:
                self.logs["stage_details"][stage_name]["status"] = "in_progress"
                try:
                    stage_results[stage_name] = stage_function(texts_by_document, unified_corpus, text_mapping)
                    self.logs["stage_details"][stage_name]["status"] = "complete"
                except Exception as stage_exception:
                    self.logs["stage_details"][stage_name]["status"] = "error"
                    self.logs["stage_details"][stage_name]["error"] = str(stage_exception)
                    raise

            # Build document-level analysis.
            document_results = {}
            document_paragraphs = {}
            for paragraph_identifier, info in text_mapping.items():
                document = info["document"]
                document_paragraphs.setdefault(document, []).append(info["text"])
            for document_name, paragraphs in document_paragraphs.items():
                document_results[document_name] = {
                    "topics": self._extract_document_topics(paragraphs, stage_results["topic_extraction"]),
                    "sentiment": self._extract_document_sentiment(paragraphs, stage_results["sentiment_analysis"]),
                    "summary": self.summarizer.summarize_paragraphs(
                        paragraphs, document_types.get(document_name)
                    ),
                    "processed_paragraphs": len(paragraphs)
                }

            return {
                "documents": document_results,
                "global_summary": stage_results["summary_generation"],
                "global_topics": stage_results["topic_extraction"],
                "global_sentiment": stage_results["sentiment_analysis"],
                "wordcloud_data": stage_results["visualization_preparation"],
                "text_mapping": text_mapping
            }
        except Exception as process_exception:
            self.logs["errors"].append(str(process_exception))
            raise Exception(str(process_exception))

    def _validate_input(self, texts_by_document: Dict[str, List[str]]) -> bool:
        """
        Validate the input format for the documents.

        Args:
            texts_by_document: A dictionary mapping document names to lists of paragraphs.

        Returns:
            True if the input format is valid; otherwise, False.
        """
        if not isinstance(texts_by_document, dict):
            return False
        for document, paragraphs in texts_by_document.items():
            if not isinstance(document, str) or not isinstance(paragraphs, list):
                return False
            if not all(isinstance(paragraph, str) for paragraph in paragraphs):
                return False
        return True

    def _preprocess_documents(self, texts_by_document: Dict[str, List[str]], corpus: List[str],
                              text_mapping: Dict[str, Any]) -> List[str]:
        """
        Preprocessing stage: return the unified corpus.

        Args:
            texts_by_document: A dictionary of documents.
            corpus: A list of all paragraphs.
            text_mapping: A mapping of paragraph identifiers to paragraph metadata.

        Returns:
            The unified corpus as a list of paragraphs.
        """
        return corpus

    def _process_topics(self, texts_by_document: Dict[str, List[str]], corpus: List[str],
                        text_mapping: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process topics from the unified corpus using the TopicModeler and link topics to documents.

        Args:
            texts_by_document: A dictionary of documents.
            corpus: A list of all paragraphs.
            text_mapping: A mapping of paragraph identifiers to paragraph metadata.

        Returns:
            A list of topic dictionaries.
        """
        topics = self.topic_modeler.extract_topics(corpus)
        document_topics = {}
        for document, paragraphs in texts_by_document.items():
            document_topics[document] = []
            for topic in topics:
                for paragraph in paragraphs:
                    if self._compute_relevance(topic["title"], paragraph) > 0.25:
                        document_topics[document].append(topic["title"])
                        break
        merged_topics = self._merge_topic_hierarchies(topics, document_topics)
        return merged_topics

    def _merge_topic_hierarchies(self, topics: List[Dict[str, Any]],
                                 document_topics: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Merge and enrich topics with cross-document information.

        Args:
            topics: A list of topic dictionaries.
            document_topics: A mapping of documents to lists of topics.

        Returns:
            A merged list of topic dictionaries.
        """
        return topics

    def _process_sentiment(self, texts_by_document: Dict[str, List[str]], corpus: List[str],
                           text_mapping: Dict[str, Any]) -> Any:
        """
        Process sentiment for the unified corpus using the SentimentAnalyzer.

        Args:
            texts_by_document: A dictionary of documents.
            corpus: A list of all paragraphs.
            text_mapping: A mapping of paragraph identifiers to paragraph metadata.

        Returns:
            The sentiment analysis results.
        """
        sentiments = self.sentiment_analyzer.analyze(corpus)
        return sentiments

    def _process_summary(self, texts_by_document: Dict[str, List[str]], corpus: List[str],
                         text_mapping: Dict[str, Any]) -> Any:
        """
        Generate a global summary for the unified corpus using an academic strategy by default.

        Args:
            texts_by_document: A dictionary of documents.
            corpus: A list of all paragraphs.
            text_mapping: A mapping of paragraph identifiers to paragraph metadata.

        Returns:
            The global summary result.
        """
        return self.summarizer.summarize_paragraphs(corpus, "academic")

    def _prepare_visualizations(self, texts_by_document: Dict[str, List[str]], corpus: List[str],
                                text_mapping: Dict[str, Any]) -> Any:
        """
        Prepare word cloud data using trigram keys derived from topics.

        Args:
            texts_by_document: A dictionary of documents.
            corpus: A list of all paragraphs.
            text_mapping: A mapping of paragraph identifiers to paragraph metadata.

        Returns:
            A data structure suitable for generating a word cloud.
        """
        topics = self._process_topics(texts_by_document, corpus, text_mapping)
        wordcloud_data = {}
        for topic in topics:
            term_importance = self._calculate_term_importance(topic["title"], topic)
            relevant_contexts = []
            for paragraph in corpus:
                relevance_score = self._compute_relevance(topic["title"], paragraph)
                if relevance_score > 0.25:
                    relevant_contexts.append({
                        "text": paragraph,
                        "relevance": relevance_score,
                        "sentiment": self._get_text_sentiment(paragraph)
                    })
            sorted_contexts = sorted(relevant_contexts, key=lambda context: context["relevance"], reverse=True)[:5]
            wordcloud_data[topic["title"]] = {
                "frequency": topic.get("cluster_size", 1) * term_importance,
                "contexts": sorted_contexts,
                "sentiment": self._compute_topic_sentiment(relevant_contexts),
                "category": self._categorize_term(topic["title"])
            }
        return wordcloud_data

    def _extract_document_topics(self, document_paragraphs: List[str], global_topics: List[Any]) -> List[str]:
        """
        Extract topics relevant to a specific document.

        Args:
            document_paragraphs: A list of paragraphs for the document.
            global_topics: Global topics extracted from the corpus.

        Returns:
            A list of topic titles relevant to the document.
        """
        relevant_topics = []
        for topic in global_topics:
            for paragraph in document_paragraphs:
                if self._compute_relevance(topic["title"], paragraph) > 0.25:
                    relevant_topics.append(topic["title"])
                    break
        return relevant_topics

    def _extract_document_sentiment(self, document_paragraphs: List[str], global_sentiments: Any) -> Any:
        """
        Compute the overall sentiment for a document.

        Args:
            document_paragraphs: A list of paragraphs for the document.
            global_sentiments: Global sentiment analysis results.

        Returns:
            The average sentiment value for the document.
        """
        sentiments = self.sentiment_analyzer.analyze(document_paragraphs)
        if sentiments:
            average_sentiment = sum(sentiment["polarity"] for sentiment in sentiments) / len(sentiments)
            return average_sentiment
        return 0.0

    def _compute_relevance(self, topic: str, text: str) -> float:
        """
        Compute the semantic relevance between a topic and a text.

        Args:
            topic: The topic string.
            text: The text to compare.

        Returns:
            A float value representing the relevance score.
        """
        try:
            if not topic or not text:
                return 0.0
            topic_embedding = self.topic_modeler._get_text_embeddings([topic])[0]
            text_embedding = self.topic_modeler._get_text_embeddings([text])[0]
            cosine_similarity_value = float(
                np.dot(topic_embedding, text_embedding) /
                (np.linalg.norm(topic_embedding) * np.linalg.norm(text_embedding))
            )
            return cosine_similarity_value
        except Exception as relevance_exception:
            return 0.0

    def _get_text_sentiment(self, text: str) -> float:
        """
        Obtain the sentiment polarity for a given text.

        Args:
            text: The text string.

        Returns:
            A float representing the sentiment polarity.
        """
        try:
            if not text or not text.strip():
                return 0.0
            sentiment_result = self.sentiment_analyzer.analyze([text])
            if sentiment_result:
                return sentiment_result[0]["polarity"]
            return 0.0
        except Exception as sentiment_exception:
            return 0.0

    def _compute_topic_sentiment(self, contexts: List[Dict[str, Any]]) -> float:
        """
        Compute the average sentiment for a list of context segments.

        Args:
            contexts: A list of dictionaries containing context information.

        Returns:
            A float representing the average sentiment.
        """
        try:
            if not contexts:
                return 0.0
            sentiment_scores = [
                context.get("sentiment", 0.0)
                for context in contexts if isinstance(context.get("sentiment", 0.0), (int, float))
            ]
            if sentiment_scores:
                return sum(sentiment_scores) / len(sentiment_scores)
            return 0.0
        except Exception as topic_sentiment_exception:
            self.logs["errors"].append(f"Error computing topic sentiment: {str(topic_sentiment_exception)}")
            return 0.0

    def _calculate_term_importance(self, topic: str, topic_data: Dict[str, Any]) -> float:
        """
        Calculate term importance based on frequency and weighting.

        Args:
            topic: The topic string.
            topic_data: A dictionary containing topic data.

        Returns:
            A float representing the term importance.
        """
        title_weight = 2.0
        key_section_weight = 1.5
        base_frequency = topic_data.get("frequency", 1)
        term_importance = (base_frequency * 1.0) + title_weight + key_section_weight
        return term_importance

    def _categorize_term(self, topic: str) -> str:
        """
        Categorize a topic into a high-level category.

        Args:
            topic: The topic string.

        Returns:
            A string representing the category of the topic.
        """
        topic_lower = topic.lower()
        if any(keyword in topic_lower for keyword in ["experiment", "analysis", "evaluation"]):
            return "Methodology"
        if any(keyword in topic_lower for keyword in ["result", "accuracy", "performance"]):
            return "Findings"
        if any(keyword in topic_lower for keyword in ["challenge", "limitation", "constraint"]):
            return "Challenges"
        return "General"
