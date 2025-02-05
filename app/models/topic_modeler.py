"""
topic_modeler.py - BERT-Based Topic Extraction for Main Ideas

This module uses a BERT-based transformer to embed and hierarchically cluster paragraphs.
It extracts concise, meaningful topics as complete trigram phrases (after filtering out stopwords),
combining semantic clustering and syntactic phrase extraction, and returns the top 25 topics.
"""

import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from typing import List, Dict, Any
import re
import string
from collections import Counter, defaultdict
from spacy.lang.en.stop_words import STOP_WORDS

class TopicModeler:
    def __init__(self, logs_reference: Dict[str, Any]) -> None:
        """
        Initialize the Topic Modeler with a transformer-based embedding model.

        Args:
            logs_reference: Dictionary for logging.
        """
        self.logs = logs_reference
        if "steps" not in self.logs or not isinstance(self.logs["steps"], list):
            self.logs["steps"] = []
        if "errors" not in self.logs or not isinstance(self.logs["errors"], list):
            self.logs["errors"] = []
        model_name = "sentence-transformers/all-mpnet-base-v2"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.logs["steps"].append("Initialized topic modeler with all-mpnet-base-v2")
        except Exception as e:
            error_msg = f"Error initializing topic modeler: {str(e)}"
            self.logs["errors"].append(error_msg)
            raise RuntimeError("Topic modeler initialization failed") from e

        self.research_terms = {
            "methodology": {"experiment", "analysis", "evaluation", "methodology"},
            "findings": {"results", "findings", "performance", "accuracy"},
            "technology": {"algorithm", "model", "system", "architecture"},
            "domain": {"finance", "trading", "market", "investment"}
        }

    def extract_topics(self, paragraphs: List[str]) -> List[Dict[str, Any]]:
        """
        Dynamically extract topics as concise trigram titles from a list of paragraphs.
        Combines semantic clustering with syntactic phrase extraction and returns the top 25 topics.

        Args:
            paragraphs: List of raw paragraphs.

        Returns:
            A list of topic dictionaries.
        """
        try:
            cleaned_texts = self._preprocess_texts(paragraphs)
            if not cleaned_texts:
                return []
            # Semantic clustering using AgglomerativeClustering
            embeddings = self._get_text_embeddings(cleaned_texts)
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=0.4,
                metric='cosine',
                linkage='complete'
            )
            cluster_labels = clustering.fit_predict(embeddings)
            topics = []
            unique_labels = set(cluster_labels)
            for label in unique_labels:
                if label == -1:
                    continue  # Skip noise
                cluster_texts = [cleaned_texts[i] for i in range(len(cleaned_texts)) if cluster_labels[i] == label]
                if cluster_texts:
                    topic_title = self._extract_representative_title(cluster_texts)
                    keywords = self._extract_keywords(cluster_texts)
                    context = self._select_best_context(cluster_texts)
                    topics.append({
                        "title": topic_title,
                        "type": "semantic",
                        "keywords": keywords,
                        "context": context,
                        "cluster_size": len(cluster_texts),
                        "paragraphs": cluster_texts
                    })
            # Syntactic phrase extraction for additional topics
            syntactic_topics = self._extract_syntactic_phrases(paragraphs)
            for topic in syntactic_topics:
                topics.append({
                    "title": topic["phrase"],
                    "type": "syntactic",
                    "keywords": [],
                    "context": topic["contexts"][0] if topic["contexts"] else "",
                    "cluster_size": topic["frequency"],
                    "paragraphs": topic["contexts"]
                })
            topics = self._filter_and_rank_topics(topics)
            topics = sorted(topics, key=lambda x: x.get("cluster_size", 1), reverse=True)[:25]
            return topics
        except Exception as e:
            self.logs["errors"].append(f"Error extracting topics: {str(e)}")
            return []

    def _preprocess_texts(self, texts: List[str]) -> List[str]:
        """Clean and prepare texts for embedding."""
        cleaned = []
        for text in texts:
            if isinstance(text, str) and text.strip():
                text = re.sub(r'\[\d+\]|\(\d{4}\)', '', text)
                text = re.sub(r'http\S+|www\.\S+', '', text)
                text = re.sub(r'[^\w\s.,!?-]', '', text)
                text = " ".join(text.split())
                cleaned.append(text)
        return cleaned

    def _get_text_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts using the transformer model."""
        embeddings = []
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(text, truncation=True, max_length=512, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state[0, 0, :].cpu().numpy()
                embeddings.append(embedding)
        return np.vstack(embeddings)

    def _extract_trigram(self, texts: List[str]) -> str:
        """
        Extract an approximate trigram from a list of texts.
        Tokenizes the combined text, filters out stop words and punctuation,
        computes trigrams, and returns the most frequent trigram.
        """
        combined = " ".join(texts).lower()
        translator = str.maketrans("", "", string.punctuation)
        combined = combined.translate(translator)
        tokens = combined.split()
        tokens = [t for t in tokens if t not in STOP_WORDS]
        if len(tokens) < 3:
            return " ".join(tokens)
        trigrams = [" ".join(tokens[i:i+3]) for i in range(len(tokens) - 2)]
        if not trigrams:
            return ""
        trigram_counts = Counter(trigrams)
        most_common, _ = trigram_counts.most_common(1)[0]
        return most_common

    def _extract_representative_title(self, texts: List[str]) -> str:
        """
        Extract a representative title for the topic using an approximate trigram.
        Ensures that meaningless fragments are removed.
        """
        trigram = self._extract_trigram(texts)
        if trigram and len(trigram.split()) == 3:
            return trigram
        first_sentence = texts[0].split('.')[0]
        return first_sentence[:100] + ('...' if len(first_sentence) > 100 else '')

    def _select_best_context(self, texts: List[str]) -> str:
        """
        Select the most representative paragraph as context for the cluster.
        """
        return max(texts, key=len)

    def _extract_keywords(self, texts: List[str]) -> List[str]:
        """
        Dynamically extract keywords from a list of texts.
        """
        combined = " ".join(texts).lower()
        keywords = set()
        for term_set in self.research_terms.values():
            keywords.update({term for term in term_set if term in combined})
        technical_pattern = r'\b[A-Z][A-Za-z\d]+\b'
        technical_terms = set(re.findall(technical_pattern, " ".join(texts)))
        keywords.update(term.lower() for term in technical_terms)
        return sorted(keywords)

    def _extract_syntactic_phrases(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Extract topics using syntactic patterns.
        """
        patterns = [
            (r'(\w+(?:\s+\w+){1,3})\s+(?:architecture|system|framework)', "architecture"),
            (r'(?:propose|present|introduce|develop)\s+(\w+(?:\s+\w+){1,3})\s+(?:method|approach|technique)', "method"),
            (r'(\w+(?:\s+\w+){1,3})\s+(?:task|problem|challenge)', "task"),
            (r'(\w+(?:\s+\w+){1,3})\s+(?:model|algorithm|solution)', "technology")
        ]
        topics = []
        for text in texts:
            for pattern, topic_type in patterns:
                matches = re.finditer(pattern, text, re.I)
                for match in matches:
                    phrase = match.group(1).strip()
                    if len(phrase.split()) >= 2:
                        topics.append({
                            "phrase": phrase,
                            "pattern": topic_type,
                            "frequency": 1,
                            "contexts": [text]
                        })
        merged = defaultdict(lambda: {"frequency": 0, "contexts": []})
        for topic in topics:
            key = (topic["phrase"], topic["pattern"])
            merged[key]["frequency"] += 1
            merged[key]["contexts"].extend(topic["contexts"])
        return [{"phrase": k[0], "pattern": k[1], "frequency": v["frequency"], "contexts": v["contexts"]}
                for k, v in merged.items()]

    def _filter_and_rank_topics(self, topics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter and rank topics based on a combined score of coherence and distinctiveness.
        """
        if not topics:
            return []
        scored_topics = []
        for topic in topics:
            coherence = self._calculate_topic_coherence(topic)
            distinctiveness = self._calculate_distinctiveness(topic, topics)
            score = coherence * distinctiveness
            scored_topics.append((topic, score))
        distinct_topics = self._remove_redundant_topics(scored_topics)
        return [item[0] for item in distinct_topics]

    def _calculate_topic_coherence(self, topic: Dict[str, Any]) -> float:
        """Calculate a coherence score based on keywords and cluster size."""
        num_keywords = len(topic.get("keywords", []))
        cluster_size = topic.get("cluster_size", 1)
        return 0.5 * num_keywords + 0.5 * cluster_size

    def _calculate_distinctiveness(self, topic: Dict[str, Any], topics: List[Dict[str, Any]]) -> float:
        """Calculate distinctiveness relative to other topics."""
        topic_title = topic.get("title", "").lower()
        similarity_sum = 0.0
        for other in topics:
            if other == topic:
                continue
            other_title = other.get("title", "").lower()
            if topic_title in other_title or other_title in topic_title:
                similarity_sum += 1.0
        return 1.0 / (1.0 + similarity_sum)

    def _remove_redundant_topics(self, scored_topics: List[tuple]) -> List[tuple]:
        """
        Remove redundant topics based on similar titles, keeping the highest scored one.
        """
        distinct = {}
        for topic, score in scored_topics:
            title = topic.get("title", "").lower()
            if title not in distinct or score > distinct[title][1]:
                distinct[title] = (topic, score)
        return list(distinct.values())
