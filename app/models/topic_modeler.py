"""
topic_modeler.py - Enhanced topic extraction with better diversity
"""

import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

class TopicModeler:
    def __init__(self, logs_ref):
        self.logs = logs_ref
        self.stop_phrases = {"et al", "et"}
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))
            self.logs["steps"].append("TopicModeler initialized with NLTK resources")
        except Exception as e:
            self.logs["errors"].append(f"Error initializing NLTK: {str(e)}")
            self.stop_words = set()

    def extract_topics(self, paragraphs: List[str], top_n: int = 5) -> List[str]:
        """Extract diverse topics using TF-IDF and NMF."""
        if not paragraphs:
            return ["No topics identified."]

        # Dynamic parameter adjustment based on corpus size
        document_count = len(paragraphs)
        if document_count < 1:
            self.logs["errors"].append("Insufficient documents for topic extraction.")
            return ["No topics identified."]

        max_df = min(0.85, (document_count - 1) / document_count)  # Ensure max_df < 1.0
        min_df = max(1, document_count // 10)  # At least 1 document

        # Log adjusted parameters
        self.logs["steps"].append(f"Adjusted parameters: max_df={max_df}, min_df={min_df}")

        # Combine TF-IDF with NMF for topic extraction
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 3),
            max_df=max_df,
            min_df=min_df
        )

        try:
            # Get TF-IDF matrix
            tfidf_matrix = vectorizer.fit_transform(paragraphs)
            
            # Apply NMF
            nmf_model = NMF(n_components=top_n, random_state=42)
            nmf_output = nmf_model.fit_transform(tfidf_matrix)
            
            # Get feature names
            feature_names = vectorizer.get_feature_names_out()
            
            # Extract topics
            topics = []
            for topic_idx, topic in enumerate(nmf_model.components_):
                top_features_idx = topic.argsort()[:-5:-1]  # Get top 4 terms
                top_features = [feature_names[i] for i in top_features_idx]
                topic_phrase = " ".join(top_features)
                if not self._contains_stop_phrase(topic_phrase):
                    topics.append(topic_phrase)
            
            # Supplement with n-gram analysis
            ngrams = self._generate_ngrams(" ".join(paragraphs))
            
            # Combine and deduplicate topics
            all_topics = topics + ngrams[:top_n]
            final_topics = self._remove_similar_topics(all_topics)
            
            return final_topics[:top_n]

        except Exception as e:
            self.logs["errors"].append(f"Error in topic extraction: {str(e)}")
            return self._fallback_topic_extraction(paragraphs, top_n)

    def build_wordcloud_data(self, paragraphs: List[str]) -> Dict[str, int]:
        """Build frequency data for visualization."""
        text = " ".join(paragraphs).lower()
        tokens = self._simple_tokenize(text)
        
        # Get single word frequencies
        word_freq = Counter(tokens)
        
        # Get phrase frequencies
        phrases = self._generate_ngrams(text, min_n=2, max_n=3)
        phrase_freq = Counter(phrases)
        
        # Combine frequencies with normalization
        max_freq = max(word_freq.values()) if word_freq else 1
        normalized_frequencies = {}
        
        # Add normalized single word frequencies
        for word, freq in word_freq.most_common(50):
            if len(word) > 3 and not self._is_common_word(word):
                normalized_frequencies[word] = (freq / max_freq) * 100
        
        # Add normalized phrase frequencies
        max_phrase_freq = max(phrase_freq.values()) if phrase_freq else 1
        for phrase, freq in phrase_freq.most_common(30):
            if not self._contains_stop_phrase(phrase):
                normalized_frequencies[phrase] = (freq / max_phrase_freq) * 75  # Slightly lower weight for phrases
        
        return dict(sorted(normalized_frequencies.items(), key=lambda x: x[1], reverse=True)[:50])

    def _is_common_word(self, word: str) -> bool:
        """Check if word is too common to be interesting."""
        common_words = {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at'
        }
        return word.lower() in common_words or word.lower() in self.stop_words

    def _remove_similar_topics(self, topics: List[str]) -> List[str]:
        """Remove topics that are too similar to each other."""
        final_topics = []
        for topic in topics:
            if not any(self._is_similar(topic, existing) for existing in final_topics):
                final_topics.append(topic)
        return final_topics

    def _is_similar(self, topic1: str, topic2: str) -> bool:
        """Check if two topics are too similar."""
        words1 = set(topic1.split())
        words2 = set(topic2.split())
        
        # Check word overlap
        overlap = len(words1.intersection(words2))
        return overlap >= min(len(words1), len(words2)) * 0.5

    def _fallback_topic_extraction(self, paragraphs: List[str], top_n: int) -> List[str]:
        """Fallback method for topic extraction using simple n-grams."""
        text = " ".join(paragraphs).lower()
        return self._generate_ngrams(text)[:top_n]

    def _generate_ngrams(self, text: str, min_n: int = 2, max_n: int = 3) -> List[str]:
        """Generate n-grams from text."""
        tokens = self._simple_tokenize(text)
        ngrams = []
        
        for n in range(min_n, max_n + 1):
            for i in range(len(tokens) - n + 1):
                ngram = " ".join(tokens[i:i+n])
                if not self._contains_stop_phrase(ngram):
                    ngrams.append(ngram)
        
        return ngrams

    def _simple_tokenize(self, text: str) -> List[str]:
        """Tokenize text with basic cleaning."""
        text = text.translate(str.maketrans("", "", string.punctuation))
        tokens = word_tokenize(text.lower())
        return [t for t in tokens if len(t) > 1 and t not in self.stop_words]

    def _contains_stop_phrase(self, phrase: str) -> bool:
        """Check if phrase contains unwanted terms."""
        return any(stop in phrase.lower() for stop in self.stop_phrases)
