"""
topic_modeler.py - Enhanced topic extraction with comprehensive stopwords and improved diversity
"""

import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from typing import List, Dict, Set
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

class TopicModeler:
   def __init__(self, logs_ref):
       self.logs = logs_ref
       # Standard stop phrases 
       self.stop_phrases = {
           "et al", "et", "eg", "ie", "etc", "example", "examples",
           "including", "included", "includes", "include",
           "particularly", "especially", "specifically", 
           "usually", "typically", "generally", "mainly",
           "mostly", "largely", "primarily", "commonly",
           "furthermore", "moreover", "additionally",
           "however", "nevertheless", "nonetheless",
           "therefore", "thus", "hence", "consequently",
           "meanwhile", "subsequently",
           "regarding", "concerning", "considering"
       }

       try:
           nltk.download('punkt', quiet=True)
           nltk.download('stopwords', quiet=True)
           
           # Get base NLTK stopwords
           self.stop_words = set(stopwords.words('english'))
           
           # Add common technical/academic words
           technical_stopwords = {
               'figure', 'fig', 'table', 'appendix', 'chapter',
               'paper', 'study', 'research', 'method', 'analysis', 
               'data', 'results', 'discussion', 'conclusion',
               'introduction', 'background', 'methodology',
               'findings', 'abstract', 'keywords', 'references',
               'et', 'al', 'ie', 'eg', 'cf', 'nb', 'ref'
           }
           self.stop_words.update(technical_stopwords)
           
           self.logs["steps"].append("TopicModeler initialized with enhanced stopwords")
           
       except Exception as error:
           self.logs["errors"].append(f"Error initializing NLTK resources: {str(error)}")
           self.stop_words = set()

   def extract_topics(self, paragraphs: List[str], top_n: int = 5) -> List[str]:
    """Extract diverse topics using TF-IDF and NMF with enhanced preprocessing."""
    if not paragraphs:
        return ["No topics identified."]

    document_count = len(paragraphs)
    if document_count < 1:
        self.logs["errors"].append("Insufficient documents for topic extraction.")
        return ["No topics identified."]

    max_document_frequency = min(0.85, (document_count - 1) / document_count)
    min_document_frequency = max(1, document_count // 10)

    self.logs["steps"].append(
        f"Topic extraction parameters: max_df={max_document_frequency}, min_df={min_document_frequency}"
    )

    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 3),
        max_df=max_document_frequency,
        min_df=min_document_frequency,
        token_pattern=r'(?u)\b[A-Za-z][A-Za-z-]+[A-Za-z]\b'
    )

    try:
        # Generate TF-IDF matrix
        tfidf_matrix = vectorizer.fit_transform(paragraphs)
        
        # Calculate safe number of components
        number_of_components = min(
            top_n * 2,  # Desired number
            tfidf_matrix.shape[0],  # Number of samples
            tfidf_matrix.shape[1],  # Number of features
            len(paragraphs)  # Number of documents
        )
        
        if number_of_components < 1:
            self.logs["errors"].append("Insufficient data for topic extraction")
            return ["No topics identified."]
            
        # Configure and apply NMF
        nmf_model = NMF(
            n_components=number_of_components,
            random_state=42,
            init='nndsvd'
        )
        nmf_output = nmf_model.fit_transform(tfidf_matrix)
        
        # Get feature names from vectorizer
        feature_names = vectorizer.get_feature_names_out()
        
        # Extract topics with filtering
        topics = []
        for topic_index, topic in enumerate(nmf_model.components_):
            top_features_indices = topic.argsort()[:-8:-1]
            top_features = [feature_names[index] for index in top_features_indices]
            
            filtered_features = []
            for feature in top_features:
                if (len(feature) > 2 and 
                    feature.lower() not in self.stop_words and 
                    not self._contains_stop_phrase(feature)):
                    filtered_features.append(feature)
            
            if filtered_features:
                topic_phrase = " ".join(filtered_features[:3])
                if not self._contains_stop_phrase(topic_phrase):
                    topics.append(topic_phrase)
        
        # Generate additional n-gram topics
        ngram_topics = self._generate_ngrams(" ".join(paragraphs))
        
        # Combine and remove duplicates
        all_topics = topics + ngram_topics[:top_n]
        final_topics = self._remove_similar_topics(all_topics)
        
        return final_topics[:top_n]

    except Exception as error:
        self.logs["errors"].append(f"Error in topic extraction: {str(error)}")
        return self._fallback_topic_extraction(paragraphs, top_n)

   def build_wordcloud_data(self, paragraphs: List[str]) -> Dict[str, int]:
       """Build frequency data for visualization with filtering."""
       if not paragraphs:
           return {}
           
       text = " ".join(paragraphs).lower()
       tokens = self._simple_tokenize(text)
       
       # Get filtered word frequencies
       word_frequencies = Counter([
           token for token in tokens 
           if len(token) > 3 and 
           token not in self.stop_words and 
           not self._is_common_word(token)
       ])
       
       # Get filtered phrase frequencies
       phrases = self._generate_ngrams(text, min_n=2, max_n=3)
       phrase_frequencies = Counter(phrases)
       
       # Normalize and combine frequencies
       maximum_word_frequency = max(word_frequencies.values()) if word_frequencies else 1
       normalized_frequencies = {}
       
       # Add normalized word frequencies
       for word, frequency in word_frequencies.most_common(50):
           if len(word) > 3 and not self._is_common_word(word):
               normalized_frequencies[word] = (frequency / maximum_word_frequency) * 100
       
       # Add normalized phrase frequencies
       maximum_phrase_frequency = max(phrase_frequencies.values()) if phrase_frequencies else 1
       for phrase, frequency in phrase_frequencies.most_common(30):
           if not self._contains_stop_phrase(phrase):
               normalized_frequencies[phrase] = (frequency / maximum_phrase_frequency) * 75
       
       return dict(sorted(
           normalized_frequencies.items(), 
           key=lambda item: item[1], 
           reverse=True
       )[:50])

   def _is_common_word(self, word: str) -> bool:
       """Check if word is too common or uninteresting."""
       return (
           word.lower() in self.stop_words or
           len(word) < 3 or 
           word.isdigit()
       )

   def _remove_similar_topics(self, topics: List[str]) -> List[str]:
       """Remove topics that are too similar to each other."""
       final_topics = []
       for topic in topics:
           if not any(self._is_similar(topic, existing) for existing in final_topics):
               final_topics.append(topic)
       return final_topics

   def _is_similar(self, topic1: str, topic2: str) -> bool:
       """Check if two topics are too similar using overlap analysis."""
       words1 = set(topic1.lower().split())
       words2 = set(topic2.lower().split())
       
       # Check direct containment
       if words1.issubset(words2) or words2.issubset(words1):
           return True
       
       # Check word overlap
       overlap = len(words1.intersection(words2))
       smaller_size = min(len(words1), len(words2))
       threshold = 0.5 if smaller_size <= 2 else 0.7
       
       return overlap >= smaller_size * threshold

   def _fallback_topic_extraction(self, paragraphs: List[str], top_n: int) -> List[str]:
       """Fallback method for topic extraction using n-grams."""
       text = " ".join(paragraphs).lower()
       ngrams = self._generate_ngrams(text)
       
       filtered_ngrams = [
           ngram for ngram in ngrams
           if not self._contains_stop_phrase(ngram) and
           not all(word in self.stop_words for word in ngram.split())
       ]
       
       return filtered_ngrams[:top_n]

   def _generate_ngrams(self, text: str, min_n: int = 2, max_n: int = 3) -> List[str]:
       """Generate n-grams from text with filtering."""
       tokens = self._simple_tokenize(text)
       ngrams = []
       
       for n in range(min_n, max_n + 1):
           for index in range(len(tokens) - n + 1):
               ngram = " ".join(tokens[index:index + n])
               if (not self._contains_stop_phrase(ngram) and
                   not all(token in self.stop_words for token in tokens[index:index + n])):
                   ngrams.append(ngram)
       
       return ngrams

   def _simple_tokenize(self, text: str) -> List[str]:
       """Tokenize text with cleaning and filtering."""
       text = text.translate(str.maketrans("", "", string.punctuation))
       tokens = word_tokenize(text.lower())
       
       return [
           token for token in tokens
           if (len(token) > 1 and
               token not in self.stop_words and
               not token.isdigit() and
               not all(char in string.punctuation for char in token))
       ]

   def _contains_stop_phrase(self, phrase: str) -> bool:
       """Check if phrase contains unwanted terms or patterns."""
       phrase_lower = phrase.lower()
       
       if any(stop in phrase_lower for stop in self.stop_phrases):
           return True
           
       words = phrase_lower.split()
       if (len(words) == 1 and len(phrase_lower) < 3) or all(word in self.stop_words for word in words):
           return True
           
       return False