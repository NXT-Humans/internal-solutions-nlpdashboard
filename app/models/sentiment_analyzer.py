import logging
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self, logs_ref):
        self.logs = logs_ref
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
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

    def analyze(self, paragraphs: List[str]) -> List[Dict]:
        """
        Analyze sentiment of each paragraph, chunking if needed.
        Returns a list of {polarity, label} for each paragraph.
        """
        results = []
        for idx, paragraph in enumerate(paragraphs):
            # If paragraph is too large, break it down further.
            polarity, label = self._analyze_chunked(paragraph)
            results.append({"polarity": polarity, "label": label})
        return results

    def _analyze_chunked(self, text: str):
        """
        Chunk a single paragraph if too large, then average the sentiments.
        """
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) <= self.max_tokens:
            return self._analyze_one_chunk(text)

        chunk_size = 256
        polarities = []
        labels = []
        for i in range(0, len(tokens), chunk_size):
            subset = tokens[i : i + chunk_size]
            chunk_text = self.tokenizer.convert_tokens_to_string(subset)
            pol, lab = self._analyze_one_chunk(chunk_text)
            polarities.append(pol)
            labels.append(lab)

        # Average polarity, majority label
        avg_polarity = sum(polarities) / len(polarities)
        final_label = max(set(labels), key=labels.count)
        return (avg_polarity, final_label)

    def _analyze_one_chunk(self, text_chunk: str):
        """
        Analyze sentiment for a chunk < self.max_tokens.
        """
        try:
            result = self.classifier(text_chunk)
            if result and isinstance(result, list):
                first = result[0]
                # For DistilBERT, "POSITIVE" vs "NEGATIVE"
                label = first["label"]
                score = first["score"]
                polarity = score if label.upper() == "POSITIVE" else -score
                return (polarity, label)
        except Exception as ex:
            logger.error(f"Sentiment error: {ex}", exc_info=True)
            self.logs["errors"].append(f"Sentiment error: {str(ex)}")
        return (0.0, "NEUTRAL")
