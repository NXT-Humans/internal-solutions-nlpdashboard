"""
nlp_pipeline.py

Coordinates the overall NLP sequence:
  1) Clean data
  2) Sentiment analysis
  3) Topic extraction (multi-word n-grams, merges partial duplicates)
  4) Summaries
  5) Word cloud data
"""

from typing import Dict, List
from app.models.summarizer import Summarizer
from app.models.topic_modeler import TopicModeler
from app.models.sentiment_analyzer import SentimentAnalyzer
from app.utils.data_cleaner import DataCleaner

class NLPPipeline:
    def __init__(self, logs_ref):
        self.logs = logs_ref
        self.summarizer = Summarizer(logs_ref)
        self.topic_modeler = TopicModeler(logs_ref)
        self.sentiment_analyzer = SentimentAnalyzer(logs_ref)
        self.cleaner = DataCleaner(logs_ref)

    def process(self, texts_by_doc: Dict[str, List[str]]) -> Dict:
        """
        texts_by_doc: { 'filename': [paragraph1, paragraph2, ...], ... }

        Returns a dictionary:
        {
          "documents": {
            filename: {
              "cleaned_paragraphs": [...],
              "paragraph_sentiments": [...],
              "paragraph_topics": [...],
              "summary": "..."
            },
            ...
          },
          "global_summary": "...",
          "global_topics": [...],
          "global_wordcloud_data": {...}
        }
        """
        self.logs["steps"].append("Starting NLP pipeline: cleaning paragraphs.")
        cleaned_data = {}
        for doc_name, paragraphs in texts_by_doc.items():
            cleaned_data[doc_name] = self.cleaner.clean_paragraphs(paragraphs)

        self.logs["steps"].append("Performing sentiment analysis.")
        sentiments = {}
        for doc_name, paragraphs in cleaned_data.items():
            sentiments[doc_name] = self.sentiment_analyzer.analyze(paragraphs)

        self.logs["steps"].append("Extracting multi-word topics from paragraphs.")
        doc_topics = {}
        for doc_name, paragraphs in cleaned_data.items():
            doc_topics[doc_name] = self.topic_modeler.extract_topics(paragraphs, top_n=5)

        self.logs["steps"].append("Generating document-level summaries.")
        doc_summaries = {}
        for doc_name, paragraphs in cleaned_data.items():
            # Optionally pass doc_topics as sub-headers if you want
            doc_summaries[doc_name] = self.summarizer.summarize_paragraphs(paragraphs)

        result = {
            "documents": {},
            "global_summary": "",
            "global_topics": [],
            "global_wordcloud_data": {}
        }

        # Build global summary, topics, wordcloud from combined text
        self.logs["steps"].append("Combining all paragraphs for global analysis.")
        all_text = []
        for doc_name, paragraphs in cleaned_data.items():
            all_text.extend(paragraphs)

        self.logs["steps"].append("Generating global summary.")
        # Optionally pass global topics as sub-headers for final summary
        global_summary = self.summarizer.summarize_paragraphs(
            all_text
        )

        self.logs["steps"].append("Extracting global multi-word topics.")
        global_topics = self.topic_modeler.extract_topics(all_text, top_n=5)

        self.logs["steps"].append("Building global n-gram word cloud data.")
        wordcloud_data = self.topic_modeler.build_wordcloud_data(all_text)

        result["global_summary"] = global_summary
        result["global_topics"] = global_topics
        result["global_wordcloud_data"] = wordcloud_data

        for doc_name in cleaned_data:
            result["documents"][doc_name] = {
                "cleaned_paragraphs": cleaned_data[doc_name],
                "paragraph_sentiments": sentiments[doc_name],
                "paragraph_topics": doc_topics[doc_name],
                "summary": doc_summaries[doc_name]
            }

        self.logs["steps"].append("NLP pipeline completed.")
        return result