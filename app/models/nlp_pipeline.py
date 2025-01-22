"""
nlp_pipeline.py

Coordinates the overall NLP sequence with enhanced logging and status tracking:
  1) Clean data with progress tracking
  2) Sentiment analysis with error handling
  3) Topic extraction with detailed substeps
  4) Summaries with status updates
  5) Word cloud data with completion tracking
"""

from typing import Dict, List, Any
from datetime import datetime
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
        
        # Initialize stage tracking
        self.logs["stage_details"] = {
            "cleaning": {"status": "pending", "started": None, "completed": None, "errors": []},
            "sentiment": {"status": "pending", "started": None, "completed": None, "errors": []},
            "topics": {"status": "pending", "started": None, "completed": None, "errors": []},
            "summary": {"status": "pending", "started": None, "completed": None, "errors": []},
            "wordcloud": {"status": "pending", "started": None, "completed": None, "errors": []}
        }
        self.logs["files_processed"] = {}

    def process(self, texts_by_doc: Dict[str, List[str]], **options) -> Dict:
        """
        Process documents with enhanced logging and status tracking.

        Args:
            texts_by_doc: { 'filename': [paragraph1, paragraph2, ...], ... }
            options: Optional processing parameters

        Returns:
            Dictionary containing analysis results and document-specific data
        """
        try:
            # Start pipeline
            self.logs["start_time"] = datetime.now().isoformat()
            self.logs["steps"].append("Starting NLP pipeline processing")
            self.logs["current_stage"] = "initialization"

            result = {
                "documents": {},
                "global_summary": "",
                "global_topics": [],
                "global_wordcloud_data": {},
                "metadata": {
                    "start_time": self.logs["start_time"],
                    "document_count": len(texts_by_doc),
                    "total_paragraphs": sum(len(paragraphs) for paragraphs in texts_by_doc.values())
                }
            }

            # Clean data
            self.logs["current_stage"] = "cleaning"
            self.logs["stage_details"]["cleaning"]["status"] = "in_progress"
            self.logs["stage_details"]["cleaning"]["started"] = datetime.now().isoformat()
            self.logs["steps"].append("Starting document cleaning")

            cleaned_data = {}
            for doc_name, paragraphs in texts_by_doc.items():
                try:
                    self.logs["files_processed"][doc_name] = {"stage": "cleaning", "status": "in_progress"}
                    cleaned_data[doc_name] = self.cleaner.clean_paragraphs(paragraphs)
                    self.logs["files_processed"][doc_name] = {"stage": "cleaning", "status": "complete"}
                except Exception as e:
                    error_msg = f"Error cleaning document {doc_name}: {str(e)}"
                    self.logs["errors"].append(error_msg)
                    self.logs["stage_details"]["cleaning"]["errors"].append(error_msg)
                    self.logs["files_processed"][doc_name] = {"stage": "cleaning", "status": "error", "error": str(e)}

            self.logs["stage_details"]["cleaning"]["completed"] = datetime.now().isoformat()
            self.logs["stage_details"]["cleaning"]["status"] = "complete"

            # Sentiment Analysis
            self.logs["current_stage"] = "sentiment"
            self.logs["stage_details"]["sentiment"]["status"] = "in_progress"
            self.logs["stage_details"]["sentiment"]["started"] = datetime.now().isoformat()
            self.logs["steps"].append("Performing sentiment analysis")

            sentiments = {}
            for doc_name, paragraphs in cleaned_data.items():
                try:
                    self.logs["files_processed"][doc_name] = {"stage": "sentiment", "status": "in_progress"}
                    sentiments[doc_name] = self.sentiment_analyzer.analyze(paragraphs)
                    self.logs["files_processed"][doc_name] = {"stage": "sentiment", "status": "complete"}
                except Exception as e:
                    error_msg = f"Error analyzing sentiment in {doc_name}: {str(e)}"
                    self.logs["errors"].append(error_msg)
                    self.logs["stage_details"]["sentiment"]["errors"].append(error_msg)
                    self.logs["files_processed"][doc_name] = {"stage": "sentiment", "status": "error", "error": str(e)}

            self.logs["stage_details"]["sentiment"]["completed"] = datetime.now().isoformat()
            self.logs["stage_details"]["sentiment"]["status"] = "complete"

            # Topic Extraction
            self.logs["current_stage"] = "topics"
            self.logs["stage_details"]["topics"]["status"] = "in_progress"
            self.logs["stage_details"]["topics"]["started"] = datetime.now().isoformat()
            self.logs["steps"].append("Extracting topics from documents")

            doc_topics = {}
            for doc_name, paragraphs in cleaned_data.items():
                try:
                    self.logs["files_processed"][doc_name] = {"stage": "topics", "status": "in_progress"}
                    doc_topics[doc_name] = self.topic_modeler.extract_topics(
                        paragraphs, 
                        top_n=options.get("max_topics", 5)
                    )
                    self.logs["files_processed"][doc_name] = {"stage": "topics", "status": "complete"}
                except Exception as e:
                    error_msg = f"Error extracting topics from {doc_name}: {str(e)}"
                    self.logs["errors"].append(error_msg)
                    self.logs["stage_details"]["topics"]["errors"].append(error_msg)
                    self.logs["files_processed"][doc_name] = {"stage": "topics", "status": "error", "error": str(e)}

            self.logs["stage_details"]["topics"]["completed"] = datetime.now().isoformat()
            self.logs["stage_details"]["topics"]["status"] = "complete"

            # Summarization
            self.logs["current_stage"] = "summary"
            self.logs["stage_details"]["summary"]["status"] = "in_progress"
            self.logs["stage_details"]["summary"]["started"] = datetime.now().isoformat()
            self.logs["steps"].append("Generating document summaries")

            doc_summaries = {}
            for doc_name, paragraphs in cleaned_data.items():
                try:
                    self.logs["files_processed"][doc_name] = {"stage": "summary", "status": "in_progress"}
                    doc_summaries[doc_name] = self.summarizer.summarize_paragraphs(
                        paragraphs,
                        doc_topics.get(doc_name, [])
                    )
                    self.logs["files_processed"][doc_name] = {"stage": "summary", "status": "complete"}
                except Exception as e:
                    error_msg = f"Error summarizing {doc_name}: {str(e)}"
                    self.logs["errors"].append(error_msg)
                    self.logs["stage_details"]["summary"]["errors"].append(error_msg)
                    self.logs["files_processed"][doc_name] = {"stage": "summary", "status": "error", "error": str(e)}

            self.logs["stage_details"]["summary"]["completed"] = datetime.now().isoformat()
            self.logs["stage_details"]["summary"]["status"] = "complete"

            # Global Analysis
            self.logs["steps"].append("Performing global analysis")
            all_text = []
            for doc_name, paragraphs in cleaned_data.items():
                all_text.extend(paragraphs)

            # Global Summary
            self.logs["steps"].append("Generating global summary")
            result["global_summary"] = self.summarizer.summarize_paragraphs(all_text)

            # Global Topics
            self.logs["steps"].append("Extracting global topics")
            result["global_topics"] = self.topic_modeler.extract_topics(
                all_text, 
                top_n=options.get("max_topics", 5)
            )

            # Word Cloud Data
            self.logs["current_stage"] = "wordcloud"
            self.logs["stage_details"]["wordcloud"]["status"] = "in_progress"
            self.logs["stage_details"]["wordcloud"]["started"] = datetime.now().isoformat()
            self.logs["steps"].append("Building word cloud data")

            try:
                result["global_wordcloud_data"] = self.topic_modeler.build_wordcloud_data(all_text)
                self.logs["stage_details"]["wordcloud"]["status"] = "complete"
            except Exception as e:
                error_msg = "Error generating word cloud data: {str(e)}"
                self.logs["errors"].append(error_msg)
                self.logs["stage_details"]["wordcloud"]["errors"].append(error_msg)
                self.logs["stage_details"]["wordcloud"]["status"] = "error"

            self.logs["stage_details"]["wordcloud"]["completed"] = datetime.now().isoformat()

            # Compile Document Results
            for doc_name in cleaned_data:
                result["documents"][doc_name] = {
                    "cleaned_paragraphs": cleaned_data[doc_name],
                    "paragraph_sentiments": sentiments.get(doc_name, []),
                    "paragraph_topics": doc_topics.get(doc_name, []),
                    "summary": doc_summaries.get(doc_name, ""),
                    "metadata": {
                        "paragraphs": len(cleaned_data[doc_name]),
                        "processing_status": self.logs["files_processed"].get(doc_name, {})
                    }
                }

            # Complete Pipeline
            self.logs["completion_time"] = datetime.now().isoformat()
            result["metadata"]["completion_time"] = self.logs["completion_time"]
            result["metadata"]["processing_duration"] = (
                datetime.fromisoformat(self.logs["completion_time"]) - 
                datetime.fromisoformat(self.logs["start_time"])
            ).total_seconds()
            
            self.logs["steps"].append("NLP pipeline completed successfully")
            return result

        except Exception as e:
            error_msg = f"Critical error in NLP pipeline: {str(e)}"
            self.logs["errors"].append(error_msg)
            self.logs["completion_time"] = datetime.now().isoformat()
            raise Exception(error_msg)