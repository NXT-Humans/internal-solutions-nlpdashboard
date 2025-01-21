"""
summarizer.py - Enhanced summarization with better structure and conciseness
"""

from transformers import pipeline
from typing import List, Dict

class Summarizer:
    def __init__(self, logs_ref):
        self.logs = logs_ref
        try:
            self.summarization_pipeline = pipeline(
                "summarization",
                model="sshleifer/distilbart-cnn-12-6",
                max_length=130,
                min_length=30,
                do_sample=False
            )
            self.logs["steps"].append("Summarizer model loaded successfully.")
        except Exception as e:
            self.logs["errors"].append(f"Error loading summarizer model: {str(e)}")
            self.summarization_pipeline = None

    def summarize_paragraphs(self, paragraphs: List[str], top_topics: List[str] = None) -> str:
        """
        Creates a structured summary with key findings and implications.
        """
        if not paragraphs:
            return "No text available for summarization."

        if not self.summarization_pipeline:
            return "Summarizer pipeline is unavailable."

        combined_text = " ".join(paragraphs)
        chunks = self._split_text_into_chunks(combined_text)

        # Get initial summaries for key findings
        key_findings = []
        for idx, chunk in enumerate(chunks):
            self.logs["steps"].append(f"Summarizing chunk {idx+1} of {len(chunks)}.")
            try:
                result = self.summarization_pipeline(
                    chunk,
                    max_length=40,
                    min_length=15,
                    do_sample=False
                )
                key_findings.append(result[0]["summary_text"])
            except Exception as e:
                self.logs["errors"].append(f"Summarization error: {str(e)}")

        # Structure the summary
        structured_summary = []
        
        # Key Findings section
        structured_summary.append("Key Findings:")
        for i, finding in enumerate(key_findings[:3], 1):
            structured_summary.append(f"{i}. {finding}")
        
        # Implications section
        structured_summary.append("\nImplications:")
        implications = self._extract_implications(chunks)
        for i, implication in enumerate(implications[:2], 1):
            structured_summary.append(f"{i}. {implication}")
        
        # Recommendations section if available
        if len(chunks) > 2:
            structured_summary.append("\nRecommendations:")
            recommendations = self._extract_recommendations(chunks)
            for i, rec in enumerate(recommendations[:2], 1):
                structured_summary.append(f"{i}. {rec}")

        return "\n".join(structured_summary)

    def _extract_implications(self, chunks: List[str]) -> List[str]:
        """Extract implications from the text."""
        implications = []
        for chunk in chunks:
            if any(word in chunk.lower() for word in ["impact", "effect", "result", "lead to", "cause"]):
                try:
                    result = self.summarization_pipeline(
                        chunk,
                        max_length=30,
                        min_length=15,
                        do_sample=False
                    )
                    implications.append(result[0]["summary_text"])
                except Exception:
                    continue
        return implications

    def _extract_recommendations(self, chunks: List[str]) -> List[str]:
        """Extract recommendations from the text."""
        recommendations = []
        for chunk in chunks:
            if any(word in chunk.lower() for word in ["should", "must", "need to", "recommend", "suggest"]):
                try:
                    result = self.summarization_pipeline(
                        chunk,
                        max_length=30,
                        min_length=15,
                        do_sample=False
                    )
                    recommendations.append(result[0]["summary_text"])
                except Exception:
                    continue
        return recommendations

    def _split_text_into_chunks(self, text: str, chunk_size: int = 800) -> List[str]:
        """Split text into manageable chunks for the summarizer."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= chunk_size:
                current_chunk.append(word)
                current_length += len(word) + 1
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
                
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks