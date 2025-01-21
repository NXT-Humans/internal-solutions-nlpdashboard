"""
visualize.py

Router for retrieving and formatting analysis results for visualization.
Handles data preparation for charts, word clouds, and interactive visualizations.
"""

import os
import json
import logging
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional

from app.log_store import PROCESSING_LOGS, PIPELINE_RESULTS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["Visualize"])

@router.get("/results")
async def get_visualization_data(
    include_raw: bool = Query(False, description="Include raw text data"),
    max_topics: int = Query(10, description="Maximum number of topics to return"),
    max_wordcloud: int = Query(50, description="Maximum number of terms in word cloud"),
    sentiment_threshold: float = Query(0.2, description="Sentiment significance threshold")
) -> Dict[str, Any]:
    """
    Retrieve and format analysis results for visualization.
    
    Args:
        include_raw: Whether to include raw text data
        max_topics: Maximum number of topics to return
        max_wordcloud: Maximum number of terms in word cloud
        sentiment_threshold: Threshold for sentiment classification
        
    Returns:
        Formatted visualization data
    """
    if not PIPELINE_RESULTS:
        raise HTTPException(
            status_code=404,
            detail="No analysis results available. Please run analysis first."
        )

    try:
        visualization_data = {
            "summary": format_summary_data(PIPELINE_RESULTS.get("global_summary", "")),
            "topics": format_topic_data(
                PIPELINE_RESULTS.get("global_topics", []),
                max_topics=max_topics
            ),
            "wordcloud": format_wordcloud_data(
                PIPELINE_RESULTS.get("global_wordcloud_data", {}),
                max_terms=max_wordcloud
            ),
            "sentiment": format_sentiment_data(
                PIPELINE_RESULTS.get("documents", {}),
                threshold=sentiment_threshold
            ),
            "metadata": PIPELINE_RESULTS.get("metadata", {})
        }

        if include_raw:
            visualization_data["raw_data"] = {
                "documents": {
                    doc_name: {
                        k: v for k, v in doc_data.items()
                        if k not in ["raw_paragraphs", "cleaned_paragraphs"]
                    }
                    for doc_name, doc_data in PIPELINE_RESULTS.get("documents", {}).items()
                }
            }

        return visualization_data

    except Exception as error:
        logger.error(f"Error preparing visualization data: {str(error)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(error))

def format_summary_data(summary_text: str) -> Dict[str, Any]:
    """Format summary text into structured sections"""
    sections = {
        "key_threats": [],
        "current_landscape": [],
        "defense_strategies": []
    }
    
    # Identify section markers
    markers = {
        "key_threats": ["Key Threats:", "Threats:", "Risks:"],
        "current_landscape": ["Key Findings:", "Overview:", "Current Landscape:"],
        "defense_strategies": ["Defense Strategies:", "Recommendations:", "Solutions:"]
    }
    
    current_section = "current_landscape"  # Default section
    
    for line in summary_text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Check if line starts a new section
        for section, section_markers in markers.items():
            if any(line.startswith(marker) for marker in section_markers):
                current_section = section
                line = line.split(':', 1)[1].strip()
                break
                
        if line:
            sections[current_section].append(line)
    
    return {
        "sections": sections,
        "full_text": summary_text
    }

def format_topic_data(topics: List[str], max_topics: int = 10) -> List[Dict[str, Any]]:
    """Format topic data for visualization"""
    formatted_topics = []
    
    for topic in topics[:max_topics]:
        # Get topic frequency and sentiment if available
        topic_data = PIPELINE_RESULTS.get("documents", {})
        frequency = sum(
            1 for doc in topic_data.values()
            if topic in doc.get("paragraph_topics", [])
        )
        
        formatted_topics.append({
            "text": topic,
            "frequency": max(frequency, 1),  # Ensure minimum frequency of 1
            "sentiment": get_topic_sentiment(topic, topic_data)
        })
    
    return formatted_topics

def format_wordcloud_data(wordcloud_data: Dict[str, Any], max_terms: int = 50) -> Dict[str, Dict[str, Any]]:
    """Format word cloud data for visualization"""
    if not wordcloud_data:
        return {}
        
    # Sort terms by frequency
    sorted_terms = sorted(
        wordcloud_data.items(),
        key=lambda x: x[1].get("frequency", 0),
        reverse=True
    )
    
    # Take top N terms
    return {
        term: {
            "frequency": data.get("frequency", 0),
            "sentiment": data.get("sentiment", 0),
            "contexts": data.get("contexts", [])[:5]  # Limit to 5 contexts per term
        }
        for term, data in sorted_terms[:max_terms]
    }

def format_sentiment_data(documents: Dict[str, Any], threshold: float = 0.2) -> Dict[str, Any]:
    """Format sentiment analysis data for visualization"""
    sentiment_data = {
        "distribution": {
            "positive": 0,
            "negative": 0,
            "neutral": 0
        },
        "by_document": {},
        "overall_score": 0.0
    }
    
    total_paragraphs = 0
    total_sentiment = 0.0
    
    for doc_name, doc_data in documents.items():
        doc_sentiments = []
        
        for sentiment in doc_data.get("paragraph_sentiments", []):
            polarity = sentiment.get("polarity", 0)
            total_sentiment += polarity
            total_paragraphs += 1
            
            # Categorize sentiment
            if polarity > threshold:
                sentiment_data["distribution"]["positive"] += 1
                category = "positive"
            elif polarity < -threshold:
                sentiment_data["distribution"]["negative"] += 1
                category = "negative"
            else:
                sentiment_data["distribution"]["neutral"] += 1
                category = "neutral"
                
            doc_sentiments.append({
                "score": polarity,
                "category": category,
                "text": sentiment.get("text", "")[:200]  # Limit text length
            })
            
        sentiment_data["by_document"][doc_name] = doc_sentiments
    
    if total_paragraphs > 0:
        sentiment_data["overall_score"] = total_sentiment / total_paragraphs
    
    return sentiment_data

def get_topic_sentiment(topic: str, documents: Dict[str, Any]) -> float:
    """Calculate average sentiment for a topic"""
    sentiments = []
    
    for doc_data in documents.values():
        for sentiment in doc_data.get("paragraph_sentiments", []):
            if topic.lower() in sentiment.get("text", "").lower():
                sentiments.append(sentiment.get("polarity", 0))
    
    return sum(sentiments) / len(sentiments) if sentiments else 0.0

@router.get("/download-results")
async def download_results(
    format: str = Query("json", description="Output format (json/csv)")
) -> Dict[str, Any]:
    """
    Prepare analysis results for download
    
    Args:
        format: Desired output format (json/csv)
        
    Returns:
        Download URL and metadata
    """
    if not PIPELINE_RESULTS:
        raise HTTPException(
            status_code=404,
            detail="No results available for download"
        )
    
    try:
        timestamp = PROCESSING_LOGS.get("completion_time", "").replace(":", "-")
        filename = f"analysis_results_{timestamp}.{format}"
        
        # Save results to file
        save_path = os.path.join("downloads", filename)
        os.makedirs("downloads", exist_ok=True)
        
        if format == "json":
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(PIPELINE_RESULTS, f, indent=2, ensure_ascii=False)
        else:
            # Implement CSV export if needed
            raise HTTPException(status_code=400, detail="CSV format not yet supported")
        
        return {
            "download_url": f"/downloads/{filename}",
            "filename": filename,
            "format": format,
            "size": os.path.getsize(save_path),
            "timestamp": timestamp
        }
        
    except Exception as error:
        logger.error(f"Error preparing download: {str(error)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(error))