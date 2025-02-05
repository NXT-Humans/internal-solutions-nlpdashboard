"""
visualize.py

Enhanced router for retrieving and formatting analysis results for visualization.
Handles enriched data structure with topic relationships, sentiment analysis,
and structured summaries.
"""

import os
import json
import logging
import re
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
from datetime import datetime

from app.log_store import PROCESSING_LOGS, PIPELINE_RESULTS

logger = logging.getLogger(__name__)
router = APIRouter(prefix="", tags=["Visualize"])

@router.get("/results")
async def get_visualization_data(
    include_raw: bool = Query(False, description="Include raw text data"),
    max_topics: int = Query(20, description="Maximum number of topics to return"),
    max_wordcloud: int = Query(100, description="Maximum number of terms in word cloud"),
    sentiment_threshold: float = Query(0.1, description="Sentiment significance threshold"),
    min_topic_score: float = Query(0.01, description="Minimum topic relevance score")
) -> Dict[str, Any]:
    """
    Retrieve and format enhanced analysis results for visualization.
    
    Args:
        include_raw: Whether to include raw text data
        max_topics: Maximum number of topics to return
        max_wordcloud: Maximum number of terms in word cloud
        sentiment_threshold: Threshold for sentiment classification
        min_topic_score: Minimum topic relevance score
        
    Returns:
        Formatted visualization data with enriched features
    """
    if not PIPELINE_RESULTS:
        raise HTTPException(
            status_code=404,
            detail="No analysis results available. Please run analysis first."
        )

    try:
        # Format summary sections
        summary_data = format_summary_data(
            PIPELINE_RESULTS.get("global_summary", {})
        )

        # Format topic data with relationships
        topics_data = format_topic_data(
            PIPELINE_RESULTS.get("global_topics", []),
            max_topics=max_topics,
            min_score=min_topic_score
        )

        # Format word cloud data with sentiment
        wordcloud_data = format_wordcloud_data(
            PIPELINE_RESULTS.get("global_wordcloud_data", {}),
            max_terms=max_wordcloud,
            sentiment_threshold=sentiment_threshold
        )

        # Format document data
        documents_data = format_document_data(
            PIPELINE_RESULTS.get("documents", {}),
            include_raw=include_raw
        )

        # Prepare metadata
        metadata = prepare_metadata(PIPELINE_RESULTS.get("metadata", {}))

        return {
            "summary": summary_data,
            "topics": topics_data,
            "wordcloud": wordcloud_data,
            "documents": documents_data,
            "metadata": metadata
        }

    except Exception as error:
        logger.error(f"Error preparing visualization data: {str(error)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(error))

def format_summary_data(summary: Dict[str, str]) -> Dict[str, Any]:
    """Format enhanced summary sections with metadata"""
    if not summary:
        return {
            "overview": "No analysis available.",
            "findings": "No findings available.",
            "challenges": "No challenges identified.",
            "solutions": "No solutions proposed."
        }

    # Extract key metrics if present in the overview
    metrics_pattern = r'(\d+(?:\.\d+)?%|\d+(?:\.\d+)?x|\d+(?:\.\d+)?-fold)'
    overview = summary.get("overview", "")
    metrics = re.findall(metrics_pattern, overview)

    return {
        "overview": {
            "text": overview,
            "metrics": metrics
        },
        "findings": {
            "text": summary.get("findings", "No findings available."),
            "metrics": re.findall(metrics_pattern, summary.get("findings", ""))
        },
        "challenges": {
            "text": summary.get("challenges", "No challenges identified."),
            "categories": extract_challenge_categories(summary.get("challenges", ""))
        },
        "solutions": {
            "text": summary.get("solutions", "No solutions proposed."),
            "approaches": extract_solution_approaches(summary.get("solutions", ""))
        }
    }

def format_topic_data(topics: List[Dict[str, Any]], max_topics: int = 20, min_score: float = 0.01) -> List[Dict[str, Any]]:
    """Format enhanced topic data with relationships and metadata"""
    if not topics:
        return []

    # Filter and sort topics
    filtered_topics = [
        topic for topic in topics
        if topic.get("score", 0) >= min_score
    ]
    sorted_topics = sorted(filtered_topics, key=lambda x: x.get("score", 0), reverse=True)
    top_topics = sorted_topics[:max_topics]

    # Format each topic
    formatted_topics = []
    for topic in top_topics:
        formatted_topic = {
            "text": topic["text"],
            "score": float(topic.get("score", 0)),
            "frequency": topic.get("frequency", 1),
            "sentiment": topic.get("sentiment", "neutral"),
            "category": topic.get("category", "general"),
            "contexts": [
                {
                    "text": ctx["text"][:200],
                    "sentiment": ctx.get("sentiment", 0)
                }
                for ctx in topic.get("contexts", [])[:3]
            ],
            "related_terms": topic.get("related_terms", [])[:5],
            "related_topics": [
                {
                    "text": rel["text"],
                    "strength": float(rel["strength"])
                }
                for rel in topic.get("related_topics", [])[:3]
            ]
        }
        formatted_topics.append(formatted_topic)

    return formatted_topics

def format_wordcloud_data(wordcloud: Dict[str, Any], max_terms: int = 100, sentiment_threshold: float = 0.1) -> Dict[str, Any]:
    """Format enhanced word cloud data with relationships"""
    if not wordcloud:
        return {}

    formatted_data = {}
    
    # Sort terms by frequency
    sorted_terms = sorted(
        wordcloud.items(),
        key=lambda x: x[1].get("frequency", 0) if isinstance(x[1], dict) else x[1],
        reverse=True
    )

    # Format top terms
    for term, data in sorted_terms[:max_terms]:
        if isinstance(data, dict):
            formatted_data[term] = {
                "frequency": data.get("frequency", 1),
                "sentiment": data.get("sentiment", "neutral"),
                "category": data.get("category", "general"),
                "related_terms": data.get("related_terms", [])[:5],
                "contexts": [
                    {
                        "text": ctx["text"][:200],
                        "sentiment": ctx.get("sentiment", 0)
                    }
                    for ctx in data.get("contexts", [])[:3]
                ]
            }
        else:
            # Handle legacy format
            formatted_data[term] = {
                "frequency": float(data),
                "sentiment": "neutral",
                "category": "general"
            }

    return formatted_data

def format_document_data(documents: Dict[str, Any], include_raw: bool = False) -> Dict[str, Any]:
    """Format document data with enhanced features"""
    formatted_docs = {}

    for doc_name, doc_data in documents.items():
        formatted_doc = {
            "summary": format_summary_data(doc_data.get("summary", {})),
            "topics": format_topic_data(doc_data.get("topics", []), max_topics=10),
            "sentiment_analysis": format_sentiment_data(doc_data.get("sentiments", [])),
            "concepts": format_concept_data(doc_data.get("concepts", [])),
            "metadata": {
                "paragraphs": len(doc_data.get("paragraphs", [])),
                "status": doc_data.get("status", {})
            }
        }

        if include_raw:
            formatted_doc["paragraphs"] = doc_data.get("paragraphs", [])

        formatted_docs[doc_name] = formatted_doc

    return formatted_docs

def format_sentiment_data(sentiments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Format enhanced sentiment analysis data"""
    if not sentiments:
        return {"overall": "neutral", "distribution": {}, "details": []}

    # Calculate overall sentiment
    polarities = [s.get("polarity", 0) for s in sentiments]
    overall_sentiment = sum(polarities) / len(polarities) if polarities else 0

    # Calculate distribution
    distribution = {
        "positive": len([s for s in sentiments if s.get("polarity", 0) > 0.1]),
        "negative": len([s for s in sentiments if s.get("polarity", 0) < -0.1]),
        "neutral": len([s for s in sentiments if -0.1 <= s.get("polarity", 0) <= 0.1])
    }

    # Format details
    details = [
        {
            "text": s.get("text", "")[:200],
            "polarity": s.get("polarity", 0),
            "subjectivity": s.get("subjectivity", 0),
            "label": s.get("label", "neutral"),
            "sentences": s.get("sentences", [])
        }
        for s in sentiments
    ]

    return {
        "overall": "positive" if overall_sentiment > 0.1 else "negative" if overall_sentiment < -0.1 else "neutral",
        "overall_score": float(overall_sentiment),
        "distribution": distribution,
        "details": details
    }

def format_concept_data(concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format extracted concepts data"""
    formatted_concepts = []
    
    for concept in concepts:
        formatted_concepts.append({
            "text": concept["text"],
            "type": concept.get("type", "general"),
            "context": concept.get("context", "")[:200]
        })

    return formatted_concepts

def extract_challenge_categories(challenges_text: str) -> List[str]:
    """Extract challenge categories from text"""
    categories = set()
    challenge_patterns = {
        "technical": r"technical|implementation|performance|scaling",
        "data": r"data quality|data availability|dataset|training data",
        "resource": r"computational|memory|processing|storage",
        "methodology": r"approach|method|algorithm|technique"
    }

    for category, pattern in challenge_patterns.items():
        if re.search(pattern, challenges_text, re.I):
            categories.add(category)

    return list(categories) if categories else ["general"]

def extract_solution_approaches(solutions_text: str) -> List[str]:
    """Extract solution approaches from text"""
    approaches = set()
    solution_patterns = {
        "algorithmic": r"algorithm|method|approach|technique",
        "architectural": r"architecture|design|structure|framework",
        "optimization": r"optimize|improve|enhance|tune",
        "integration": r"integrate|combine|merge|incorporate"
    }

    for approach, pattern in solution_patterns.items():
        if re.search(pattern, solutions_text, re.I):
            approaches.add(approach)

    return list(approaches) if approaches else ["general"]

def prepare_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare enhanced metadata"""
    return {
        "document_count": metadata.get("document_count", 0),
        "total_paragraphs": metadata.get("total_paragraphs", 0),
        "processing_duration": metadata.get("processing_duration", 0),
        "start_time": metadata.get("start_time", ""),
        "completion_time": metadata.get("completion_time", ""),
        "version": "2.0.0",  # Added version tracking
        "features": [
            "enhanced_topics",
            "sentiment_analysis",
            "concept_extraction",
            "relationship_mapping"
        ]
    }

@router.get("/download-results")
async def download_results(
    format: str = Query("json", description="Output format (json/csv)")
) -> Dict[str, Any]:
    """Prepare analysis results for download"""
    if not PIPELINE_RESULTS:
        raise HTTPException(
            status_code=404,
            detail="No results available for download"
        )
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_results_{timestamp}.{format}"
        
        # Save results to file
        save_path = os.path.join("downloads", filename)
        os.makedirs("downloads", exist_ok=True)
        
        if format == "json":
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(PIPELINE_RESULTS, f, indent=2, ensure_ascii=False)
        else:
            raise HTTPException(status_code=400, detail="Only JSON format is currently supported")
        
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