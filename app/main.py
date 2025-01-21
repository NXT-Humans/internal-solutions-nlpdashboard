from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from starlette.templating import Jinja2Templates
from starlette.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
import logging
import os
import json
from typing import Dict, Any, List
from textblob import TextBlob
from datetime import datetime

import numpy as np  # For converting np.int64 / np.float64 to Python types if needed

# Import routers and configurations
from app.routers import upload, analyze, visualize
from app.log_store import PROCESSING_LOGS, PIPELINE_RESULTS
from app.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'app_logs_{datetime.now().strftime("%Y%m%d")}.log')
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI(
    title="NLP Dashboard",
    description="Advanced text analysis and visualization platform with context-aware insights",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="app/templates")

# Include routers
app.include_router(upload.router)
app.include_router(analyze.router)
app.include_router(visualize.router)

################################
#        Utility Methods
################################

def initialize_processing_logs():
    """Initialize or reset processing logs"""
    PROCESSING_LOGS.clear()
    PROCESSING_LOGS["steps"] = []
    PROCESSING_LOGS["errors"] = []
    PROCESSING_LOGS["start_time"] = datetime.now().isoformat()
    logger.info("Processing logs initialized")

def fix_np_types(obj: Any) -> Any:
    """
    Recursively convert NumPy dtypes (int64, float64, ndarrays, etc.)
    to native Python types. Prevents JSON serialization errors when
    using Jinja's tojson filter.
    """
    if isinstance(obj, dict):
        return {k: fix_np_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [fix_np_types(x) for x in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def chunk_text_if_needed(text: str, max_tokens: int = 512) -> List[str]:
    """
    Break a large string into multiple pieces, each up to `max_tokens` words.
    Prevents sending overly long sequences to huggingface/transformers models.

    Args:
        text: The full input string.
        max_tokens: Maximum number of tokens (words) per chunk.

    Returns:
        A list of chunked strings, each below the token limit.
    """
    # Simple word-based approach. For more precise behavior, you could do
    # actual BPE tokenization with the huggingface tokenizer.
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_tokens):
        chunk_slice = words[i:i + max_tokens]
        chunk_str = " ".join(chunk_slice)
        chunks.append(chunk_str)
    return chunks

################################
#   App Lifecycle Events
################################

@app.on_event("startup")
async def startup_event():
    """Application startup event handler"""
    logger.info("Starting up NLP Dashboard application")
    initialize_processing_logs()
    
    required_directories = {
        "app/static": "Static files directory",
        "app/static/js": "JavaScript files directory",
        "app/static/css": "CSS files directory",
        "app/templates": "Template files directory",
        "temp": "Temporary files directory",
        "logs": "Application logs directory"
    }
    
    for directory, description in required_directories.items():
        try:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Verified {description}: {directory}")
        except Exception as error:
            logger.error(f"Failed to create {description} {directory}: {str(error)}")
            raise

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event handler"""
    logger.info("Shutting down NLP Dashboard application")
    
    # Clear global stores
    PIPELINE_RESULTS.clear()
    PROCESSING_LOGS["end_time"] = datetime.now().isoformat()
    
    # Clean up temporary files
    temp_directory = "temp"
    if os.path.exists(temp_directory):
        for item in os.listdir(temp_directory):
            item_path = os.path.join(temp_directory, item)
            try:
                if os.path.isfile(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    os.rmdir(item_path)
            except Exception as error:
                logger.error(f"Error cleaning up {item_path}: {error}")


################################
#  Error / Exception Handlers
################################

from starlette.exceptions import HTTPException as StarletteHTTPException

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions and render error template"""
    logger.error(f"HTTP error occurred: {exc.detail}")
    return templates.TemplateResponse(
        "error.html",
        {
            "request": request,
            "error_code": exc.status_code,
            "error_message": exc.detail,
            "title": f"Error {exc.status_code}"
        },
        status_code=exc.status_code
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions and render error template"""
    logger.error(f"Unhandled exception occurred: {str(exc)}", exc_info=True)
    return templates.TemplateResponse(
        "error.html",
        {
            "request": request,
            "error_code": 500,
            "error_message": "An unexpected error occurred. Please try again later.",
            "title": "Internal Server Error"
        },
        status_code=500
    )


################################
#           Routes
################################

@app.get("/")
async def home(request: Request):
    """Render home page"""
    logger.info("Rendering home page")
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "title": "NLP Dashboard - Home",
            "processing_logs": PROCESSING_LOGS if PROCESSING_LOGS.get("steps") or PROCESSING_LOGS.get("errors") else None
        }
    )

@app.get("/dashboard")
async def dashboard(request: Request):
    """
    Render dashboard with analysis results and document context.
    Uses chunking to avoid passing extremely large strings
    into any huggingface or large language model calls.
    """
    logger.info("Rendering dashboard page")
    
    # Initialize default data structures
    summary_sections = {
        "key_threats": "No threat analysis available.",
        "current_landscape": "No landscape analysis available.",
        "defense_strategies": "No defense strategies available."
    }

    if not PIPELINE_RESULTS:
        logger.warning("No pipeline results available")
        template_data = {
            "request": request,
            "title": "AI Financial Crime Analysis Dashboard",
            "key_threats": summary_sections["key_threats"],
            "current_landscape": summary_sections["current_landscape"],
            "defense_strategies": summary_sections["defense_strategies"],
            "topics_data": [{"text": "No topics available", "frequency": 1, "sentiment_score": 0, "sentiment": "neutral"}],
            "wordcloud_data": {"no data": {"frequency": 1, "sentiment": 0}},
            "documents": {},
            "processing_logs": PROCESSING_LOGS
        }
    else:
        logger.info(f"Processing pipeline results with keys: {PIPELINE_RESULTS.keys()}")
        
        # Convert all NumPy types in PIPELINE_RESULTS to Python standard types
        cleaned_results = fix_np_types(PIPELINE_RESULTS)

        # If there's a large global_summary, chunk it
        if "global_summary" in cleaned_results:
            # Example: if the summary is too large, break it into multiple parts
            # so that any huggingface-based summarization or embedding won't exceed max tokens
            chunked_summaries = []
            summary_chunks = chunk_text_if_needed(cleaned_results["global_summary"], max_tokens=512)
            for chunk in summary_chunks:
                # You could pass each chunk to your summarizer or pipeline here
                # Instead of placeholders, we simply store them for demonstration:
                chunked_summaries.append(chunk)

            # Rejoin or pass to process_summary_sections
            combined_summary = ". ".join(chunked_summaries)
            summary_sections = process_summary_sections(combined_summary)

        # Process topics with sentiment and context
        topics_data = []
        if "global_topics" in cleaned_results:
            for topic in cleaned_results["global_topics"]:
                sentiment_score = analyze_sentiment(topic)
                topic_frequency = count_topic_frequency(topic, cleaned_results.get("documents", {}))
                topic_contexts = find_topic_contexts(topic, cleaned_results.get("documents", {}))
                
                topics_data.append({
                    "text": topic,
                    "frequency": topic_frequency,
                    "sentiment_score": sentiment_score,
                    "sentiment": categorize_sentiment(sentiment_score),
                    "contexts": topic_contexts
                })
        
        # Process wordcloud with sentiment and context
        wordcloud_data = {}
        if "global_wordcloud_data" in cleaned_results:
            for word, frequency in cleaned_results["global_wordcloud_data"].items():
                sentiment_score = analyze_sentiment(word)
                word_contexts = find_topic_contexts(word, cleaned_results.get("documents", {}))
                
                wordcloud_data[word] = {
                    "frequency": frequency,
                    "sentiment": sentiment_score,
                    "contexts": word_contexts
                }
        
        # Process documents for context lookup
        documents_data = {}
        if "documents" in cleaned_results:
            for doc_name, doc_data in cleaned_results["documents"].items():
                documents_data[doc_name] = {
                    "cleaned_paragraphs": doc_data.get("cleaned_paragraphs", []),
                    "raw_paragraphs": doc_data.get("raw_paragraphs", []),
                    "paragraph_sentiments": doc_data.get("paragraph_sentiments", []),
                    "summary": doc_data.get("summary", "No summary available")
                }
        
        template_data = {
            "request": request,
            "title": "AI Financial Crime Analysis Dashboard",
            "key_threats": summary_sections["key_threats"],
            "current_landscape": summary_sections["current_landscape"],
            "defense_strategies": summary_sections["defense_strategies"],
            "topics_data": topics_data,
            "wordcloud_data": wordcloud_data,
            "documents": documents_data,
            "processing_logs": PROCESSING_LOGS
        }
        
        logger.info(
            f"Dashboard data prepared - Topics: {len(topics_data)}, "
            f"Wordcloud terms: {len(wordcloud_data)}, "
            f"Documents: {len(documents_data)}"
        )
    
    return templates.TemplateResponse("dashboard.html", template_data)

def process_summary_sections(summary_text: str) -> Dict[str, str]:
    """Process summary text into themed sections"""
    classification_keywords = {
        "key_threats": [
            "threat", "risk", "danger", "attack", "fraud", "vulnerability",
            "malicious", "exploit", "breach", "compromise"
        ],
        "defense_strategies": [
            "defend", "protect", "strategy", "solution", "prevent", "mitigate",
            "secure", "safeguard", "monitor", "detect"
        ]
    }
    
    sections = {
        "key_threats": [],
        "current_landscape": [],
        "defense_strategies": []
    }
    
    sentences = [s.strip() + "." for s in summary_text.split(".") if s.strip()]
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        if any(keyword in sentence_lower for keyword in classification_keywords["key_threats"]):
            sections["key_threats"].append(sentence)
        elif any(keyword in sentence_lower for keyword in classification_keywords["defense_strategies"]):
            sections["defense_strategies"].append(sentence)
        else:
            sections["current_landscape"].append(sentence)
    
    return {
        "key_threats": " ".join(sections["key_threats"]) or "No immediate threats identified.",
        "current_landscape": " ".join(sections["current_landscape"]) or "No landscape analysis available.",
        "defense_strategies": " ".join(sections["defense_strategies"]) or "No defense strategies identified."
    }

def analyze_sentiment(text: str) -> float:
    """Analyze text sentiment using TextBlob"""
    try:
        return TextBlob(text).sentiment.polarity
    except Exception as error:
        logger.error(f"Error in sentiment analysis: {error}")
        return 0.0

def categorize_sentiment(score: float) -> str:
    """Categorize sentiment score into positive, negative, or neutral"""
    if score < -0.1:
        return "negative"
    elif score > 0.1:
        return "positive"
    return "neutral"

def count_topic_frequency(topic: str, documents: Dict) -> int:
    """Count occurrences of topic in documents"""
    count = 0
    topic_lower = topic.lower()
    
    for doc_data in documents.values():
        # Check cleaned paragraphs
        for paragraph in doc_data.get("cleaned_paragraphs", []):
            if topic_lower in paragraph.lower():
                count += 1
        
        # Check raw paragraphs if available
        for paragraph in doc_data.get("raw_paragraphs", []):
            if topic_lower in paragraph.lower():
                count += 1
    
    return max(count, 1)  # Ensure minimum frequency of 1

def find_topic_contexts(topic: str, documents: Dict) -> List[Dict[str, Any]]:
    """Find context paragraphs containing the topic"""
    contexts = []
    topic_lower = topic.lower()
    
    for doc_name, doc_data in documents.items():
        # Check both cleaned and raw paragraphs
        paragraphs = doc_data.get("cleaned_paragraphs", []) + doc_data.get("raw_paragraphs", [])
        
        for paragraph in paragraphs:
            if topic_lower in paragraph.lower():
                contexts.append({
                    "document": doc_name,
                    "text": paragraph,
                    "sentiment": analyze_sentiment(paragraph)
                })
                
                # Limit number of contexts per topic
                if len(contexts) >= 5:
                    break
    
    # Sort contexts by sentiment relevance (descending absolute value)
    return sorted(contexts, key=lambda x: abs(x["sentiment"]), reverse=True)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "environment": Config.ENVIRONMENT,
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True if Config.ENVIRONMENT == "development" else False,
        workers=1,
        log_level="info"
    )
