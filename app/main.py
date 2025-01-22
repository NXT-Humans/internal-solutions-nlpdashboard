from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from starlette.templating import Jinja2Templates
from starlette.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
import logging
import os
import json
import spacy
from typing import Dict, Any, List
from textblob import TextBlob
from datetime import datetime
import numpy as np
from collections import Counter
from pathlib import Path

from app.routers import upload, analyze, visualize
from app.log_store import PROCESSING_LOGS, PIPELINE_RESULTS
from app.config import Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'app_logs_{datetime.now().strftime("%Y%m%d")}.log')
    ]
)
logger = logging.getLogger(__name__)

try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    logger.info("Downloading spaCy model...")
    os.system('python -m spacy download en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")

templates = Jinja2Templates(directory="app/templates")

app.include_router(upload.router)
app.include_router(analyze.router)
app.include_router(visualize.router)

def initialize_processing_logs():
    """Initialize or reset processing logs"""
    PROCESSING_LOGS.clear()
    PROCESSING_LOGS.update({
        "steps": [],
        "errors": [],
        "timestamps": [],
        "stage_details": {},
        "files_processed": {},
        "current_stage": None,
        "start_time": datetime.now().isoformat()
    })
    logger.info("Processing logs initialized")

def generate_document_title(content: str, topics: List[str], summary: str) -> str:
    """Generate meaningful title from document content"""
    if not content:
        return "Document Analysis Dashboard"
        
    document_themes = []
    document_entities = []
    
    # Process document content with spaCy
    document = nlp(content[:10000])  # Process first 10000 chars for efficiency
    
    # Extract key entities and themes
    for entity in document.ents:
        if entity.label_ in ['ORG', 'PRODUCT', 'GPE', 'EVENT', 'TOPIC']:
            document_entities.append(entity.text)
    
    # Extract key noun phrases and themes
    key_phrases = []
    for chunk in document.noun_chunks:
        if len(chunk.text.split()) >= 2:  # Multi-word phrases only
            key_phrases.append(chunk.text)
            
    # Use word frequency for theme detection
    word_frequencies = Counter([
        token.text.lower() for token in document 
        if not token.is_stop and not token.is_punct and len(token.text) > 3
    ])
    document_themes = [word for word, frequency in word_frequencies.most_common(3)]
    
    # Generate title using extracted information
    if document_entities and key_phrases:
        # Use most relevant entity and phrase
        return f"Analysis of {document_entities[0]}: {key_phrases[0].title()}"
    
    elif topics and topics[0]:
        # Use main topic with context
        main_topic = topics[0]
        if document_themes:
            return f"{main_topic.title()}: {document_themes[0].title()} Analysis"
        return f"{main_topic.title()} Analysis Report"
    
    elif summary:
        # Generate from summary content
        summary_document = nlp(summary[:500])
        for sentence in summary_document.sents:
            # Use first meaningful sentence
            clean_sentence = sentence.text.strip()
            if len(clean_sentence.split()) > 3:
                return clean_sentence.title()
    
    # Fallback to any available themes
    if document_themes:
        return f"{document_themes[0].title()}: Document Analysis"
        
    return "Document Analysis Report"

def fix_numpy_types(obj: Any) -> Any:
    """Convert NumPy types to Python native types"""
    if isinstance(obj, dict):
        return {k: fix_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [fix_numpy_types(x) for x in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

@app.on_event("startup")
async def startup_event():
    """Application startup event handler"""
    logger.info("Starting up application")
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
    logger.info("Shutting down application")
    
    PIPELINE_RESULTS.clear()
    PROCESSING_LOGS["end_time"] = datetime.now().isoformat()
    
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

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions"""
    logger.error(f"HTTP error occurred: {exc.detail}")
    return templates.TemplateResponse(
        "error.html",
        {
            "request": request,
            "error_code": exc.status_code,
            "error_message": exc.detail,
        },
        status_code=exc.status_code
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception occurred: {str(exc)}", exc_info=True)
    return templates.TemplateResponse(
        "error.html",
        {
            "request": request,
            "error_code": 500,
            "error_message": "An unexpected error occurred. Please try again later.",
        },
        status_code=500
    )

@app.get("/")
async def home(request: Request):
    """Render home page"""
    logger.info("Rendering home page")
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "processing_logs": PROCESSING_LOGS if PROCESSING_LOGS.get("steps") or PROCESSING_LOGS.get("errors") else None
        }
    )

@app.get("/dashboard")
async def dashboard(request: Request):
    """Render dashboard with analysis results"""
    logger.info("Rendering dashboard page")
    
    if not PIPELINE_RESULTS:
        logger.warning("No analysis results available")
        return templates.TemplateResponse(
            "dashboard.html",
            {
                "request": request,
                "processing_status": PROCESSING_LOGS
            }
        )

    try:
        # Process analysis results
        cleaned_results = fix_numpy_types(PIPELINE_RESULTS)
        
        # Extract all text content for analysis
        document_texts = []
        for document_data in cleaned_results.get("documents", {}).values():
            document_texts.extend(document_data.get("cleaned_paragraphs", []))
        
        combined_text = " ".join(document_texts)
        
        # Generate dynamic title from content
        document_title = generate_document_title(
            combined_text,
            cleaned_results.get("global_topics", []),
            cleaned_results.get("global_summary", "")
        )
        
        # Process document sections
        summary_sections = process_summary_sections(cleaned_results.get("global_summary", ""))
        
        # Process topics with context and sentiment
        topics_data = []
        for topic in cleaned_results.get("global_topics", []):
            sentiment_score = analyze_sentiment(topic)
            topic_contexts = find_topic_contexts(topic, cleaned_results.get("documents", {}))
            topics_data.append({
                "text": topic,
                "frequency": count_topic_frequency(topic, cleaned_results.get("documents", {})),
                "sentiment_score": sentiment_score,
                "sentiment": categorize_sentiment(sentiment_score),
                "contexts": topic_contexts
            })
        
        # Process word cloud data
        wordcloud_data = {}
        for word, frequency in cleaned_results.get("global_wordcloud_data", {}).items():
            sentiment_score = analyze_sentiment(word)
            word_contexts = find_topic_contexts(word, cleaned_results.get("documents", {}))
            wordcloud_data[word] = {
                "frequency": frequency,
                "sentiment": sentiment_score,
                "contexts": word_contexts
            }
        
        # Process individual documents
        documents_data = {}
        for document_name, document_data in cleaned_results.get("documents", {}).items():
            documents_data[document_name] = {
                "cleaned_paragraphs": document_data.get("cleaned_paragraphs", []),
                "paragraph_sentiments": document_data.get("paragraph_sentiments", []),
                "summary": document_data.get("summary", "")
            }
        
        template_data = {
            "request": request,
            "title": document_title,
            "key_threats": summary_sections["key_threats"],
            "current_landscape": summary_sections["current_landscape"],
            "defense_strategies": summary_sections["defense_strategies"],
            "topics_data": topics_data,
            "wordcloud_data": wordcloud_data,
            "documents": documents_data,
            "processing_logs": PROCESSING_LOGS
        }
        
        return templates.TemplateResponse("dashboard.html", template_data)
        
    except Exception as error:
        logger.error(f"Error processing dashboard data: {str(error)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing analysis results: {str(error)}")

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
    
    sentences = [sentence.strip() + "." for sentence in summary_text.split(".") if sentence.strip()]
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        if any(keyword in sentence_lower for keyword in classification_keywords["key_threats"]):
            sections["key_threats"].append(sentence)
        elif any(keyword in sentence_lower for keyword in classification_keywords["defense_strategies"]):
            sections["defense_strategies"].append(sentence)
        else:
            sections["current_landscape"].append(sentence)
    
    return {
        "key_threats": " ".join(sections["key_threats"]) or "No significant threats identified in analysis.",
        "current_landscape": " ".join(sections["current_landscape"]) or "No landscape analysis available.",
        "defense_strategies": " ".join(sections["defense_strategies"]) or "No strategies identified in content."
    }

def analyze_sentiment(text: str) -> float:
    """Analyze text sentiment"""
    try:
        return TextBlob(text).sentiment.polarity
    except Exception as error:
        logger.error(f"Error in sentiment analysis: {error}")
        return 0.0

def categorize_sentiment(score: float) -> str:
    """Categorize sentiment score"""
    if score < -0.1:
        return "negative"
    elif score > 0.1:
        return "positive"
    return "neutral"

def count_topic_frequency(topic: str, documents: Dict) -> int:
    """Count topic occurrences"""
    count = 0
    topic_lower = topic.lower()
    
    for document_data in documents.values():
        for paragraph in document_data.get("cleaned_paragraphs", []):
            if topic_lower in paragraph.lower():
                count += 1
    
    return max(count, 1)

def find_topic_contexts(topic: str, documents: Dict) -> List[Dict[str, Any]]:
    """Find context paragraphs for topic"""
    contexts = []
    topic_lower = topic.lower()
    
    for document_name, document_data in documents.items():
        paragraphs = document_data.get("cleaned_paragraphs", [])
        
        for paragraph in paragraphs:
            if topic_lower in paragraph.lower():
                contexts.append({
                    "document": document_name,
                    "text": paragraph,
                    "sentiment": analyze_sentiment(paragraph)
                })
                
                if len(contexts) >= 5:
                    break
    
    return sorted(contexts, key=lambda x: abs(x["sentiment"]), reverse=True)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "environment": Config.ENVIRONMENT,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=Config.ENVIRONMENT == "development",
        workers=1,
        log_level="info"
    )