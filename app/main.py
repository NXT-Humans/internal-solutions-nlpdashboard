"""
main.py - FastAPI Application Entry Point with Dashboard Routes

Handles:
- Application initialization and configuration
- Router inclusion for upload, analyze, and visualization endpoints
- Static file mounting and template rendering for the dashboard
- API endpoint for analysis results
- Error handling and logging

Version: 1.0.0
"""

import logging
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
import spacy
import numpy as np

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

app = FastAPI(
    title="NLP Dashboard",
    description="Document analysis and insights dashboard",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

required_dirs = [
    Path("app/static"),
    Path("app/static/js"),
    Path("app/static/css"),
    Path("app/templates"),
    Path("temp"),
    Path("logs")
]
for directory in required_dirs:
    directory.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

app.include_router(upload.router)
app.include_router(analyze.router)
app.include_router(visualize.router)

def fix_numpy_types(obj: Any) -> Any:
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
    elif isinstance(obj, bytes):
        return obj.decode('utf-8', errors='ignore')
    elif hasattr(obj, 'to_dict'):
        return fix_numpy_types(obj.to_dict())
    return obj

@app.get("/")
async def home(request: Request):
    logger.info("Rendering home page")
    return templates.TemplateResponse("index.html", {"request": request, "processing_logs": PROCESSING_LOGS})

@app.get("/dashboard")
async def dashboard(request: Request):
    logger.info("Rendering dashboard page")
    if not PIPELINE_RESULTS:
        logger.warning("No analysis results available")
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "title": "Analysis Dashboard",
            "subtitle": "No analysis results available",
            "summary_data": {},
            "topics_data": [],
            "wordcloud_data": {},
            "documents": {},
            "processing_logs": PROCESSING_LOGS,
            "metadata": {
                "document_count": 0,
                "topic_count": 0,
                "word_count": 0,
                "processing_duration": 0,
                "start_time": datetime.now().isoformat(),
                "completion_time": datetime.now().isoformat()
            }
        })
    try:
        cleaned_results = fix_numpy_types(PIPELINE_RESULTS)
        return templates.TemplateResponse("dashboard.html", {"request": request, **cleaned_results})
    except Exception as error:
        logger.error(f"Error processing dashboard data: {str(error)}", exc_info=True)
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error_code": 500,
            "error_message": f"Error processing analysis results: {str(error)}"
        })

@app.get("/api/analysis-results")
async def get_analysis_results():
    if not PIPELINE_RESULTS:
        raise HTTPException(status_code=404, detail="No analysis results available")
    return PIPELINE_RESULTS

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat(), "version": "1.0.0"}

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    logger.error(f"HTTP error occurred: {exc.detail}")
    return templates.TemplateResponse("error.html", {
        "request": request,
        "error_code": exc.status_code,
        "error_message": exc.detail
    }, status_code=exc.status_code)

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception occurred: {str(exc)}", exc_info=True)
    return templates.TemplateResponse("error.html", {
        "request": request,
        "error_code": 500,
        "error_message": "An unexpected error occurred. Please try again later."
    }, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=Config.ENVIRONMENT == "development", workers=1, log_level="info")
