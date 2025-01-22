"""
upload.py

Router for handling file uploads, extraction, and initial processing.
Supports ZIP files containing PDFs, DOCX, TXT, CSV, and JSON documents.
"""

import os
import shutil
import zipfile
import uuid
import logging
from pathlib import Path
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import RedirectResponse
from typing import Set, Dict, Any
from datetime import datetime

from app.log_store import PROCESSING_LOGS, PIPELINE_RESULTS
from app.utils.file_parser import parse_zip_file
from app.models.nlp_pipeline import NLPPipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["Upload"])

# Define valid file extensions
VALID_FILE_EXTENSIONS: Set[str] = {
    '.pdf', '.docx', '.txt', '.csv', '.json'
}

# Define system files to ignore
IGNORED_FILE_PATTERNS: Set[str] = {
    '__MACOSX',
    '._',
    '.DS_Store',
    'Thumbs.db',
    'desktop.ini'
}

@router.post("/zip")
async def upload_zip(file: UploadFile = File(...)):
    """Handle ZIP file upload containing documents for analysis"""
    # Initialize processing logs
    PROCESSING_LOGS["steps"] = []
    PROCESSING_LOGS["errors"] = []
    PROCESSING_LOGS["start_time"] = datetime.now().isoformat()
    PROCESSING_LOGS["steps"].append("Started file upload processing")

    # Validate file type
    if not file.filename.lower().endswith('.zip'):
        error_message = "File must be a ZIP archive"
        PROCESSING_LOGS["errors"].append(error_message)
        raise HTTPException(status_code=400, detail=error_message)

    # Create unique temporary directory
    temp_folder = Path(f"temp_{uuid.uuid4()}")
    try:
        temp_folder.mkdir(parents=True, exist_ok=True)
        zip_path = temp_folder / file.filename
        
        # Save uploaded file
        PROCESSING_LOGS["steps"].append("Saving uploaded file")
        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract valid files
        PROCESSING_LOGS["steps"].append("Extracting ZIP contents")
        extracted_files = []
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            for zip_info in zip_ref.filelist:
                # Skip ignored files and empty directories
                if any(pattern in zip_info.filename for pattern in IGNORED_FILE_PATTERNS):
                    continue
                if zip_info.filename.endswith('/'):
                    continue
                    
                # Check file extension
                file_extension = Path(zip_info.filename).suffix.lower()
                if file_extension in VALID_FILE_EXTENSIONS:
                    zip_ref.extract(zip_info, temp_folder)
                    extracted_files.append(zip_info.filename)
                    PROCESSING_LOGS["steps"].append(f"Extracted: {zip_info.filename}")

        if not extracted_files:
            error_message = "No valid documents found in ZIP file"
            PROCESSING_LOGS["errors"].append(error_message)
            return RedirectResponse(url="/", status_code=303)

        # Parse extracted documents
        PROCESSING_LOGS["steps"].append("Parsing extracted files")
        try:
            texts_by_document = parse_zip_file(temp_folder, PROCESSING_LOGS)
            PROCESSING_LOGS["steps"].append("Successfully parsed all files")
        except Exception as error:
            error_message = f"Error parsing files: {str(error)}"
            PROCESSING_LOGS["errors"].append(error_message)
            return RedirectResponse(url="/", status_code=303)

        # Run NLP pipeline
        PROCESSING_LOGS["steps"].append("Starting NLP analysis pipeline")
        pipeline = NLPPipeline(PROCESSING_LOGS)
        try:
            results = pipeline.process(texts_by_document)
            PIPELINE_RESULTS.clear()
            PIPELINE_RESULTS.update(results)
            PROCESSING_LOGS["steps"].append("NLP pipeline completed successfully")
            PROCESSING_LOGS["completion_time"] = datetime.now().isoformat()
        except Exception as error:
            error_message = f"NLP pipeline error: {str(error)}"
            PROCESSING_LOGS["errors"].append(error_message)
            return RedirectResponse(url="/", status_code=303)

        # Clean up temporary files
        try:
            if zip_path.exists():
                os.remove(zip_path)
            PROCESSING_LOGS["steps"].append("Cleaned up temporary files")
        except Exception as error:
            logger.error(f"Error cleaning up temporary files: {str(error)}")

        # Redirect to dashboard on success
        return RedirectResponse(url="/dashboard", status_code=303)

    except Exception as error:
        error_message = f"Error processing upload: {str(error)}"
        logger.error(error_message, exc_info=True)
        PROCESSING_LOGS["errors"].append(error_message)
        # On error, redirect back to upload page where status will be shown
        return RedirectResponse(url="/", status_code=303)

@router.get("/upload-status")
async def get_upload_status() -> Dict[str, Any]:
    """Return current upload and processing status"""
    return {
        "status": "error" if PROCESSING_LOGS["errors"] else "success",
        "steps": PROCESSING_LOGS["steps"],
        "errors": PROCESSING_LOGS["errors"],
        "timestamp": PROCESSING_LOGS.get("start_time"),
        "completion_time": PROCESSING_LOGS.get("completion_time")
    }