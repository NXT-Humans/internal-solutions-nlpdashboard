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
from fastapi.responses import JSONResponse, RedirectResponse
from typing import Set, Dict, Any

from app.log_store import PROCESSING_LOGS, PIPELINE_RESULTS
from app.utils.file_parser import parse_zip_file
from app.models.nlp_pipeline import NLPPipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["Upload"])

# Valid file extensions
VALID_EXTENSIONS: Set[str] = {
    '.pdf', '.docx', '.txt', '.csv', '.json'
}

# Files and patterns to ignore
IGNORE_PATTERNS: Set[str] = {
    '__MACOSX',
    '._',
    '.DS_Store',
    'Thumbs.db',
    'desktop.ini'
}

@router.post("/zip")
async def upload_zip(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Handle ZIP file upload containing documents for analysis.
    
    Args:
        file: Uploaded ZIP file
        
    Returns:
        Redirect to dashboard or error response
    """
    PROCESSING_LOGS["steps"].append("Received ZIP file upload request")

    # Validate file type
    if not file.filename.lower().endswith('.zip'):
        error_msg = "File must be a ZIP archive"
        PROCESSING_LOGS["errors"].append(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)

    # Create unique temporary directory
    temp_folder = Path(f"temp_{uuid.uuid4()}")
    try:
        temp_folder.mkdir(parents=True, exist_ok=True)
        zip_path = temp_folder / file.filename

        # Save uploaded file
        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        PROCESSING_LOGS["steps"].append(f"ZIP file saved: {zip_path}")

        # Extract valid files
        extracted_files = []
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            for zip_info in zip_ref.filelist:
                # Skip ignored files and empty directories
                if any(pattern in zip_info.filename for pattern in IGNORE_PATTERNS):
                    continue
                if zip_info.filename.endswith('/'):
                    continue
                    
                # Check file extension
                ext = Path(zip_info.filename).suffix.lower()
                if ext in VALID_EXTENSIONS:
                    zip_ref.extract(zip_info, temp_folder)
                    extracted_files.append(zip_info.filename)
                    PROCESSING_LOGS["steps"].append(f"Extracted: {zip_info.filename}")

        if not extracted_files:
            raise HTTPException(
                status_code=400,
                detail="No valid documents found in ZIP file"
            )

        PROCESSING_LOGS["steps"].append("ZIP extracted successfully")

        # Parse extracted documents
        try:
            texts_by_doc = parse_zip_file(temp_folder, PROCESSING_LOGS)
            PROCESSING_LOGS["steps"].append("All files parsed successfully")
        except Exception as error:
            PROCESSING_LOGS["errors"].append(f"Error parsing files: {str(error)}")
            raise HTTPException(status_code=500, detail=str(error))

        # Run NLP pipeline
        pipeline = NLPPipeline(PROCESSING_LOGS)
        try:
            results = pipeline.process(texts_by_doc)
            PIPELINE_RESULTS.clear()
            PIPELINE_RESULTS.update(results)
            PROCESSING_LOGS["steps"].append("NLP pipeline results stored")
        except Exception as error:
            PROCESSING_LOGS["errors"].append(f"NLP pipeline error: {str(error)}")
            raise HTTPException(status_code=500, detail=str(error))

        return RedirectResponse(url="/dashboard", status_code=303)

    except HTTPException:
        raise
    except Exception as error:
        error_msg = f"Error processing upload: {str(error)}"
        PROCESSING_LOGS["errors"].append(error_msg)
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)
    finally:
        # Clean up temporary files
        try:
            if zip_path.exists():
                os.remove(zip_path)
            if temp_folder.exists():
                shutil.rmtree(temp_folder)
        except Exception as error:
            logger.error(f"Error cleaning up temporary files: {str(error)}")

@router.get("/upload-status")
async def get_upload_status() -> Dict[str, Any]:
    """
    Get current upload and processing status.
    """
    return {
        "status": "error" if PROCESSING_LOGS["errors"] else "success",
        "steps": PROCESSING_LOGS["steps"],
        "errors": PROCESSING_LOGS["errors"],
        "timestamp": PROCESSING_LOGS.get("timestamp")
    }