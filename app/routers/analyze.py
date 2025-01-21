"""
analyze.py

Router for running the NLP pipeline and handling analysis requests.
Processes extracted text from files and generates analysis results
using only the dependencies specified (pdfminer.six, PyMuPDF, python-docx, etc.).
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import JSONResponse

import fitz  # PyMuPDF for PDF handling
import docx  # python-docx for DOCX handling

from app.models.nlp_pipeline import NLPPipeline
from app.log_store import PROCESSING_LOGS, PIPELINE_RESULTS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["Analyze"])


def extract_text(file_path: str) -> str:
    """
    Extract textual content from a file using dependencies specified in the environment.
    Supports:
      - PDF (via PyMuPDF / fitz)
      - DOCX (via python-docx)
      - Plain text files
    Raises an exception if the file is missing or format is unsupported.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".pdf":
        # Use PyMuPDF to extract PDF text
        text_accumulator = []
        try:
            pdf_document = fitz.open(file_path)
            for page in pdf_document:
                page_text = page.get_text()
                if page_text:
                    text_accumulator.append(page_text)
            pdf_document.close()
        except Exception as e:
            raise RuntimeError(f"Failed to extract PDF text from {file_path}: {str(e)}")

        return "\n".join(text_accumulator)

    elif extension == ".docx":
        # Use python-docx to extract DOCX text
        try:
            doc = docx.Document(file_path)
            return "\n".join(paragraph.text for paragraph in doc.paragraphs)
        except Exception as e:
            raise RuntimeError(f"Failed to extract DOCX text from {file_path}: {str(e)}")

    else:
        # Treat everything else as plain text
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception as e:
            raise RuntimeError(f"Failed to read text file {file_path}: {str(e)}")


@router.post("/run-pipeline")
async def run_pipeline(
    temp_folder: str = Body(..., embed=True),
    options: Optional[Dict[str, Any]] = Body(default=None)
) -> Dict[str, Any]:
    """
    Run the NLP pipeline on extracted text from files inside a given folder.

    Args:
        temp_folder: Path to the folder containing files to analyze.
        options: Optional dict with pipeline configuration:
            - max_topics (int): Maximum number of topics to detect.
            - include_raw (bool): Whether to include raw text in final results.
            - detailed_sentiment (bool): Whether to include more detailed sentiment analysis.

    Returns:
        A dict with status, message, path to the JSON analysis file,
        number of documents processed, and completion time.
    """
    PROCESSING_LOGS["steps"].append(f"Starting pipeline on folder: {temp_folder}")

    if not os.path.isdir(temp_folder):
        error_message = f"Temporary folder not found: {temp_folder}"
        PROCESSING_LOGS["errors"].append(error_message)
        raise HTTPException(status_code=400, detail=error_message)

    try:
        texts_by_doc = {}
        for filename in os.listdir(temp_folder):
            if filename.startswith('.'):  # Skip hidden/system files
                continue
            file_path = os.path.join(temp_folder, filename)
            if os.path.isfile(file_path):
                try:
                    extracted = extract_text(file_path)
                    if extracted:
                        # Convert big text into a list of paragraphs
                        paragraphs = [p.strip() for p in extracted.split("\n") if p.strip()]
                        texts_by_doc[filename] = paragraphs
                except Exception as ex:
                    msg = f"Error extracting text from {filename}: {str(ex)}"
                    PROCESSING_LOGS["errors"].append(msg)

        if not texts_by_doc:
            raise HTTPException(status_code=400, detail="No text could be extracted from documents")

        pipeline_options = {
            "max_topics": options.get("max_topics", 10) if options else 10,
            "include_raw": options.get("include_raw", False) if options else False,
            "detailed_sentiment": options.get("detailed_sentiment", False) if options else False
        }

        # Create and run the NLP pipeline
        pipeline = NLPPipeline(PROCESSING_LOGS)
        results = pipeline.process(texts_by_doc, **pipeline_options)

        # Save analysis output as JSON
        output_path = os.path.join(temp_folder, "analysis_results.json")
        with open(output_path, "w", encoding="utf-8") as out_file:
            json.dump(results, out_file, ensure_ascii=False, indent=2)

        PIPELINE_RESULTS.clear()
        PIPELINE_RESULTS.update(results)

        PROCESSING_LOGS["completion_time"] = datetime.now().isoformat()
        PROCESSING_LOGS["steps"].append("NLP pipeline completed successfully")

        return {
            "status": "success",
            "message": "Analysis completed successfully",
            "result_file": output_path,
            "document_count": len(texts_by_doc),
            "completion_time": PROCESSING_LOGS["completion_time"]
        }

    except HTTPException:
        raise
    except Exception as err:
        error_msg = f"Error in pipeline execution: {str(err)}"
        logger.error(error_msg, exc_info=True)
        PROCESSING_LOGS["errors"].append(error_msg)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": error_msg,
                "error_details": str(err)
            }
        )


@router.get("/analysis-status")
async def get_analysis_status() -> Dict[str, Any]:
    """
    Return the current analysis status and any error messages.
    """
    return {
        "status": "error" if PROCESSING_LOGS["errors"] else "success",
        "steps": PROCESSING_LOGS["steps"],
        "errors": PROCESSING_LOGS["errors"],
        "completion_time": PROCESSING_LOGS.get("completion_time"),
        "results_available": bool(PIPELINE_RESULTS)
    }


@router.get("/analysis-results")
async def get_analysis_results() -> Dict[str, Any]:
    """
    Retrieve the latest analysis results if they are present.
    """
    if not PIPELINE_RESULTS:
        raise HTTPException(
            status_code=404,
            detail="No analysis results available. Please run the pipeline first."
        )
    return {
        "status": "success",
        "data": PIPELINE_RESULTS,
        "timestamp": PROCESSING_LOGS.get("completion_time")
    }


@router.delete("/clear-results")
async def clear_results() -> Dict[str, str]:
    """
    Clear all current analysis results and logs from memory.
    """
    PIPELINE_RESULTS.clear()
    PROCESSING_LOGS["steps"] = []
    PROCESSING_LOGS["errors"] = []
    PROCESSING_LOGS.pop("completion_time", None)

    return {"message": "Analysis results and logs cleared successfully"}
