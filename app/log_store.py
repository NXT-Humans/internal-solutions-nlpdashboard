"""
log_store.py - Enhanced logging with detailed processing status
"""

from datetime import datetime
from typing import Dict, List, Any

class ProcessingStage:
    def __init__(self, name: str):
        self.name = name
        self.start_time = datetime.now()
        self.end_time = None
        self.status = "processing"
        self.message = ""
        self.substeps = []

    def complete(self, message: str = ""):
        self.end_time = datetime.now()
        self.status = "complete"
        self.message = message

    def fail(self, error_message: str):
        self.end_time = datetime.now()
        self.status = "error"
        self.message = error_message

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status,
            "message": self.message,
            "substeps": self.substeps
        }

PROCESSING_LOGS = {
    "steps": [],
    "errors": [],
    "timestamps": [],
    "stage_details": {},
    "files_processed": {},
    "current_stage": None,
    "start_time": None,
    "completion_time": None
}

def log_step(step: str, stage: str = None):
    """Log a processing step with timestamp."""
    timestamp = datetime.now()
    PROCESSING_LOGS["steps"].append(step)
    PROCESSING_LOGS["timestamps"].append(timestamp.isoformat())
    
    if stage:
        if stage not in PROCESSING_LOGS["stage_details"]:
            PROCESSING_LOGS["stage_details"][stage] = ProcessingStage(stage)
        PROCESSING_LOGS["stage_details"][stage].substeps.append({
            "message": step,
            "timestamp": timestamp.isoformat()
        })

def log_file_status(filename: str, status: str):
    """Log the processing status of a specific file."""
    PROCESSING_LOGS["files_processed"][filename] = {
        "status": status,
        "timestamp": datetime.now().isoformat()
    }

def log_error(error: str, stage: str = None):
    """Log an error with optional stage information."""
    timestamp = datetime.now()
    PROCESSING_LOGS["errors"].append(error)
    
    if stage and stage in PROCESSING_LOGS["stage_details"]:
        PROCESSING_LOGS["stage_details"][stage].fail(error)

def start_processing():
    """Initialize or reset processing logs."""
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

def complete_processing():
    """Mark processing as complete."""
    PROCESSING_LOGS["completion_time"] = datetime.now().isoformat()
    
    # Mark any incomplete stages as complete
    for stage in PROCESSING_LOGS["stage_details"].values():
        if stage.status == "processing":
            stage.complete()

# Pipeline will use these functions to log detailed status
PIPELINE_RESULTS = {}