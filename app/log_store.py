"""
log_store.py

Holds shared global dictionaries to prevent circular imports.
"""

PROCESSING_LOGS = {
    "steps": [],
    "errors": []
}

# We'll store pipeline results here so the dashboard can read them.
PIPELINE_RESULTS = {}
