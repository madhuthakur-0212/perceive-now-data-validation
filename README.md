Perceive Now – Data Validation & Schema Drift Demo

This repository contains my technical assessment for the Data Engineer – Validation & Quality role at Perceive Now.
It includes code for contradiction detection, schema drift detection, confidence scoring, and audit-ready outputs.

📂 Contents

contradiction_detector.py – Detects inconsistencies between two datasets, computes confidence scores, stores output in SQLite.

schema_drift_detector.py – Compares two dataset versions, detects schema changes, and logs drift.

salesforce.csv – Sample dataset A.

hubspot.csv – Sample dataset B.

validation_summary.csv – Output from the contradiction detector.

validation.db – SQLite audit log for contradictions.

drift_report.json – Output from schema drift detection.

schema_drift.db – SQLite audit log for schema changes.

PerceiveNow_Data_Validation_Assessment_Arun_Kumar.pdf – Final document explaining logic, architecture, and reasoning.

 How to Run the Scripts
1. Install Dependencies
pip install pandas numpy matplotlib

2. Run Contradiction Detector
python contradiction_detector.py --a salesforce.csv --b hubspot.csv --run_once


Outputs:

validation_summary.csv

validation.db

Confidence-score visualization

3. Run Schema Drift Detector
python schema_drift_detector.py --v1 salesforce.csv --v2 hubspot.csv --out drift_report.json


Outputs:

drift_report.json

schema_drift.db

 What This Solution Demonstrates

Automated contradiction detection

Field-level and record-level validation

Numeric delta, fuzzy matching, timestamp comparison

Confidence scoring with reliability, recency, and agreement

Schema drift detection for added/removed/type-changed fields

Audit history stored in SQLite

Clean, reproducible Python code

📄 Full Problem Explanation

See the included PDF:

PerceiveNow_Data_Validation_Explnation.pdf

This document includes:

Story 1 reasoning

Story 2 reasoning

Architecture

Detection logic

Confidence scoring math

Lineage & governance considerations

Screenshots + explanations