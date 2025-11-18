#!/usr/bin/env python3
"""
schema_drift_detector.py
Compares two dataset versions and reports new/removed/type-mismatched fields.
Usage:
  python schema_drift_detector.py --v1 dataset_v1.csv --v2 dataset_v2.csv --out drift_report.json
"""

import argparse
import pandas as pd
import json
import sqlite3
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def infer_schema(df):
    schema = {}
    for c in df.columns:
        dtype = str(df[c].dtype)
        # simplify types
        if 'int' in dtype:
            t = 'int'
        elif 'float' in dtype:
            t = 'float'
        elif 'datetime' in dtype:
            t = 'datetime'
        else:
            t = 'string'
        nullable = df[c].isnull().any()
        schema[c] = {'type': t, 'nullable': nullable, 'sample_cardinality': int(df[c].nunique(dropna=True))}
    return schema

def compare_schemas(s1, s2):
    fields_s1 = set(s1.keys())
    fields_s2 = set(s2.keys())
    new_fields = sorted(list(fields_s2 - fields_s1))
    removed_fields = sorted(list(fields_s1 - fields_s2))
    type_mismatches = {}
    for f in fields_s1 & fields_s2:
        if s1[f]['type'] != s2[f]['type']:
            type_mismatches[f] = [s1[f]['type'], s2[f]['type']]
    return {
        "new_fields": new_fields,
        "removed_fields": removed_fields,
        "type_mismatches": type_mismatches
    }

def persist_report(report, db_path='schema_drift.db', table='drift_reports'):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_ts TEXT,
            report_json TEXT
        )
    """)
    c.execute(f"INSERT INTO {table} (report_ts, report_json) VALUES (?, ?)", (datetime.utcnow().isoformat(), json.dumps(report)))
    conn.commit()
    conn.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--v1', required=True)
    parser.add_argument('--v2', required=True)
    parser.add_argument('--out', default='drift_report.json')
    parser.add_argument('--db', default='schema_drift.db')
    args = parser.parse_args()
    df1 = pd.read_csv(args.v1)
    df2 = pd.read_csv(args.v2)
    s1 = infer_schema(df1)
    s2 = infer_schema(df2)
    report = compare_schemas(s1, s2)
    # Add extra metadata
    report['_meta'] = {'v1': args.v1, 'v2': args.v2, 'generated_ts': datetime.utcnow().isoformat()}
    with open(args.out, 'w') as f:
        json.dump(report, f, indent=2)
    logging.info("Wrote drift report to %s", args.out)
    persist_report(report, args.db)
    logging.info("Persisted drift report into %s", args.db)

if __name__ == '__main__':
    main()
