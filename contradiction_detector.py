#!/usr/bin/env python3
"""
contradiction_detector.py
Lightweight contradiction detector + confidence scoring demo.
Outputs validation_summary.csv and stores results in SQLite (validation.db).

Usage:
    python contradiction_detector.py --a salesforce.csv --b hubspot.csv --db validation.db --run_once
"""

import argparse
import pandas as pd
import numpy as np
import sqlite3
import logging
import time
from datetime import datetime
from difflib import SequenceMatcher
import json
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def similar(a, b):
    if pd.isna(a) and pd.isna(b):
        return 1.0
    a = str(a)
    b = str(b)
    return SequenceMatcher(None, a, b).ratio()

def numeric_relative_diff(a, b):
    try:
        a_f = float(a)
        b_f = float(b)
    except Exception:
        return np.inf
    denom = max(abs(a_f), abs(b_f), 1e-6)
    return abs(a_f - b_f) / denom

def score_confidence(row, src_scores):
    """
    Compose confidence from source reliability, recency and agreement.
    row contains:
      - src_a, src_b, ts_a, ts_b, field, value_a, value_b, status
    src_scores: dict source_id -> reliability [0..1]
    """
    S_a = src_scores.get(row['src_a'], 0.7)
    S_b = src_scores.get(row['src_b'], 0.7)
    # choose the "winning" source for recency
    ts_a = pd.to_datetime(row.get('ts_a', None))
    ts_b = pd.to_datetime(row.get('ts_b', None))
    now = pd.Timestamp.now()
    def recency_score(ts):
        if pd.isna(ts):
            return 0.2
        age_hours = (now - ts).total_seconds() / 3600.0
        # map age to score: <24h -> 1.0, 1 week -> 0.5, >30d -> 0.1
        if age_hours < 24:
            return 1.0
        if age_hours < 24*7:
            return 0.7
        if age_hours < 24*30:
            return 0.4
        return 0.1

    R_a = recency_score(ts_a)
    R_b = recency_score(ts_b)
    # agreement: if status == match -> high
    A = 1.0 if row['status'] == 'match' else 0.0
    # conflict frequency: read from row if provided else 0
    C = float(row.get('conflict_freq', 0.0))
    # weight simple average preferring source with better reliability
    alpha, beta, gamma, delta = 0.4, 0.25, 0.25, 0.2
    S = max(S_a, S_b)
    R = max(R_a, R_b)
    raw = (alpha * S) + (beta * R) + (gamma * A) - (delta * C)
    conf = max(0.0, min(1.0, raw))
    return conf

def detect_contradictions(df_a, df_b, key='customer_id', numeric_threshold=0.05, string_threshold=0.85, src_a_name='A', src_b_name='B', src_scores=None):
    # join
    merged = pd.merge(df_a.add_prefix('a_'), df_b.add_prefix('b_'), left_on=f'a_{key}', right_on=f'b_{key}', how='outer')
    # normalize the canonical key column name
    merged['customer_id'] = merged.get(f'a_{key}').combine_first(merged.get(f'b_{key}'))
    fields = []
    # identify union of fields excluding key and source timestamps
    a_cols = [c.replace('a_', '') for c in df_a.add_prefix('a_').columns if c != f'a_{key}']
    b_cols = [c.replace('b_', '') for c in df_b.add_prefix('b_').columns if c != f'b_{key}']

    union_fields = sorted(set(a_cols) | set(b_cols))
    rows = []
    for idx, r in merged.iterrows():
        cust = r['customer_id']
        for fld in union_fields:
            a_col = f'a_{fld}' if f'a_{fld}' in merged.columns else None
            b_col = f'b_{fld}' if f'b_{fld}' in merged.columns else None
            val_a = r[a_col] if a_col else np.nan
            val_b = r[b_col] if b_col else np.nan
            status = 'match'
            conf = 1.0
            # handle both missing
            if pd.isna(val_a) and pd.isna(val_b):
                status = 'missing_both'
                conf = 0.0
            elif pd.isna(val_a) != pd.isna(val_b):
                status = 'missing_one'
                conf = 0.3
            else:
                # both present -> decide type
                # try numeric
                try:
                    a_f = float(val_a)
                    b_f = float(val_b)
                    rel = numeric_relative_diff(a_f, b_f)
                    if rel > numeric_threshold:
                        status = 'contradiction'
                    else:
                        status = 'match'
                except Exception:
                    # string compare
                    sim = similar(val_a, val_b)
                    if sim < string_threshold:
                        status = 'contradiction'
                    else:
                        status = 'match'
            # timestamps if exist
            ts_a = r.get('a_last_updated') if 'a_last_updated' in r else None
            ts_b = r.get('b_last_updated') if 'b_last_updated' in r else None
            row = {
                'customer_id': cust,
                'field': fld,
                'value_a': val_a,
                'value_b': val_b,
                'status': status,
                'src_a': src_a_name,
                'src_b': src_b_name,
                'ts_a': ts_a,
                'ts_b': ts_b,
                'conflict_freq': 0.0  # placeholder for demo
            }
            row['confidence_score'] = score_confidence(row, src_scores or {})
            rows.append(row)
    return pd.DataFrame(rows)

def persist_to_sqlite(df, db_path='validation.db', table_name='validation_summary'):
    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists='append', index=False)
    conn.close()

def visualize_confidence(df):
    # simple bar chart of average confidence by status
    agg = df.groupby('status')['confidence_score'].mean().reset_index()
    agg.plot(kind='bar', x='status', y='confidence_score', legend=False)
    plt.title('Average confidence by status')
    plt.tight_layout()
    plt.show()

def simulate_periodic_run(a_file, b_file, db_path, runs=3, interval_seconds=2, **kwargs):
    for i in range(runs):
        logging.info("Run %d/%d", i+1, runs)
        df_a = pd.read_csv(a_file)
        df_b = pd.read_csv(b_file)
        df_out = detect_contradictions(df_a, df_b, src_scores=kwargs.get('src_scores'))
        tstamp = pd.Timestamp.now().isoformat()
        df_out['run_ts'] = tstamp
        df_out.to_csv('validation_summary.csv', index=False)
        persist_to_sqlite(df_out, db_path=db_path)
        logging.info("Wrote validation_summary.csv and persisted to %s", db_path)
        time.sleep(interval_seconds)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--a', required=True, help='CSV A (salesforce)')
    parser.add_argument('--b', required=True, help='CSV B (hubspot)')
    parser.add_argument('--db', default='validation.db')
    parser.add_argument('--run_once', action='store_true')
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--interval', type=int, default=5)
    args = parser.parse_args()
    # Example source reliability map
    src_scores = {'Salesforce': 0.95, 'HubSpot': 0.9}
    if args.run_once:
        df_a = pd.read_csv(args.a)
        df_b = pd.read_csv(args.b)
        out = detect_contradictions(df_a, df_b, src_a_name='Salesforce', src_b_name='HubSpot', src_scores=src_scores)
        out['run_ts'] = pd.Timestamp.now().isoformat()
        out.to_csv('validation_summary.csv', index=False)
        persist_to_sqlite(out, args.db)
        logging.info("Completed single run. Output -> validation_summary.csv and %s", args.db)
        try:
            visualize_confidence(out)
        except Exception as e:
            logging.info("Visualization skipped: %s", e)
    else:
        simulate_periodic_run(args.a, args.b, args.db, runs=args.runs, interval_seconds=args.interval, src_scores=src_scores)

if __name__ == '__main__':
    main()
