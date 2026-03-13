#!/usr/bin/env python3
"""
results_db.py — Results database schema and utilities

Usage:
  python scoring/results_db.py --init          # Initialise database
  python scoring/results_db.py --serve --port 8001  # Run query API
"""

import sqlite3
import argparse
import json
import uuid
from datetime import datetime
from pathlib import Path

DB_PATH = Path("/data/benchmark-results/results.db")


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id              TEXT PRIMARY KEY,
    model_id            TEXT NOT NULL,
    model_hf_repo       TEXT NOT NULL,
    model_revision      TEXT NOT NULL,
    model_family        TEXT NOT NULL,
    model_tier          TEXT NOT NULL,
    quantization        TEXT NOT NULL,
    vllm_version        TEXT,
    temperature         REAL NOT NULL,
    max_tokens          INTEGER NOT NULL,
    system_prompt_hash  TEXT,
    benchmark_type      TEXT NOT NULL CHECK(benchmark_type IN ('academic','challenge','infrastructure')),
    benchmark_name      TEXT NOT NULL,
    topology            TEXT CHECK(topology IN ('single','planner-executor','orchestrator-coder-reviewer', NULL)),
    repetition          INTEGER DEFAULT 1 CHECK(repetition BETWEEN 1 AND 3),
    timestamp_start     TEXT NOT NULL,
    timestamp_end       TEXT,
    wall_time_s         REAL,
    tokens_used         INTEGER,
    -- CAS components (challenge runs only)
    functional_score    REAL,
    quality_score       REAL,
    completeness_score  REAL,
    efficiency_score    REAL,
    cas                 REAL,
    -- Judge scores
    judge_gpt4o_score   REAL,
    judge_claude_score  REAL,
    judge_delta         REAL,
    flagged_for_review  INTEGER DEFAULT 0,
    -- Infrastructure
    vram_peak_gb        REAL,
    power_peak_w        REAL,
    gpu_util_avg_pct    REAL,
    -- Metadata
    dgx_driver_version  TEXT,
    notes               TEXT,
    created_at          TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS academic_metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL REFERENCES runs(run_id),
    benchmark       TEXT NOT NULL,
    metric_name     TEXT NOT NULL,
    metric_value    REAL NOT NULL,
    UNIQUE(run_id, benchmark, metric_name)
);

CREATE TABLE IF NOT EXISTS infrastructure_metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL REFERENCES runs(run_id),
    metric_name     TEXT NOT NULL,
    metric_value    REAL NOT NULL,
    unit            TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS judge_outputs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL REFERENCES runs(run_id),
    judge_model     TEXT NOT NULL,
    dimension       TEXT NOT NULL,
    score           REAL NOT NULL,
    reasoning       TEXT,
    timestamp       TEXT DEFAULT (datetime('now'))
);

-- Convenience view: all challenge runs with CAS
CREATE VIEW IF NOT EXISTS challenge_results AS
SELECT
    r.run_id,
    r.model_id,
    r.model_family,
    r.model_tier,
    r.quantization,
    r.benchmark_name AS challenge,
    r.topology,
    r.repetition,
    r.wall_time_s,
    r.tokens_used,
    r.functional_score,
    r.quality_score,
    r.completeness_score,
    r.efficiency_score,
    r.cas,
    r.judge_delta,
    r.flagged_for_review
FROM runs r
WHERE r.benchmark_type = 'challenge'
ORDER BY r.model_id, r.benchmark_name, r.topology, r.repetition;

-- Convenience view: CAS mean/std per model × challenge × topology
CREATE VIEW IF NOT EXISTS cas_summary AS
SELECT
    model_id,
    model_family,
    model_tier,
    challenge,
    topology,
    COUNT(*) AS n_runs,
    ROUND(AVG(cas), 2) AS cas_mean,
    ROUND(MAX(cas) - MIN(cas), 2) AS cas_range,
    ROUND(AVG(functional_score), 2) AS functional_mean,
    ROUND(AVG(quality_score), 2) AS quality_mean,
    ROUND(AVG(completeness_score), 2) AS completeness_mean,
    ROUND(AVG(efficiency_score), 2) AS efficiency_mean
FROM challenge_results
GROUP BY model_id, model_family, model_tier, challenge, topology;
"""


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()
    print(f"Database initialised at {DB_PATH}")


def new_run_id() -> str:
    return str(uuid.uuid4())


def log_run_start(run_id: str, config: dict) -> None:
    """Call this before a benchmark run begins."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO runs (
            run_id, model_id, model_hf_repo, model_revision, model_family,
            model_tier, quantization, vllm_version, temperature, max_tokens,
            system_prompt_hash, benchmark_type, benchmark_name, topology,
            repetition, timestamp_start, dgx_driver_version, notes
        ) VALUES (
            :run_id, :model_id, :model_hf_repo, :model_revision, :model_family,
            :model_tier, :quantization, :vllm_version, :temperature, :max_tokens,
            :system_prompt_hash, :benchmark_type, :benchmark_name, :topology,
            :repetition, :timestamp_start, :dgx_driver_version, :notes
        )
    """, {**config, "run_id": run_id, "timestamp_start": datetime.utcnow().isoformat()})
    conn.commit()
    conn.close()


def log_run_end(run_id: str, results: dict) -> None:
    """Call this after a benchmark run completes."""
    conn = sqlite3.connect(DB_PATH)
    results["timestamp_end"] = datetime.utcnow().isoformat()
    results["run_id"] = run_id

    # Compute CAS if components present
    if all(k in results for k in ["functional_score","quality_score","completeness_score","efficiency_score"]):
        results["cas"] = (
            0.30 * results["functional_score"] +
            0.25 * results["quality_score"] +
            0.25 * results["completeness_score"] +
            0.20 * results["efficiency_score"]
        )
        if "judge_gpt4o_score" in results and "judge_claude_score" in results:
            results["judge_delta"] = abs(results["judge_gpt4o_score"] - results["judge_claude_score"])
            results["flagged_for_review"] = 1 if results["judge_delta"] > 15 else 0

    fields = [k for k in results if k != "run_id"]
    updates = ", ".join(f"{f} = :{f}" for f in fields)
    conn.execute(f"UPDATE runs SET {updates} WHERE run_id = :run_id", results)
    conn.commit()
    conn.close()


def log_academic_metric(run_id: str, benchmark: str, metric_name: str, value: float) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT OR REPLACE INTO academic_metrics (run_id, benchmark, metric_name, metric_value) VALUES (?,?,?,?)",
        (run_id, benchmark, metric_name, value)
    )
    conn.commit()
    conn.close()


def log_judge_output(run_id: str, judge_model: str, dimension: str, score: float, reasoning: str = "") -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO judge_outputs (run_id, judge_model, dimension, score, reasoning) VALUES (?,?,?,?,?)",
        (run_id, judge_model, dimension, score, reasoning)
    )
    conn.commit()
    conn.close()


def query(sql: str, params=()) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def serve(port: int = 8001):
    """Lightweight Flask API for notebook access."""
    from flask import Flask, jsonify, request
    app = Flask(__name__)

    @app.get("/runs")
    def get_runs():
        filters = request.args.to_dict()
        where_clauses = [f"{k} = ?" for k in filters]
        sql = "SELECT * FROM runs" + (" WHERE " + " AND ".join(where_clauses) if where_clauses else "")
        return jsonify(query(sql, list(filters.values())))

    @app.get("/challenge_results")
    def get_challenge_results():
        return jsonify(query("SELECT * FROM challenge_results"))

    @app.get("/cas_summary")
    def get_cas_summary():
        return jsonify(query("SELECT * FROM cas_summary"))

    @app.get("/academic_metrics")
    def get_academic():
        return jsonify(query("SELECT r.model_id, r.model_family, r.model_tier, a.benchmark, a.metric_name, a.metric_value FROM academic_metrics a JOIN runs r ON a.run_id = r.run_id"))

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "db": str(DB_PATH)})

    app.run(host="0.0.0.0", port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    if args.init:
        init_db()
    elif args.serve:
        serve(args.port)
    else:
        parser.print_help()
