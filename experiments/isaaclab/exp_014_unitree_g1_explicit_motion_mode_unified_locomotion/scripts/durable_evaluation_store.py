"""Transactional, simulation-independent persistence for held-out evaluations."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


class DurableEvaluationStore:
    """SQLite WAL store owned by the persistence parent, never the simulation worker."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS run_manifest (
              run_id TEXT PRIMARY KEY, candidate_sha TEXT NOT NULL,
              sealed_sha TEXT NOT NULL, contract_version TEXT NOT NULL,
              expected_count INTEGER NOT NULL, manifest_json BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS episodes (
              run_id TEXT NOT NULL, episode_id TEXT NOT NULL,
              condition_id INTEGER NOT NULL, status TEXT NOT NULL,
              attempt_count INTEGER NOT NULL DEFAULT 0,
              PRIMARY KEY(run_id, episode_id),
              FOREIGN KEY(run_id) REFERENCES run_manifest(run_id));
            CREATE TABLE IF NOT EXISTS episode_results (
              run_id TEXT NOT NULL, episode_id TEXT NOT NULL, result_json BLOB NOT NULL,
              PRIMARY KEY(run_id, episode_id),
              FOREIGN KEY(run_id, episode_id) REFERENCES episodes(run_id, episode_id));
            CREATE TABLE IF NOT EXISTS result_hashes (
              run_id TEXT NOT NULL, episode_id TEXT NOT NULL, sha256 TEXT NOT NULL,
              PRIMARY KEY(run_id, episode_id),
              FOREIGN KEY(run_id, episode_id) REFERENCES episodes(run_id, episode_id));
            CREATE TABLE IF NOT EXISTS access_ledger (
              sequence INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
              episode_id TEXT, event TEXT NOT NULL, detail_json BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS process_events (
              sequence INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
              worker_id TEXT, event TEXT NOT NULL, detail_json BLOB NOT NULL);
            """
        )

    def close(self): self.db.close()

    def create_run(self, run_id, candidate_sha, sealed_sha, contract_version, episodes):
        manifest = {"run_id": run_id, "candidate_sha": candidate_sha, "sealed_sha": sealed_sha,
                    "contract_version": contract_version, "episodes": episodes}
        with self.db:
            self.db.execute("INSERT INTO run_manifest VALUES(?,?,?,?,?,?)",
                            (run_id, candidate_sha, sealed_sha, contract_version, len(episodes), canonical(manifest)))
            self.db.executemany("INSERT INTO episodes(run_id,episode_id,condition_id,status) VALUES(?,?,?,'PENDING')",
                                [(run_id, item["episode_id"], item["condition_id"]) for item in episodes])
            self.db.execute("INSERT INTO access_ledger(run_id,event,detail_json) VALUES(?,?,?)",
                            (run_id, "RUN_CREATED", canonical({"expected_count": len(episodes)})))

    def start_episode(self, run_id, episode_id, worker_id="worker"):
        with self.db:
            row = self.db.execute("SELECT status FROM episodes WHERE run_id=? AND episode_id=?", (run_id, episode_id)).fetchone()
            if row is None: raise KeyError(episode_id)
            if row[0] == "COMPLETED": raise RuntimeError("completed episode may not be rerun")
            self.db.execute("UPDATE episodes SET status='STARTED',attempt_count=attempt_count+1 WHERE run_id=? AND episode_id=?", (run_id, episode_id))
            self.db.execute("INSERT INTO access_ledger(run_id,episode_id,event,detail_json) VALUES(?,?,?,?)", (run_id, episode_id, "EPISODE_STARTED", canonical({"worker_id": worker_id})))

    def commit_result(self, run_id, episode_id, result, provenance, inject_before_commit=False):
        """Atomically commits result, hash, and COMPLETED status in one FULL transaction."""
        payload = {"result": result, "provenance": provenance}; data = canonical(payload); digest = hashlib.sha256(data).hexdigest()
        try:
            self.db.execute("BEGIN IMMEDIATE")
            status = self.db.execute("SELECT status FROM episodes WHERE run_id=? AND episode_id=?", (run_id, episode_id)).fetchone()
            if status is None or status[0] != "STARTED": raise RuntimeError("episode must be STARTED")
            required = ("candidate_sha", "sealed_sha", "contract_version", "code_version")
            if any(not provenance.get(key) for key in required): raise ValueError("incomplete provenance")
            self.db.execute("INSERT INTO episode_results VALUES(?,?,?)", (run_id, episode_id, data))
            self.db.execute("INSERT INTO result_hashes VALUES(?,?,?)", (run_id, episode_id, digest))
            self.db.execute("UPDATE episodes SET status='COMPLETED' WHERE run_id=? AND episode_id=?", (run_id, episode_id))
            self.db.execute("INSERT INTO access_ledger(run_id,episode_id,event,detail_json) VALUES(?,?,?,?)", (run_id, episode_id, "EPISODE_COMPLETED", canonical({"result_sha256": digest})))
            if inject_before_commit: raise RuntimeError("injected transaction crash")
            self.db.commit()
        except BaseException:
            self.db.rollback(); raise
        return digest

    def inject_orphan_result_for_repair_test(self, run_id, episode_id, result, provenance):
        data = canonical({"result": result, "provenance": provenance}); digest = hashlib.sha256(data).hexdigest()
        with self.db:
            self.db.execute("INSERT INTO episode_results VALUES(?,?,?)", (run_id, episode_id, data))
            self.db.execute("INSERT INTO result_hashes VALUES(?,?,?)", (run_id, episode_id, digest))

    def validate_and_repair(self, run_id):
        completed_without = self.db.execute("SELECT COUNT(*) FROM episodes e LEFT JOIN episode_results r USING(run_id,episode_id) WHERE e.run_id=? AND e.status='COMPLETED' AND r.episode_id IS NULL", (run_id,)).fetchone()[0]
        orphans = self.db.execute("SELECT e.episode_id FROM episodes e JOIN episode_results r USING(run_id,episode_id) WHERE e.run_id=? AND e.status!='COMPLETED'", (run_id,)).fetchall()
        if completed_without: raise RuntimeError("corrupt transaction: completed ledger without durable result")
        with self.db:
            for (episode_id,) in orphans:
                self.db.execute("UPDATE episodes SET status='COMPLETED' WHERE run_id=? AND episode_id=?", (run_id, episode_id))
                self.db.execute("INSERT INTO access_ledger(run_id,episode_id,event,detail_json) VALUES(?,?,?,?)", (run_id, episode_id, "LEDGER_REPAIRED_FROM_DURABLE_RESULT", canonical({})))
        return [row[0] for row in orphans]

    def invariants(self, run_id):
        completed_without = self.db.execute("SELECT COUNT(*) FROM episodes e LEFT JOIN episode_results r USING(run_id,episode_id) WHERE e.run_id=? AND e.status='COMPLETED' AND r.episode_id IS NULL", (run_id,)).fetchone()[0]
        duplicate = self.db.execute("SELECT COUNT(*) FROM (SELECT episode_id,COUNT(*) n FROM episode_results WHERE run_id=? GROUP BY episode_id HAVING n>1)", (run_id,)).fetchone()[0]
        missing_hash = self.db.execute("SELECT COUNT(*) FROM episode_results r LEFT JOIN result_hashes h USING(run_id,episode_id) WHERE r.run_id=? AND h.episode_id IS NULL", (run_id,)).fetchone()[0]
        return {"completed_without_result": completed_without, "duplicate_episode_result": duplicate,
                "missing_completed_result": completed_without, "result_without_provenance": missing_hash}

    def aggregate(self, run_id):
        expected = self.db.execute("SELECT expected_count FROM run_manifest WHERE run_id=?", (run_id,)).fetchone()[0]
        rows = self.db.execute("SELECT e.episode_id,e.condition_id,r.result_json FROM episodes e JOIN episode_results r USING(run_id,episode_id) WHERE e.run_id=? AND e.status='COMPLETED' ORDER BY e.episode_id", (run_id,)).fetchall()
        if len(rows) != expected: raise RuntimeError("aggregate from incomplete set prohibited")
        decoded = [(episode_id, condition, json.loads(blob)["result"]) for episode_id, condition, blob in rows]
        conditions = {}
        for condition in sorted({item[1] for item in decoded}):
            subset = [item[2] for item in decoded if item[1] == condition]
            conditions[str(condition)] = {"episodes": len(subset), "joint_success": sum(bool(x["joint_success"]) for x in subset) / len(subset)}
        output = {"episodes": len(decoded), "joint_success": sum(bool(item[2]["joint_success"]) for item in decoded) / len(decoded), "conditions": conditions}
        return canonical(output)
