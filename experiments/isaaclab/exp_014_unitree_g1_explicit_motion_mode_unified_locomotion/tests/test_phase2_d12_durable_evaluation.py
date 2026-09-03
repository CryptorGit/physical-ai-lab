from __future__ import annotations

import importlib.util
import hashlib
import json
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve(); SCRIPT = HERE.parents[1] / "scripts/durable_evaluation_store.py"
spec = importlib.util.spec_from_file_location("durable_store", SCRIPT); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
Store = mod.DurableEvaluationStore
util_spec = importlib.util.spec_from_file_location("forensic_utils", HERE.parents[1] / "scripts/d12_forensic_utils.py"); util = importlib.util.module_from_spec(util_spec); util_spec.loader.exec_module(util)
PROV = {"candidate_sha": "candidate", "sealed_sha": "sealed", "contract_version": "v1", "code_version": "commit"}


class DurableEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.path = Path(self.tmp.name) / "results.sqlite"; self.store = Store(self.path)
        self.store.create_run("run", "candidate", "sealed", "v1", [{"episode_id": "e0", "condition_id": 0}, {"episode_id": "e1", "condition_id": 1}])

    def tearDown(self): self.store.close(); self.tmp.cleanup()
    def result(self, value=True): return {"joint_success": value}

    def test_crash1_after_sim_before_result_is_resumable(self):
        self.store.start_episode("run", "e0"); self.assertEqual(self.store.db.execute("SELECT status FROM episodes WHERE episode_id='e0'").fetchone()[0], "STARTED")

    def test_crash2_transaction_before_commit_rolls_back(self):
        self.store.start_episode("run", "e0")
        with self.assertRaises(RuntimeError): self.store.commit_result("run", "e0", self.result(), PROV, inject_before_commit=True)
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM episode_results").fetchone()[0], 0)

    def test_crash3_durable_orphan_repairs_without_rerun(self):
        self.store.start_episode("run", "e0"); self.store.inject_orphan_result_for_repair_test("run", "e0", self.result(), PROV)
        self.assertEqual(self.store.validate_and_repair("run"), ["e0"])

    def test_crash4_completed_always_has_result(self):
        self.store.start_episode("run", "e0"); self.store.commit_result("run", "e0", self.result(), PROV)
        self.assertEqual(self.store.invariants("run")["completed_without_result"], 0)

    def test_crash5_teardown_after_batch_preserves_all(self):
        for episode in ("e0", "e1"):
            self.store.start_episode("run", episode); self.store.commit_result("run", episode, self.result(), PROV)
        self.store.close(); self.store = Store(self.path); self.assertEqual(json.loads(self.store.aggregate("run"))["episodes"], 2)

    def test_crash6_aggregate_is_regenerable_bitwise(self):
        for episode in ("e0", "e1"):
            self.store.start_episode("run", episode); self.store.commit_result("run", episode, self.result(), PROV)
        self.assertEqual(self.store.aggregate("run"), self.store.aggregate("run"))

    def test_duplicate_result_rejected(self):
        self.store.start_episode("run", "e0"); self.store.commit_result("run", "e0", self.result(), PROV)
        with self.assertRaises(RuntimeError): self.store.start_episode("run", "e0")

    def test_incomplete_aggregate_is_prohibited(self):
        with self.assertRaises(RuntimeError): self.store.aggregate("run")

    def test_missing_provenance_rolls_back(self):
        self.store.start_episode("run", "e0")
        with self.assertRaises(ValueError): self.store.commit_result("run", "e0", self.result(), {}, inject_before_commit=False)
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM episode_results").fetchone()[0], 0)

    def test_corrupt_completed_without_result_fails_closed(self):
        with self.store.db: self.store.db.execute("UPDATE episodes SET status='COMPLETED' WHERE episode_id='e0'")
        with self.assertRaises(RuntimeError): self.store.validate_and_repair("run")

    def test_resume_started_episode_allowed(self):
        self.store.start_episode("run", "e0"); self.store.start_episode("run", "e0")
        self.assertEqual(self.store.db.execute("SELECT attempt_count FROM episodes WHERE episode_id='e0'").fetchone()[0], 2)

    def test_required_invariants_zero_after_valid_commit(self):
        self.store.start_episode("run", "e0"); self.store.commit_result("run", "e0", self.result(), PROV)
        self.assertEqual(self.store.invariants("run"), {"completed_without_result": 0, "duplicate_episode_result": 0, "missing_completed_result": 0, "result_without_provenance": 0})


class ForensicAndProtectionTests(unittest.TestCase):
    def test_forensic_window_parser(self):
        with tempfile.NamedTemporaryFile() as stream:
            stat = Path(stream.name).stat(); self.assertTrue(util.within_window(stat, stat.st_mtime - 1, stat.st_mtime + 1))

    def test_forensic_hash_is_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.bin"; path.write_bytes(b"evidence"); before = path.stat().st_mtime_ns
            self.assertEqual(util.sha256_file(path), hashlib.sha256(b"evidence").hexdigest())
            self.assertEqual(before, path.stat().st_mtime_ns)

    def test_forensic_relevance_classifier(self):
        self.assertEqual(util.relevance(Path("phase_2_d11_stop_student_heldout/ledger.json")), "DIRECT_D11")

    def test_protected_d6_d11_committed_paths_unchanged(self):
        repo = HERE.parents[4]
        paths = [f"results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d{i}*" for i in range(6, 12)]
        output = subprocess.check_output(["git", "diff", "--name-only", "88460496e0b10550649390b46c6248e8c7b2c5b7", "--", *paths], cwd=repo, text=True)
        self.assertEqual(output.strip(), "")


if __name__ == "__main__": unittest.main()
