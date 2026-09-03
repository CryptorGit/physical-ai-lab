"""Reproduce the A4 V2 candidate without persisting a policy checkpoint."""
from __future__ import annotations
import argparse,json,subprocess,sys,tempfile
from pathlib import Path
import torch
from w2_p1_a5_common import A5,reproduce_a4

OUT=A5.parent/"phase_w2_p1_a6_rear_yaw_acquisition_diagnosis"

def once():
    _,fingerprint,_,_,_=reproduce_a4(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return fingerprint

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--child-output");args=ap.parse_args()
    if args.child_output:
        Path(args.child_output).write_text(json.dumps(once(),sort_keys=True)+"\n",encoding="utf-8");return
    same=[once(),once()]
    with tempfile.TemporaryDirectory(prefix="exp013_a6_") as td:
        target=Path(td)/"fresh.json"
        subprocess.run([sys.executable,__file__,"--child-output",str(target)],check=True)
        fresh=json.loads(target.read_text(encoding="utf-8"))
    expected="db65a3069d665b8012fd9d264b7fd54e629a22d25b05a9ff793e23bfc549ac5f"
    payload={"same_process":same,"fresh_process":fresh,"expected_tensor_hash":expected,
             "tensor_hash_exact":all(x["tensor_hash"]==expected for x in [*same,fresh]),
             "trace_hash_exact":len({x["trace_hash"] for x in [*same,fresh]})==1,
             "persistent_checkpoint_created":0}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"a4_candidate_reproduction.json").write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")

if __name__=="__main__":main()
