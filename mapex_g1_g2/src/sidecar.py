#!/usr/bin/env python3
"""Exact V4-R4.1 as isolated Python >=3.10 worker. One persistent tracker; JSONL."""
import sys, json, os, hashlib, traceback
import numpy as np
from pathlib import Path
R41="35dd0b22411d235067ee48e859ef9018e1bb184be49034a761a83b728fea4bc9"
GEN="fb954080339c5cdf9cbfbe57b088983831f8e2aa1728d4d369cce70f64745a7d"
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def emit(o):
    print(json.dumps(o,default=lambda x: x.item() if isinstance(x,np.generic) else x.tolist() if isinstance(x,np.ndarray) else str(x)),flush=True)
def main():
    root=Path(__file__).resolve().parent/"v41src"
    if sha(root/"v4r41.py")!=R41 or sha(root/"v4r4_generator_reference.py")!=GEN:
        emit({"event":"FATAL","reason":"EXACT_R41_HASH_MISMATCH"});return 3
    sys.path.insert(0,str(root))
    from v4r41 import infer
    from v4r4_generator_reference import RoomHypothesisTracker
    tracker=RoomHypothesisTracker()
    emit({"event":"READY","r41_sha256":R41,"generator_sha256":GEN})
    last=0
    for line in sys.stdin:
        req={}
        try:
            req=json.loads(line)
            if req.get("action")=="CLOSE": emit({"event":"BYE"});return 0
            if req.get("action")!="INFER":raise ValueError("unknown request")
            decision=int(req["decision"])
            if decision<=last:raise ValueError("nonmonotonic decision")
            src=Path(req["input"]);dest=Path(req["output"]);report=Path(req["report"])
            if dest.exists() or report.exists():raise FileExistsError("output already exists")
            with np.load(src,allow_pickle=False) as z:
                obs=z["observed"].copy();mean=z["mean"].copy();var=z["variance"].copy()
            if obs.shape!=mean.shape or obs.shape!=var.shape:raise ValueError("input grid mismatch")
            if obs.ndim!=2:raise ValueError("wrong dimensions")
            out=infer(obs,mean,var,resolution_m=float(req["resolution"]),tracker=tracker,snapshot_id="native_r41_%06d"%decision)
            assert out["semantic_r41"].shape==obs.shape
            tmp=str(dest)+".partial.npz"
            np.savez_compressed(tmp,semantic=out["semantic_r41"],baseline=out["semantic_r3"],hypotheses=out["hypothesis_mask"])
            os.replace(tmp,dest)
            payload={"decision":decision,"run_mode":"LIVE_CAUSAL_R41","input_sha256":sha(src),
              "semantic_sha256":sha(dest),"source_r41_sha256":R41,
              "report":out["report"],"room_hypotheses":out["hypotheses"],"events":out["events"],"candidate_audit":out["audit"]}
            tmp=str(report)+".partial"
            with open(tmp,"w",encoding="utf8") as f:json.dump(payload,f,indent=2,ensure_ascii=False,default=lambda x: x.item() if isinstance(x,np.generic) else x.tolist() if isinstance(x,np.ndarray) else str(x))
            os.replace(tmp,report)
            last=decision
            emit({"event":"OK","decision":decision,"semantic_sha256":sha(dest)})
        except Exception as e:
            emit({"event":"ERROR","decision":req.get("decision"),"error":repr(e),"traceback":traceback.format_exc()})
            return 2
    return 0
if __name__=="__main__":sys.exit(main())
