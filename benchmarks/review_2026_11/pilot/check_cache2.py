"""Determinism / cache check: one EmbedKit default fit (synthetic, seed 0) per process.
argv[1] = max cache size (8 = as shipped, 0 = disabled). Prints target_dim, ID estimates, k_skewness history."""
import sys; sys.modules["faiss"] = None
import json
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))
from datasets import make_synthetic
import embedkit.utils.neighbors as nbrs
from embedkit import EmbedKit
nbrs._MAX_CACHE = int(sys.argv[1]); nbrs.clear_cache()
Xtr, ytr, Xte, yte, info = make_synthetic(sep=1.0, nuis_scale=2.0)
ek = EmbedKit(mode="self_supervised", random_state=0).fit(Xtr)
print(json.dumps(dict(maxcache=nbrs._MAX_CACHE, target_dim=ek._config["target_dim"],
      id=ek.analysis_report.intrinsic_dim.estimates, stopped=ek._trainer.stopped_epoch_,
      kskew=[round(v, 4) for v in ek._trainer.history["k_skewness"]]), default=float))
