"""Does the address-keyed kNN cache corrupt EmbedKit's own early-stopping monitor?
Runs EmbedKit default (synthetic, seed 0) with the cache AS SHIPPED (only clear_cache() before fit)
and with the cache disabled, and compares the monitored k_skewness history."""
import sys; sys.modules["faiss"] = None
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))
from datasets import make_synthetic  # noqa
import embedkit.utils.neighbors as nbrs  # noqa
from embedkit import EmbedKit  # noqa
from embedkit.analysis.hubness import HubnessAnalyzer  # noqa

# minimal repro: two different arrays, same shape, first one freed
nbrs.clear_cache()
rng = np.random.default_rng(0)
a = rng.normal(size=(500, 32)).astype(np.float32); addr_a = a.ctypes.data
s1 = HubnessAnalyzer().fit(a).k_skewness
del a
b = (rng.normal(size=(500, 32)) ** 3).astype(np.float32); addr_b = b.ctypes.data
s2 = HubnessAnalyzer().fit(b).k_skewness
nbrs.clear_cache()
s2_true = HubnessAnalyzer().fit(b).k_skewness
repro = dict(same_address=addr_a == addr_b, kskew_a=s1, kskew_b_cached=s2, kskew_b_fresh=s2_true)
print("repro", repro)

Xtr, ytr, Xte, yte, info = make_synthetic(sep=1.0, nuis_scale=2.0)
out = {"repro": repro}
for label, maxc in (("as_shipped", 8), ("cache_disabled", 0)):
    nbrs._MAX_CACHE = maxc
    nbrs.clear_cache()
    ek = EmbedKit(mode="self_supervised", random_state=0).fit(Xtr)
    h = ek._trainer.history.get("k_skewness", [])
    out[label] = dict(k_skewness_history=h, stopped_epoch=ek._trainer.stopped_epoch_,
                      n_distinct=len(set(h)))
    print(label, out[label], flush=True)
(HERE / "results_cache_check.json").write_text(json.dumps(out, indent=1, default=float))
