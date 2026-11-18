# Updated for 0.1.3: centred kernel alignment, encode_labels, MONITORS registry.
import warnings; warnings.simplefilter("ignore")
import numpy as np, torch
from embedkit.analysis.kernel import KernelDiagnostics
rng = np.random.default_rng(0)
X = rng.standard_normal((600, 32)).astype(np.float32)
print("--- kernel alignment, labels independent of X (should be ~0 if centred)")
for c in [2, 10]:
    y = rng.integers(0, c, 600)
    print(c, "classes:", round(KernelDiagnostics().fit(X, y).centered_kernel_alignment, 3))
yc = rng.standard_normal(600)
print("continuous 1-D target independent of X (RBF label kernel):", KernelDiagnostics().fit(X, yc).centered_kernel_alignment)
for name, y in [("string labels", np.array(["a", "b"] * 300)), ("python list, n>subsample", list(rng.integers(0, 3, 5000)))]:
    try:
        Xb = X if len(y) == 600 else rng.standard_normal((5000, 8)).astype(np.float32)
        KernelDiagnostics().fit(Xb, y); print(name, "ok")
    except Exception as e: print(name, "CRASH:", type(e).__name__, str(e)[:80])
print("--- kernel spectrum: top-50 truncation")
r = KernelDiagnostics().fit(X); print("eff_rank", round(r.effective_rank, 2), "of max 50 | spectral_gap", round(r.spectral_gap, 1), "| cond", f"{r.condition_number:.1f}")
print("--- supervised Trainer with string labels / device")
from embedkit import EmbedKit
from embedkit.improvement import Trainer, EmbeddingRefiner, GaussianNoise, NTXentLoss, SupConLoss
t = Trainer(EmbeddingRefiner(32, 8), GaussianNoise(0.1), SupConLoss(), epochs=1, eval_every=1000)
print("device chosen:", t.device, "| mps available:", torch.backends.mps.is_available())
try: t.fit(X, np.array(["a", "b"] * 300)); print("string labels ok")
except Exception as e: print("string labels CRASH:", type(e).__name__, str(e)[:90])
print("--- early stopping: best weights restored? monitor=isotropy direction")
import inspect, embedkit.improvement.trainer as tr
src = inspect.getsource(tr.Trainer.fit)
print("best weights restored:", "_copy_state" in src, "| warmup_epochs used:", "warmup" in inspect.getsource(tr.Trainer._build_schedulers), "| isotropy direction:", tr.MONITORS["isotropy"][1])
print("--- auto-config: which report fields does api.py consume?")
import embedkit.api as api
s = inspect.getsource(api)
for f in ["suggested_k", "suggested_sigma", "suggested_target_dim", "severity", "recommendations", "hubness", "isotropy"]:
    print(f, f"report.{f}" in s)
