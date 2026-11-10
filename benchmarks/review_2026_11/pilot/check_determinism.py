"""Locate the source of run-to-run nondeterminism of a seeded Trainer.fit (synthetic, seed 0)."""
import sys; sys.modules["faiss"] = None
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))
import numpy as np, torch
from datasets import make_synthetic
import embedkit.utils.neighbors as nbrs
from embedkit.improvement.augmentation import KNNPairs, GaussianNoise
from embedkit.improvement.losses import NTXentLoss, AlignUniformLoss, CombinedLoss
from embedkit.improvement.model import EmbeddingRefiner
from embedkit.improvement.trainer import Trainer
nbrs._MAX_CACHE = 0
Xtr, *_ = make_synthetic(sep=1.0, nuis_scale=2.0)
def run(aug, threads=None):
    if threads: torch.set_num_threads(threads)
    m = EmbeddingRefiner(512, 15)
    t = Trainer(m, aug(), CombinedLoss([(NTXentLoss(0.07), 1.0), (AlignUniformLoss(), 0.5)]), epochs=3, random_state=0)
    t.fit(Xtr)
    return t.history["loss"], getattr(t.augmentation, "neighbor_index", None)
for name, aug in (("KNNPairs", lambda: KNNPairs(k=5)), ("GaussianNoise", lambda: GaussianNoise())):
    l1, n1 = run(aug); l2, n2 = run(aug)
    print(name, "loss run1", np.round(l1, 6), "run2", np.round(l2, 6),
          "nbr_idx_equal", None if n1 is None else bool(np.array_equal(n1, n2)))
l1, _ = run(lambda: KNNPairs(k=5), threads=1); l2, _ = run(lambda: KNNPairs(k=5), threads=1)
print("KNNPairs 1 thread", np.round(l1, 6), np.round(l2, 6))
# hypothesis: EmbeddingRefiner weights are drawn before Trainer.fit seeds torch -> seed torch before building the model
def run_seeded():
    torch.manual_seed(123)
    m = EmbeddingRefiner(512, 15)
    t = Trainer(m, KNNPairs(k=5), CombinedLoss([(NTXentLoss(0.07), 1.0), (AlignUniformLoss(), 0.5)]), epochs=3, random_state=0)
    t.fit(Xtr); return t.history["loss"]
print("seed torch before model build", np.round(run_seeded(), 6), np.round(run_seeded(), 6))
