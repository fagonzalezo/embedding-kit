import sys; sys.modules["faiss"] = None  # force sklearn backend: faiss+torch segfaults in this env
import warnings, numpy as np, torch
warnings.simplefilter("ignore")
from embedkit.utils.neighbors import knn, clear_cache
from embedkit.analysis.geometry import NeighborConsistency, IsotropyAnalyzer, DistanceConcentration
from embedkit.improvement.losses import NTXentLoss, TripletLoss
from embedkit.improvement.augmentation import KNNPairs, GaussianNoise, CompositeAugmentation, EmbeddingMixup

rng = np.random.default_rng(0)

print("--- 1. kNN cache keyed on memory address: stale hits")
stale = 0; trials = 50
for t in range(trials):
    clear_cache()
    A = rng.standard_normal((500, 32)).astype(np.float32)
    addr = A.ctypes.data
    _, ia = knn(A, 5)
    del A
    B = rng.standard_normal((500, 32)).astype(np.float32)
    same_addr = B.ctypes.data == addr
    _, ib = knn(B, 5)
    clear_cache()
    _, ib_true = knn(B, 5)
    if not np.array_equal(ib, ib_true):
        stale += 1
print(f"stale results in {stale}/{trials} fresh-array trials")
clear_cache()
A = rng.standard_normal((500, 32)).astype(np.float32)
_, i1 = knn(A, 5); A[:] = rng.standard_normal((500, 32)); _, i2 = knn(A, 5)
print("in-place mutation returns stale neighbours:", np.array_equal(i1, i2))

print("--- 1b. NeighborConsistency per_perturbation (identical values => stale cache)")
clear_cache()
X = rng.standard_normal((1500, 64)).astype(np.float32)
r = NeighborConsistency(k=10, noise_std=0.3).fit(X)
print(r.per_perturbation, "std", r.std_consistency)

print("--- 3. NT-Xent chunked vs full")
torch.manual_seed(0)
z1, z2 = torch.randn(50, 8), torch.randn(50, 8)
full = NTXentLoss(0.1)(z1, z2).item()
for cs in [64, 99, 50, 30]:
    try: print("chunk", cs, NTXentLoss(0.1, chunk_size=cs)(z1, z2).item(), "full", full)
    except Exception as e: print("chunk", cs, "CRASH:", str(e)[:90])
print("--- 4. Triplet euclidean gradient")
for mining in ["hard", "semi-hard", "easy"]:
    z1 = torch.randn(16, 8, requires_grad=True); z2 = torch.randn(16, 8)
    y = torch.randint(0, 3, (16,))
    l = TripletLoss(mining=mining)(z1, z2, y); l.backward()
    print(mining, "loss", l.item(), "grad NaN:", torch.isnan(z1.grad).any().item())

print("--- 5. Composite [GaussianNoise, KNNPairs]: is noise on view j discarded?")
Xn = rng.standard_normal((300, 16)).astype(np.float32)
kp = KNNPairs(k=5); comp = CompositeAugmentation([GaussianNoise(std=1.0), kp]); comp.precompute(Xn)
xb = torch.from_numpy(Xn[:32]); idx = torch.arange(32)
xi, xj = comp(xb, idx)
ref = torch.from_numpy(Xn)
print("view j is an exact dataset row (noise dropped):", bool(((xj[:, None, :] - ref[None]).abs().sum(-1).min(1).values < 1e-6).all()))
print("view i is noisy:", bool((xi - xb).abs().sum() > 0))

print("--- 6. Isotropy across d_max threshold (same isotropic Gaussian, n=3000)")
for d in [1000, 2500]:
    Xg = rng.standard_normal((3000, d)).astype(np.float32)
    r = IsotropyAnalyzer().fit(Xg)
    print(d, "PR", round(r.participation_ratio, 1), "PR/d", round(r.participation_ratio / d, 3), "eff_rank", round(r.effective_rank, 1), "isotropy_score", round(r.isotropy_score, 3), "n_eigs", len(r.eigenvalue_spectrum))

print("--- 7. concentration_ratio for iid Gaussians (never reaches 0.8?) and one duplicate")
for d in [16, 256, 2048]:
    Xg = rng.standard_normal((2000, d)).astype(np.float32)
    r = DistanceConcentration().fit(Xg); print(d, "conc_ratio", round(r.concentration_ratio, 3), "rel_contrast", round(r.relative_contrast, 3))
Xg[1] = Xg[0]; print("with one duplicate:", round(DistanceConcentration().fit(Xg).concentration_ratio, 3))

print("--- 8. Hubness skewness null: iid Gaussian vs n (k=10 val-split sizes)")
from embedkit.analysis.hubness import HubnessAnalyzer
for n in [100, 368, 1000, 5000]:
    vals = []
    for s in range(8):
        clear_cache()
        Xg = np.random.default_rng(s).standard_normal((n, 64)).astype(np.float32)
        vals.append(HubnessAnalyzer(k=10).fit(Xg).k_skewness)
    print(n, "mean", round(np.mean(vals), 2), "sd across seeds", round(np.std(vals), 2))

print("--- 9. GaussianNoise(std=0.05) vs per-coordinate std of unit-norm 768-d embeddings")
E = rng.standard_normal((1000, 768)); E /= np.linalg.norm(E, axis=1, keepdims=True)
print("per-coord std", E.std().round(4), "-> noise/signal norm ratio", round(0.05 * np.sqrt(768), 2))

print("--- 10. StepLR stepped per batch; isotropy monitor direction; restore-best")
from embedkit.improvement import EmbeddingRefiner, Trainer
Xs = rng.standard_normal((1024, 32)).astype(np.float32)
t = Trainer(EmbeddingRefiner(32, 8), GaussianNoise(0.1), NTXentLoss(0.5), epochs=30, batch_size=64, scheduler="step", lr=1e-3)
opt = t._build_optimizer(); sch = t._build_scheduler(opt, 16)
for _ in range(30 * 16): opt.step(); sch.step()
print("lr after 30 epochs with StepLR(step_size=10) stepped per batch:", opt.param_groups[0]["lr"], "(intended 1e-6)")
