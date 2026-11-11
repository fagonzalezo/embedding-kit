"""Download a benchmark dataset and cache its embeddings under the data root. Cluster only.

    python benchmarks/extract_features.py --dataset cifar100_dinov2 --data-root $DATA --device cuda

This is the only script in the benchmark that downloads anything (raw data into
``<data-root>/raw``, models into ``$HF_HOME``). It writes ``<data-root>/features/<name>.npz``
(``X_train``, ``y_train``, ``X_test``, ``y_test`` and, depending on the set, ``y_extra_*`` or the
STS pairs) and ``<name>.json`` with the provenance: model, revision, package versions, date.
``benchmarks/bench_datasets.py`` reads these files. Do not run it on a laptop: large data are
kept on the cluster (see ``benchmarks/README.md``).

Sets:
- ``cifar100_dinov2``: CIFAR-100 (50k / 10k), DINOv2 ViT-B/14 CLS embedding (768-d); fine labels,
  coarse (superclass) labels as ``y_extra_coarse``.
- ``stl10_clip``: STL-10 labelled split (5k / 8k), CLIP ViT-B/32 image embedding (512-d).
- ``banking77_mpnet``: Banking77 intents (10k / 3k, 77 classes), all-mpnet-base-v2 (768-d).
- ``stsb_mpnet``: STS-B. ``X_train`` holds the unique sentences of the train split (for fitting a
  transform, unlabelled); ``X_test`` the unique test sentences, with ``pairs_a``, ``pairs_b``
  (indices into ``X_test``) and gold ``scores``; the validation split likewise as ``X_val``,
  ``val_pairs_a``, ``val_pairs_b``, ``val_scores`` (used for selection).
- ``glove100``, ``sift128``, ``gist960``: ANN-Benchmarks vectors (train / test as published),
  unlabelled.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pickle
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bench_datasets as D  # noqa: E402

MODELS = {
    "dinov2": "facebook/dinov2-base",
    "clip": "openai/clip-vit-base-patch32",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
}
ANN = {
    "glove100": "glove-100-angular",
    "sift128": "sift-128-euclidean",
    "gist960": "gist-960-euclidean",
}


def _versions() -> dict:
    out = {}
    for pkg in ("torch", "torchvision", "transformers", "sentence_transformers", "datasets",
                "numpy", "h5py"):
        try:
            out[pkg] = __import__(pkg).__version__
        except Exception:
            pass
    return out


def _hf_revision(model_id: str) -> str | None:
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(model_id).sha
    except Exception:
        return None


# --- images -------------------------------------------------------------------------------------


def _collate_images(batch):
    return [x for x, _ in batch], np.array([y for _, y in batch])


def _embed_images(dataset, encode, batch_size: int, device: str):
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                        collate_fn=_collate_images)
    feats, labels = [], []
    with torch.inference_mode():
        for images, y in loader:
            feats.append(encode(images).float().cpu().numpy())
            labels.append(y)
    return np.concatenate(feats).astype(np.float32), np.concatenate(labels).astype(np.int64)


def _dinov2_encoder(device: str):
    import torch
    from transformers import AutoImageProcessor, AutoModel

    proc = AutoImageProcessor.from_pretrained(MODELS["dinov2"])
    model = AutoModel.from_pretrained(MODELS["dinov2"]).to(device).eval()

    def encode(images):
        inputs = proc(images=images, return_tensors="pt").to(device)
        with torch.autocast(device_type="cuda", enabled=device.startswith("cuda")):
            return model(**inputs).pooler_output  # layer-normed CLS token

    return encode


def _clip_encoder(device: str):
    import torch
    from transformers import CLIPModel, CLIPProcessor

    proc = CLIPProcessor.from_pretrained(MODELS["clip"])
    model = CLIPModel.from_pretrained(MODELS["clip"]).to(device).eval()

    def encode(images):
        inputs = proc(images=images, return_tensors="pt").to(device)
        with torch.autocast(device_type="cuda", enabled=device.startswith("cuda")):
            out = model.get_image_features(**inputs)
        # transformers >= 5 returns a model output whose pooler_output is the projected embedding.
        return out if isinstance(out, torch.Tensor) else out.pooler_output

    return encode


def cifar100_dinov2(root: Path, device: str, batch_size: int) -> tuple[dict, dict]:
    from torchvision.datasets import CIFAR100

    raw = root / "raw"
    train = CIFAR100(raw, train=True, download=True)
    test = CIFAR100(raw, train=False, download=True)
    encode = _dinov2_encoder(device)
    Xtr, ytr = _embed_images(train, encode, batch_size, device)
    Xte, yte = _embed_images(test, encode, batch_size, device)
    coarse = {}
    for split in ("train", "test"):
        with open(raw / "cifar-100-python" / split, "rb") as f:
            coarse[split] = np.asarray(pickle.load(f, encoding="latin1")["coarse_labels"])
    arrays = dict(X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte,
                  y_extra_coarse_train=coarse["train"], y_extra_coarse_test=coarse["test"])
    meta = dict(model=MODELS["dinov2"], embedding="pooler_output (CLS)", n_classes=100,
                n_coarse_classes=20, images="CIFAR-100 32x32, resized by the DINOv2 processor")
    return arrays, meta


# The original STL-10 host (ai.stanford.edu) is unreachable; this mirror has the same splits
# (5000 train, 8000 test) and the same label order as torchvision's STL10.
STL10_HF = "tanganke/stl10"
STL10_CLASSES = ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship",
                 "truck"]


class _HFImages:
    """(PIL image, label) pairs from a Hugging Face split, in its stored order."""

    def __init__(self, split):
        self.split = split

    def __len__(self):
        return len(self.split)

    def __getitem__(self, i):
        row = self.split[i]
        return row["image"].convert("RGB"), int(row["label"])


def stl10_clip(root: Path, device: str, batch_size: int) -> tuple[dict, dict]:
    from datasets import load_dataset
    from huggingface_hub import HfApi

    ds = load_dataset(STL10_HF, cache_dir=str(root / "raw" / "hf_datasets"))
    names = ds["train"].features["label"].names
    if names != STL10_CLASSES:
        raise RuntimeError(f"unexpected STL-10 label names: {names}")
    encode = _clip_encoder(device)
    Xtr, ytr = _embed_images(_HFImages(ds["train"]), encode, batch_size, device)
    Xte, yte = _embed_images(_HFImages(ds["test"]), encode, batch_size, device)
    try:
        revision = HfApi().dataset_info(STL10_HF).sha
    except Exception:
        revision = None
    meta = dict(model=MODELS["clip"], embedding="get_image_features (projected)", n_classes=10,
                images=f"STL-10 96x96 from the {STL10_HF} mirror", dataset_revision=revision)
    return dict(X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte), meta


# --- text ---------------------------------------------------------------------------------------


def _sentence_encoder(device: str):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODELS["mpnet"], device=device)

    def encode(texts, batch_size):
        # Raw (unnormalised) embeddings: normalisation is one of the transforms under test.
        return model.encode(list(texts), batch_size=batch_size, convert_to_numpy=True,
                            normalize_embeddings=False, show_progress_bar=False).astype(np.float32)

    return encode


def banking77_mpnet(root: Path, device: str, batch_size: int) -> tuple[dict, dict]:
    from datasets import load_dataset

    ds = load_dataset("mteb/banking77", cache_dir=str(root / "raw" / "hf_datasets"))
    encode = _sentence_encoder(device)
    arrays = dict(
        X_train=encode(ds["train"]["text"], batch_size),
        y_train=np.asarray(ds["train"]["label"], dtype=np.int64),
        X_test=encode(ds["test"]["text"], batch_size),
        y_test=np.asarray(ds["test"]["label"], dtype=np.int64),
    )
    return arrays, dict(model=MODELS["mpnet"], hf_dataset="mteb/banking77", n_classes=77,
                        normalized=False)


def stsb_mpnet(root: Path, device: str, batch_size: int) -> tuple[dict, dict]:
    from datasets import load_dataset

    ds = load_dataset("sentence-transformers/stsb", cache_dir=str(root / "raw" / "hf_datasets"))
    encode = _sentence_encoder(device)
    train_sents = sorted(set(ds["train"]["sentence1"]) | set(ds["train"]["sentence2"]))
    arrays = dict(X_train=encode(train_sents, batch_size))
    for split, prefix in (("validation", "val_"), ("test", "")):
        part = ds[split]
        sents = sorted(set(part["sentence1"]) | set(part["sentence2"]))
        index = {s: i for i, s in enumerate(sents)}
        arrays["X_val" if prefix else "X_test"] = encode(sents, batch_size)
        arrays[prefix + "pairs_a"] = np.array([index[s] for s in part["sentence1"]], np.int64)
        arrays[prefix + "pairs_b"] = np.array([index[s] for s in part["sentence2"]], np.int64)
        arrays[prefix + "scores"] = np.asarray(part["score"], dtype=np.float64)
    return arrays, dict(model=MODELS["mpnet"], hf_dataset="sentence-transformers/stsb",
                        normalized=False, n_test_pairs=len(ds["test"]),
                        n_val_pairs=len(ds["validation"]))


# --- ANN-Benchmarks -----------------------------------------------------------------------------


def _ann(name: str):
    def build(root: Path, device: str, batch_size: int) -> tuple[dict, dict]:
        import h5py

        fname = f"{ANN[name]}.hdf5"
        path = root / "raw" / "ann-benchmarks" / fname
        url = f"http://ann-benchmarks.com/{fname}"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".part")
            # ann-benchmarks.com answers 403 to urllib's default User-Agent.
            req = urllib.request.Request(url, headers={"User-Agent": "embedkit-bench"})
            with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as f:
                shutil.copyfileobj(resp, f, length=1 << 20)
            tmp.replace(path)
        with h5py.File(path, "r") as f:
            arrays = dict(X_train=np.asarray(f["train"], dtype=np.float32),
                          X_test=np.asarray(f["test"], dtype=np.float32))
        return arrays, dict(source_url=url, metric=ANN[name].rsplit("-", 1)[1])

    return build


BUILDERS = {
    "cifar100_dinov2": cifar100_dinov2,
    "stl10_clip": stl10_clip,
    "banking77_mpnet": banking77_mpnet,
    "stsb_mpnet": stsb_mpnet,
    **{name: _ann(name) for name in ANN},
}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", required=True, choices=sorted(BUILDERS))
    p.add_argument("--data-root", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    root = D.data_root(args.data_root)
    out = root / "features" / f"{args.dataset}.npz"
    if out.exists() and not args.overwrite:
        print(f"{out} exists; nothing to do (use --overwrite)")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    arrays, meta = BUILDERS[args.dataset](root, args.device, args.batch_size)
    model_id = meta.get("model")
    meta.update(
        dataset=args.dataset,
        created=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        model_revision=_hf_revision(model_id) if model_id else None, versions=_versions(),
        device=args.device, seconds=round(time.perf_counter() - t0, 1),
        shapes={k: list(v.shape) for k, v in arrays.items()},
    )
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(out)
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {out} {meta['shapes']} in {meta['seconds']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
