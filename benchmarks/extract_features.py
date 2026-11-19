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
- E4 (WP 7.1) natural classification sets, named ``<data>_<encoder>``, built by one generic
  (source, encoder) builder each; the 31 new ones are
  images ``{cifar10, cifar100, stl10, pets, flowers102}`` x ``{resnet50, dinov2, clip}`` (all but
  ``cifar100_dinov2``, ``stl10_clip`` above and ``pets_resnet50``, a local file of
  ``examples/04``): resnet50 = torchvision ResNet-50 IMAGENET1K_V2 penultimate layer (2,048-d,
  resize 256, centre crop 224, ImageNet normalisation); dinov2 and clip as above; CIFAR-100
  keeps ``y_extra_coarse``; Flowers102 trains on train+val; and
  text ``{banking77, emotion, massive, agnews, newsgroups}`` x ``{mpnet, minilm, bge, e5}``
  (all but ``banking77_mpnet``): sentence-transformers with the ``Normalize`` module removed and
  ``max_seq_length`` 256 (bge keeps its CLS pooling; e5 gets the ``"query: "`` prefix; mpnet is
  all-mpnet-base-v2, minilm all-MiniLM-L6-v2, bge bge-base-en-v1.5, e5 e5-base-v2). Every text set
  records ``n_test_in_train`` (test texts that occur verbatim in train; nothing is dropped) and
  the model and dataset revisions. ``E4_SETS`` in ``bench_datasets.py`` lists all 35.
- ``stsb_mpnet``: STS-B. ``X_train`` holds the unique sentences of the train split (for fitting a
  transform, unlabelled); ``X_test`` the unique test sentences, with ``pairs_a``, ``pairs_b``
  (indices into ``X_test``) and gold ``scores``; the validation split likewise as ``X_val``,
  ``val_pairs_a``, ``val_pairs_b``, ``val_scores`` (used for selection).
- ``glove100``, ``sift128``, ``gist960``: ANN-Benchmarks vectors (train / test as published),
  unlabelled.
- ``yelp_mpnet`` (E8, regression with heavy ties): Yelp review stars, a random 20k / 5k subset
  (seed 0) of ``Yelp/yelp_review_full``, all-mpnet-base-v2 (768-d); targets 1.0 to 5.0.
- ``fluorescence_esm2`` (E8, continuous regression): TAPE GFP fluorescence
  (``proteinea/fluorescence``, train 21,446 / test 27,217, as published: the test sequences have
  more mutations than the training ones), ESM-2 35M mean-pooled over residues (480-d); targets
  ``log_fluorescence``.
"""

from __future__ import annotations

import argparse
import contextlib
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
    "esm2": "facebook/esm2_t12_35M_UR50D",
    # E4 text encoders (sentence-transformers); dinov2 / clip above are shared with E4.
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "bge": "BAAI/bge-base-en-v1.5",
    "e5": "intfloat/e5-base-v2",
    # WP 8.3 extra text encoders (E4X), chosen for diversity before any statistic was seen.
    "gte": "thenlper/gte-base",
    "distilroberta": "sentence-transformers/all-distilroberta-v1",
    "glove": "sentence-transformers/average_word_embeddings_glove.6B.300d",
}
# Pinned revisions (commit hashes) of E4's text models and datasets. Offline nothing is known, so
# they are empty: an unpinned one is resolved to the current commit hash before loading, loaded
# at that hash, and the hash is written to the JSON. Fill in a hash to pin it for good.
MODEL_REVISIONS: dict[str, str] = {}
DATASET_REVISIONS: dict[str, str] = {}
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


def _hf_dataset_sha(repo: str) -> str | None:
    try:
        from huggingface_hub import HfApi

        return HfApi().dataset_info(repo).sha
    except Exception:
        return None


@contextlib.contextmanager
def _lock(path: Path):
    """Exclusive advisory lock, so array tasks sharing a raw download do not race (best effort)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        try:
            import fcntl

            fcntl.flock(f, fcntl.LOCK_EX)
        except Exception:
            pass
        yield


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


# --- E4 generic image sets: (source, encoder) ---------------------------------------------------


def _resnet50_encoder(device: str):
    """torchvision ResNet-50 IMAGENET1K_V2, fc removed: 2,048-d penultimate features.

    The transform is that of ``examples/04_image_classification.py`` (resize 256, centre crop 224,
    ImageNet normalisation), applied here to the PIL images, in fp32 (no autocast).
    """
    import torch
    import torch.nn as nn
    from torchvision import models, transforms

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    backbone.fc = nn.Identity()
    backbone = backbone.to(device).eval()

    def encode(images):
        batch = torch.stack([transform(im.convert("RGB")) for im in images]).to(device)
        return backbone(batch)

    return encode


def _torchvision_encoder(arch: str):
    """WP 8.3 extra image encoders (E4X): a torchvision ImageNet classifier without its last
    linear layer, with the weights' own preprocessing (``weights.transforms()``), in fp32.

    ``vit``: ``vit_b_16`` IMAGENET1K_V1, the class token after the final layer norm (768).
    ``convnext``: ``convnext_tiny`` IMAGENET1K_V1, the pooled, layer-normed features (768).
    """
    def make(device: str):
        import torch
        import torch.nn as nn
        from torchvision import models

        if arch == "vit":
            weights = models.ViT_B_16_Weights.IMAGENET1K_V1
            net = models.vit_b_16(weights=weights)
            net.heads = nn.Identity()
        else:
            weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
            net = models.convnext_tiny(weights=weights)
            net.classifier[2] = nn.Identity()
        transform = weights.transforms()
        net = net.to(device).eval()

        def encode(images):
            batch = torch.stack([transform(im.convert("RGB")) for im in images]).to(device)
            return net(batch)

        return encode

    return make


RESNET50_WEIGHTS_URL = "https://download.pytorch.org/models/resnet50-11ad3fa6.pth"
IMAGE_ENCODERS = {"resnet50": _resnet50_encoder, "dinov2": _dinov2_encoder, "clip": _clip_encoder}
E4X_IMAGE_ENCODERS = {"vit": _torchvision_encoder("vit"),
                      "convnext": _torchvision_encoder("convnext")}
IMAGE_ENCODERS.update(E4X_IMAGE_ENCODERS)
IMAGE_ENCODER_META = {
    "resnet50": dict(model="torchvision/resnet50", model_revision="IMAGENET1K_V2 " +
                     RESNET50_WEIGHTS_URL, embedding="penultimate layer (fc removed)",
                     preprocessing="resize 256, centre crop 224, ImageNet normalisation, fp32"),
    "dinov2": dict(model=MODELS["dinov2"], embedding="pooler_output (CLS)"),
    "clip": dict(model=MODELS["clip"], embedding="get_image_features (projected)"),
    "vit": dict(model="torchvision/vit_b_16", model_revision="IMAGENET1K_V1 "
                "https://download.pytorch.org/models/vit_b_16-c867db91.pth",
                embedding="class token after the final layer norm (heads removed)",
                preprocessing="ViT_B_16_Weights.IMAGENET1K_V1.transforms(), fp32"),
    "convnext": dict(model="torchvision/convnext_tiny", model_revision="IMAGENET1K_V1 "
                     "https://download.pytorch.org/models/convnext_tiny-983f1562.pth",
                     embedding="pooled, layer-normed features (last linear layer removed)",
                     preprocessing="ConvNeXt_Tiny_Weights.IMAGENET1K_V1.transforms(), fp32"),
}


def _cifar_source(name: str, coarse: bool):
    def load(root: Path):
        import torchvision.datasets as tvd

        cls = tvd.CIFAR100 if coarse else tvd.CIFAR10
        raw = root / "raw"
        with _lock(raw / f".{name}.lock"):
            train = cls(raw, train=True, download=True)
            test = cls(raw, train=False, download=True)
        extra, meta = {}, dict(images=f"{name.upper()} 32x32")
        if coarse:
            for split in ("train", "test"):
                with open(raw / "cifar-100-python" / split, "rb") as f:
                    extra[f"y_extra_coarse_{split}"] = np.asarray(
                        pickle.load(f, encoding="latin1")["coarse_labels"])
            meta["n_coarse_classes"] = 20
        return train, test, extra, meta

    return load


def _stl10_source(root: Path):
    from datasets import load_dataset

    ds = load_dataset(STL10_HF, cache_dir=str(root / "raw" / "hf_datasets"))
    names = ds["train"].features["label"].names
    if names != STL10_CLASSES:
        raise RuntimeError(f"unexpected STL-10 label names: {names}")
    meta = dict(images=f"STL-10 96x96 from the {STL10_HF} mirror",
                dataset_revision=_hf_dataset_sha(STL10_HF))
    return _HFImages(ds["train"]), _HFImages(ds["test"]), {}, meta


def _pets_source(root: Path):
    from torchvision.datasets import OxfordIIITPet

    raw = root / "raw"
    with _lock(raw / ".pets.lock"):
        train = OxfordIIITPet(raw, split="trainval", target_types="category", download=True)
        test = OxfordIIITPet(raw, split="test", target_types="category", download=True)
    return train, test, {}, dict(images="Oxford-IIIT Pets, trainval / test, category labels")


def _flowers_source(root: Path):
    from torch.utils.data import ConcatDataset
    from torchvision.datasets import Flowers102

    raw = root / "raw"
    with _lock(raw / ".flowers102.lock"):
        parts = {s: Flowers102(raw, split=s, download=True) for s in ("train", "val", "test")}
    return (ConcatDataset([parts["train"], parts["val"]]), parts["test"], {},
            dict(images="Flowers102, train + val as training, test as test"))


# source -> (loader(root) -> train, test, extra arrays, meta; classes; n_train; n_test)
IMAGE_SOURCES = {
    "cifar10": (_cifar_source("cifar10", False), 10, 50_000, 10_000),
    "cifar100": (_cifar_source("cifar100", True), 100, 50_000, 10_000),
    "stl10": (_stl10_source, 10, 5_000, 8_000),
    "pets": (_pets_source, 37, 3_680, 3_669),
    "flowers102": (_flowers_source, 102, 2_040, 6_149),
}


def _check_labels(y_train, y_test, n_classes: int, what: str):
    seen = np.unique(np.concatenate([y_train, y_test]))
    if not np.array_equal(seen, np.arange(n_classes)):
        raise RuntimeError(f"{what}: expected labels 0..{n_classes - 1}, got {seen[:5]}..."
                           f"{seen[-3:]} ({len(seen)} distinct)")


def _image_set(source: str, encoder: str):
    def build(root: Path, device: str, batch_size: int) -> tuple[dict, dict]:
        load, n_classes, n_train, n_test = IMAGE_SOURCES[source]
        train, test, extra, smeta = load(root)
        if (len(train), len(test)) != (n_train, n_test):
            raise RuntimeError(f"{source}: expected {n_train} / {n_test} images, got "
                               f"{len(train)} / {len(test)}")
        encode = IMAGE_ENCODERS[encoder](device)
        Xtr, ytr = _embed_images(train, encode, batch_size, device)
        Xte, yte = _embed_images(test, encode, batch_size, device)
        _check_labels(ytr, yte, n_classes, source)
        arrays = dict(X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte, **extra)
        meta = {**IMAGE_ENCODER_META[encoder], **smeta, "source": source,
                "n_classes": n_classes}
        return arrays, meta

    return build


# --- text ---------------------------------------------------------------------------------------


def _sentence_encoder(device: str):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODELS["mpnet"], device=device)
    # Raw (unnormalised) embeddings: normalisation is one of the transforms under test. Some
    # models (all-mpnet-base-v2 among them) end with a Normalize module, which
    # normalize_embeddings=False does not bypass, so drop it.
    for key in [k for k, m in model._modules.items() if type(m).__name__ == "Normalize"]:
        del model._modules[key]

    def encode(texts, batch_size):
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
                        normalized=False, pooling="mean, Normalize module removed")


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
                        normalized=False, pooling="mean, Normalize module removed",
                        n_test_pairs=len(ds["test"]),
                        n_val_pairs=len(ds["validation"]))


def yelp_mpnet(root: Path, device: str, batch_size: int) -> tuple[dict, dict]:
    from datasets import load_dataset

    ds = load_dataset("Yelp/yelp_review_full", cache_dir=str(root / "raw" / "hf_datasets"))
    rng = np.random.default_rng(0)
    encode = _sentence_encoder(device)
    arrays = {}
    for split, n, key in (("train", 20_000, "train"), ("test", 5_000, "test")):
        part = ds[split]
        idx = np.sort(rng.choice(len(part), n, replace=False))
        rows = part.select(idx.tolist())
        arrays[f"X_{key}"] = encode(rows["text"], batch_size)
        arrays[f"y_{key}"] = np.asarray(rows["label"], dtype=np.float64) + 1.0  # 1..5 stars
        arrays[f"idx_{key}"] = idx
    return arrays, dict(model=MODELS["mpnet"], hf_dataset="Yelp/yelp_review_full",
                        target="stars 1-5", subset="random 20000 train / 5000 test, seed 0",
                        normalized=False, pooling="mean, Normalize module removed",
                        max_seq_length=384)


def fluorescence_esm2(root: Path, device: str, batch_size: int) -> tuple[dict, dict]:
    import torch
    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer

    ds = load_dataset("proteinea/fluorescence", cache_dir=str(root / "raw" / "hf_datasets"))
    tok = AutoTokenizer.from_pretrained(MODELS["esm2"])
    model = AutoModel.from_pretrained(MODELS["esm2"]).to(device).eval()

    @torch.no_grad()
    def encode(seqs):
        out = []
        for start in range(0, len(seqs), batch_size):
            batch = tok(list(seqs[start: start + batch_size]), return_tensors="pt", padding=True)
            batch = {k: v.to(device) for k, v in batch.items()}
            h = model(**batch).last_hidden_state
            # Mean over residues: drop <cls>, <eos> and padding.
            mask = batch["attention_mask"].clone()
            mask[:, 0] = 0
            mask[torch.arange(len(mask)), batch["attention_mask"].sum(1) - 1] = 0
            m = mask.unsqueeze(-1).to(h.dtype)
            out.append(((h * m).sum(1) / m.sum(1)).float().cpu().numpy())
        return np.concatenate(out)

    arrays = {}
    for split, key in (("train", "train"), ("test", "test")):
        part = ds[split]
        arrays[f"X_{key}"] = encode(part["primary"])
        arrays[f"y_{key}"] = np.asarray(part["log_fluorescence"], dtype=np.float64)
    return arrays, dict(model=MODELS["esm2"], hf_dataset="proteinea/fluorescence",
                        target="log_fluorescence", pooling="mean over residues (no cls/eos)",
                        splits="TAPE train / test as published")


# --- E4 generic text sets: (source, encoder) ----------------------------------------------------


def _load_hf(repo: str, root: Path, config: str | None = None):
    """``load_dataset`` at a pinned (or just resolved) revision. Returns (dataset dict, revision).

    Parquet repos only (no trust_remote_code). A repo that still ships a loading script is read
    from its ``refs/convert/parquet`` branch instead, and the JSON says so.
    """
    from datasets import load_dataset

    pinned = DATASET_REVISIONS.get(repo)
    rev = pinned or _hf_dataset_sha(repo)
    kw = dict(cache_dir=str(root / "raw" / "hf_datasets"))
    try:
        return load_dataset(repo, config, revision=rev, **kw), rev
    except Exception as e:
        if pinned:
            raise
        print(f"{repo}: load at {rev} failed ({type(e).__name__}: {e}); "
              "trying refs/convert/parquet")
        return load_dataset(repo, config, revision="refs/convert/parquet", **kw), \
            "refs/convert/parquet"


def _column(split, name: str, repo: str):
    if name not in split.column_names:
        raise RuntimeError(f"{repo}: column {name!r} missing; the split has "
                           f"{split.column_names}. Check the dataset card and fix TEXT_SOURCES.")
    return split[name]


def _dense_labels(train_labels, test_labels, what: str):
    """int64 labels 0..C-1 (sorted distinct values of train + test); ints or strings accepted."""
    both = np.asarray(list(train_labels) + list(test_labels))
    if both.dtype.kind not in "iuUSO":
        raise RuntimeError(f"{what}: labels of dtype {both.dtype} are not class ids")
    classes, inv = np.unique(both, return_inverse=True)
    inv = inv.astype(np.int64)
    return inv[: len(train_labels)], inv[len(train_labels):], [str(c) for c in classes]


def _cap(rng, split, n: int):
    idx = np.sort(rng.choice(len(split), n, replace=False))
    return split.select(idx.tolist()), idx


# name -> dict(repo, config, train / test split names, text / label column, number of classes,
# cap = (n_train, n_test) of a seeded random subset). Column names are from the dataset cards and
# checked at load time. emotion: config "split" (the "unsplit" one has no test set).
TEXT_SOURCES = {
    "banking77": dict(repo="mteb/banking77", config=None, train="train", test="test",
                      text="text", label="label", n_classes=77),
    "emotion": dict(repo="dair-ai/emotion", config="split", train="train", test="test",
                    text="text", label="label", n_classes=6),
    "massive": dict(repo="mteb/amazon_massive_intent", config="en", train="train", test="test",
                    text="text", label="label", n_classes=60),
    "agnews": dict(repo="fancyzhx/ag_news", config=None, train="train", test="test",
                   text="text", label="label", n_classes=4, cap=(20_000, 5_000)),
    "newsgroups": dict(repo="SetFit/20_newsgroups", config=None, train="train", test="test",
                       text="text", label="label", n_classes=20),
}
E4_TEXT_PREFIX = {"mpnet": "", "minilm": "", "bge": "", "e5": "query: "}
E4X_TEXT_PREFIX = {"gte": "", "distilroberta": "", "glove": ""}  # WP 8.3 extra encoders
TEXT_PREFIX = {**E4_TEXT_PREFIX, **E4X_TEXT_PREFIX}


def _st_encoder(key: str, device: str, prefix: str = ""):
    """sentence-transformers encoder without its ``Normalize`` module, max_seq_length 256.

    Returns (encode(texts, batch_size), meta). The model keeps its own pooling (bge: CLS).
    """
    from sentence_transformers import SentenceTransformer

    model_id = MODELS[key]
    rev = MODEL_REVISIONS.get(model_id) or _hf_revision(model_id)
    model = SentenceTransformer(model_id, device=device, revision=rev)
    removed = [k for k, m in model._modules.items() if type(m).__name__ == "Normalize"]
    for k in removed:
        del model._modules[k]
    model.max_seq_length = 256
    try:
        pooling = next(m for m in model._modules.values()
                       if type(m).__name__ == "Pooling").get_pooling_mode_str()
    except Exception:
        pooling = "unknown"

    def encode(texts, batch_size):
        texts = [prefix + t for t in texts]
        return model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                            normalize_embeddings=False, show_progress_bar=False).astype(np.float32)

    meta = dict(model=model_id, model_revision=rev, normalized=False, max_seq_length=256,
                pooling=f"{pooling}, Normalize module removed" if removed else pooling,
                prefix=prefix)
    return encode, meta


def _text_set(source: str, encoder: str):
    def build(root: Path, device: str, batch_size: int) -> tuple[dict, dict]:
        spec = TEXT_SOURCES[source]
        ds, rev = _load_hf(spec["repo"], root, spec["config"])
        parts = {}
        rng = np.random.default_rng(0)
        extra = {}
        for key in ("train", "test"):
            split = ds[spec[key]]
            texts_col = _column(split, spec["text"], spec["repo"])
            _column(split, spec["label"], spec["repo"])
            if "cap" in spec:
                n = spec["cap"][0 if key == "train" else 1]
                split, idx = _cap(rng, split, n)
                extra[f"idx_{key}"] = idx
                texts_col = split[spec["text"]]
            parts[key] = (list(texts_col), split[spec["label"]])
        (ttr, ltr), (tte, lte) = parts["train"], parts["test"]
        ytr, yte, classes = _dense_labels(ltr, lte, source)
        if len(classes) != spec["n_classes"]:
            raise RuntimeError(f"{source}: expected {spec['n_classes']} classes, found "
                               f"{len(classes)}")
        train_set = set(ttr)
        n_test_in_train = int(sum(t in train_set for t in tte))
        encode, emeta = _st_encoder(encoder, device, TEXT_PREFIX[encoder])
        arrays = dict(X_train=encode(ttr, batch_size), y_train=ytr,
                      X_test=encode(tte, batch_size), y_test=yte, **extra)
        meta = dict(emeta, source=source, hf_dataset=spec["repo"], hf_config=spec["config"],
                    dataset_revision=rev, n_classes=len(classes), class_values=classes,
                    n_train=len(ttr), n_test=len(tte), n_test_in_train=n_test_in_train)
        if "cap" in spec:
            meta["subset"] = "random {} train / {} test, seed 0".format(*spec["cap"])
        return arrays, meta

    return build


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
    "yelp_mpnet": yelp_mpnet,
    "fluorescence_esm2": fluorescence_esm2,
    **{name: _ann(name) for name in ANN},
}
# E4: every (source, encoder) pair not already built above (and not the local pets_resnet50).
for _src in IMAGE_SOURCES:
    for _enc in IMAGE_ENCODERS:
        BUILDERS.setdefault(f"{_src}_{_enc}", _image_set(_src, _enc))
for _src in TEXT_SOURCES:
    for _enc in TEXT_PREFIX:
        BUILDERS.setdefault(f"{_src}_{_enc}", _text_set(_src, _enc))
BUILDERS.pop("pets_resnet50", None)  # built by examples/04, not here


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
    meta.setdefault("model_revision", _hf_revision(model_id) if model_id else None)
    meta.update(
        dataset=args.dataset,
        created=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        versions=_versions(),
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
