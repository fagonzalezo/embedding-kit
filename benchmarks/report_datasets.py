"""Results of the default selection per dataset: the E10 runs as user documentation

    python benchmarks/report_datasets.py --collect     # per-run files -> summary_datasets.csv
    python benchmarks/report_datasets.py               # summary_datasets.csv -> the document

``--collect`` reads ``<results>/e10/<set>/<method>.json`` (``run.py --experiment e10``; the
per-run files are not committed) and writes ``<results>/e10/summary_datasets.csv``: one row per
data set with its description, the raw and selected scores, paired bootstrap intervals (the same
computation as ``report_e10.py``), the best candidate in hindsight and the selection's own
validation table. Without ``--collect`` the document is written from that CSV alone, so it can be
regenerated from the repository: ``docs/benchmarks/RESULTS_BY_DATASET.md`` by default.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bench_datasets as D  # noqa: E402
from report_e10 import CANDIDATES, compare, load  # noqa: E402

ROOT = HERE.parent
DOC = ROOT / "docs" / "benchmarks" / "RESULTS_BY_DATASET.md"

# Source data sets: modality, display name, what is predicted, link-free description.
SOURCES = {
    "agnews": ("text", "AG News", "news topic",
               "news articles, title and description; a random subset of the full set"),
    "banking77": ("text", "Banking77", "customer intent", "online-banking customer queries"),
    "emotion": ("text", "Emotion", "emotion", "English tweets"),
    "massive": ("text", "MASSIVE (en-US)", "assistant intent", "voice-assistant utterances"),
    "newsgroups": ("text", "20 Newsgroups", "newsgroup", "Usenet posts"),
    "cifar10": ("image", "CIFAR-10", "object class", "32 x 32 natural images"),
    "cifar100": ("image", "CIFAR-100", "object class", "32 x 32 natural images"),
    "flowers102": ("image", "Flowers-102", "flower species", "flower photographs"),
    "pets": ("image", "Oxford-IIIT Pets", "cat or dog breed", "pet photographs"),
    "stl10": ("image", "STL-10", "object class", "96 x 96 natural images"),
    "synthetic_regression": ("regression", "Synthetic regression", "continuous target",
                             "8-d latent through a fixed random tanh network, plus nuisance "
                             "directions"),
    "synthetic_regression_ties": ("regression", "Synthetic regression (5 levels)",
                                  "the same target on 5 levels (heavy ties)", "as above"),
    "yelp": ("regression", "Yelp reviews", "review stars (1 to 5) as a number",
             "restaurant reviews"),
    "fluorescence": ("regression", "TAPE fluorescence", "log-fluorescence of GFP variants",
                     "protein sequences; test variants carry more mutations (extrapolation)"),
}

# Encoders: display name and model. The dimension is read from the runs.
ENCODERS = {
    "minilm": ("MiniLM", "sentence-transformers/all-MiniLM-L6-v2"),
    "mpnet": ("MPNet", "sentence-transformers/all-mpnet-base-v2"),
    "bge": ("BGE", "BAAI/bge-base-en-v1.5"),
    "e5": ("E5", "intfloat/e5-base-v2, prefix \"query: \""),
    "gte": ("GTE", "thenlper/gte-base"),
    "distilroberta": ("DistilRoBERTa", "sentence-transformers/all-distilroberta-v1"),
    "glove": ("GloVe (averaged)", "average_word_embeddings_glove.6B.300d"),
    "resnet50": ("ResNet-50", "torchvision resnet50 (ImageNet), pooled features"),
    "dinov2": ("DINOv2", "facebook/dinov2-base, CLS token"),
    "clip": ("CLIP", "openai/clip-vit-base-patch32, image features"),
    "vit": ("ViT-B/16", "torchvision vit_b_16 (ImageNet), class token"),
    "convnext": ("ConvNeXt-T", "torchvision convnext_tiny (ImageNet), pooled features"),
    "esm2": ("ESM-2 35M", "facebook/esm2_t12_35M_UR50D, mean over residues"),
    "synthetic": ("none (synthetic)", "generated features"),
}
# What the inputs are (the tables' "Data" column); the section a source belongs to is
# ``SOURCES[source][0]``.
DATA = {"yelp": "text", "fluorescence": "protein", "synthetic_regression": "synthetic",
        "synthetic_regression_ties": "synthetic", "esm2": "protein", "synthetic": "synthetic",
        "mpnet": "text"}


def split_name(name: str) -> tuple[str, str]:
    """``<source>_<encoder>`` -> (source key, encoder key)."""
    if name.startswith("synthetic_regression"):
        return name, "synthetic"
    src, enc = name.rsplit("_", 1)
    return src, enc


def candidate_dim(name: str, d: int) -> int:
    """Output dimension of a candidate of the selection (``refiner-128``, ``pca_l2-64``, ...)."""
    if name in ("identity", "center_l2"):
        return d
    if name == "refiner":
        return min(128, d // 2)
    return int(name.rsplit("-", 1)[1])


def collect(e10_dir: Path) -> pd.DataFrame:
    data = load(e10_dir)
    rows = []
    for name, runs in sorted(data.items()):
        if "metrics" not in runs.get("select", {}) or "metrics" not in runs.get("identity", {}):
            continue
        src, enc = split_name(name)
        meta = json.loads((e10_dir / name / "select.json").read_text())["dataset_meta"]
        sel, raw = runs["select"], runs["identity"]
        regression = "knn_r2" in sel["metrics"]
        prim = "knn_r2" if regression else "knn_accuracy"
        params = sel["params"]
        n_test = len(sel["metrics"][prim][1])
        d = json.loads((e10_dir / name / "identity.json").read_text())["stages"]["test"][0]["dim"]
        row = dict(
            dataset=name, modality=SOURCES[src][0], source=src, encoder=enc,
            extra_set=name in getattr(D, "E4X_SETS", ()), task=meta.get("task"),
            n_classes=meta.get("n_classes"), n_train=params.get("n_selection_rows"),
            n_test=n_test, d=d, metric=prim, selected=params.get("selected"),
            selected_dim=json.loads((e10_dir / name / "select.json").read_text())
            ["stages"]["test"][0]["dim"],
            fit_time=sel.get("fit_time"))
        for key, metric in (("", prim), ("euc_", "knn_accuracy_euclidean"),
                            ("probe_", "linear_probe_c1")):
            if metric not in sel["metrics"]:
                continue
            diff, lo, hi = compare(runs, "select", "identity", metric)
            row.update({f"{key}raw": raw["metrics"][metric][0],
                        f"{key}after": sel["metrics"][metric][0],
                        f"{key}diff": diff, f"{key}lo": lo, f"{key}hi": hi})
        cands = {c: runs[c]["metrics"][prim][0] for c in CANDIDATES
                 if "metrics" in runs.get(c, {}) and prim in runs[c]["metrics"]}
        best = max(cands, key=cands.get)
        row.update(best=best, best_dim=candidate_dim(best, d), best_value=cands[best],
                   regret=cands[best] - row["after"])
        report = json.loads(params.get("selection_report", "[]"))
        challengers = [r for r in report if r["candidate"] != "identity"]
        if challengers:
            top = max(challengers, key=lambda r: r["diff_vs_identity"])
            row.update(val_top=top["candidate"], val_top_diff=top["diff_vs_identity"],
                       val_top_lo=top["diff_lo"], val_top_hi=top["diff_hi"])
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", default=str(HERE / "results"))
    p.add_argument("--collect", action="store_true",
                   help="read the per-run files and write summary_datasets.csv")
    p.add_argument("--out", default=str(DOC))
    args = p.parse_args(argv)
    e10_dir = Path(args.results) / "e10"
    csv = e10_dir / "summary_datasets.csv"
    if args.collect:
        df = collect(e10_dir)
        if df.empty:
            sys.exit(f"No E10 results under {e10_dir}")
        df.to_csv(csv, index=False)
        print(f"wrote {csv} ({len(df)} sets)")
    df = pd.read_csv(csv)
    Path(args.out).write_text(render(df))
    print(f"wrote {args.out}")


MODALITIES = (("text", "Text classification"), ("image", "Image classification"),
              ("regression", "Regression"))
METHOD = {"identity": "raw (kept)", "center_l2": "centre + L2", "refiner": "refiner",
          "refiner-128": "refiner", "pca-64": "PCA", "pca-128": "PCA", "pca-256": "PCA",
          "pca_l2-64": "PCA", "pca_l2-128": "PCA", "pca_l2-256": "PCA"}


def pts(x: float, regression: bool = False) -> str:
    """A score as a percentage (classification) or an R^2 (regression)."""
    return f"{x:.3f}" if regression else f"{100 * x:.1f}"


def delta(x: float, regression: bool = False) -> str:
    return f"{x:+.3f}" if regression else f"{100 * x:+.1f}"


def method(name: str, dim: int, d: int, kept: bool = True) -> str:
    label = METHOD.get(name, name) if kept else METHOD.get(name, name).replace(" (kept)", "")
    return label if name == "identity" or dim == d else f"{label} → {dim}"


def change(r, key: str = "", regression: bool = False) -> str:
    """Δ with its interval; bold when the interval excludes 0; a dash when nothing changed."""
    diff, lo, hi = r[f"{key}diff"], r[f"{key}lo"], r[f"{key}hi"]
    if diff == 0 and lo == 0 and hi == 0:
        return "—"
    text = f"{delta(diff, regression)} [{delta(lo, regression)}, {delta(hi, regression)}]"
    return f"**{text}**" if lo > 0 else text


def before_after(r, key: str) -> str:
    if r.selected == "identity":
        return pts(r[f"{key}raw"])
    mark = " ▲" if r[f"{key}lo"] > 0 else (" ▼" if r[f"{key}hi"] < 0 else "")
    return f"{pts(r[f'{key}raw'])} → {pts(r[f'{key}after'])}{mark}"


def hindsight(r, regression: bool = False) -> str:
    if r.regret <= 0:
        return "same"
    return (f"{method(r.best, r.best_dim, r.d, kept=False)}: {pts(r.best_value, regression)} "
            f"({delta(r.regret, regression)})")


def order(df: pd.DataFrame) -> pd.DataFrame:
    src = {s: i for i, s in enumerate(SOURCES)}
    enc = {e: i for i, e in enumerate(ENCODERS)}
    return df.assign(_s=df.source.map(src), _e=df.encoder.map(enc)).sort_values(["_s", "_e"])


def sources_table(df: pd.DataFrame) -> list[str]:
    out = ["| Data | Source | Predicts | Classes | Train / test rows | Content |",
           "|---|---|---|---|---|---|"]
    for src, (modality, name, target, content) in SOURCES.items():
        part = df[df.source == src]
        if part.empty:
            continue
        r = part.iloc[0]
        classes = "—" if modality == "regression" else f"{int(r.n_classes)}"
        out.append(f"| {DATA.get(src, modality)} | {name} | {target} | {classes} | "
                   f"{int(r.n_train):,} / {int(r.n_test):,} | {content} |")
    return out


def encoders_table(df: pd.DataFrame) -> list[str]:
    out = ["| Data | Encoder | Model | d | Sources |", "|---|---|---|---|---|"]
    for enc, (name, model) in ENCODERS.items():
        part = df[df.encoder == enc]
        if part.empty:
            continue
        modality = DATA.get(enc, part.modality.iloc[0])
        dims = sorted(set(part.d))
        srcs = ", ".join(SOURCES[s][1] for s in SOURCES if s in set(part.source))
        out.append(f"| {modality} | {name} | {model} | {', '.join(str(x) for x in dims)} | "
                   f"{srcs} |")
    return out


def results_table(part: pd.DataFrame, regression: bool) -> list[str]:
    if regression:
        out = ["| Set | d | Raw kNN R² | Picked | After (kNN R²) | Δ R² [95% CI] "
               "| Best in hindsight (kNN R²) | Fit time |", "|---|---|---|---|---|---|---|---|"]
    else:
        out = ["| Set | d | Raw kNN accuracy (%) | Picked | After (%) | Δ points [95% CI] | "
               "Euclidean kNN accuracy (%) | Linear probe accuracy (%) | "
               "Best in hindsight (kNN accuracy, %) | Fit time |",
               "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in order(part).iterrows():
        cells = [f"`{r.dataset}`", f"{int(r.d)}", pts(r.raw, regression),
                 method(r.selected, int(r.selected_dim), int(r.d)), pts(r.after, regression),
                 change(r, "", regression)]
        if not regression:
            cells += [before_after(r, "euc_"), before_after(r, "probe_")]
        cells += [hindsight(r, regression), f"{r.fit_time:.0f} s"]
        out.append("| " + " | ".join(cells) + " |")
    return out


def spearman(a, b) -> float:
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def render(df: pd.DataFrame) -> str:
    cls = df[df.modality != "regression"]
    reg = df[df.modality == "regression"]
    better = df[df.lo > 0]
    cls_better = cls[cls.lo > 0]
    kept = df[df.selected == "identity"]
    regret = df[df.regret > 0]
    share = (cls_better["diff"] / (1 - cls_better.raw))
    rho_all = spearman(cls.raw, cls["diff"])
    rho_img = spearman(cls[cls.modality == "image"].raw, cls[cls.modality == "image"]["diff"])
    rho_txt = spearman(cls[cls.modality == "text"].raw, cls[cls.modality == "text"]["diff"])
    within = [spearman(g.raw, g["diff"]) for _, g in cls.groupby("source")
              if g["diff"].nunique() > 1]
    by_enc = cls.groupby("encoder")["diff"].median().sort_values()
    by_src = cls.groupby("source").agg(raw=("raw", "median"), gain=("diff", "median"),
                                       better=("lo", lambda x: int((x > 0).sum())),
                                       n=("dataset", "size"),
                                       hindsight=("best_value", "max"))
    probe_worse = cls[cls.probe_hi < 0]
    regret_kept = regret[regret.selected == "identity"]
    regret_changed = regret[regret.selected != "identity"]
    probe_floor = probe_worse.probe_raw.min() if len(probe_worse) else np.nan
    dino = cls[cls.encoder == "dinov2"]
    dino_high = int((dino.raw >= 0.98).sum())
    bge_kept = int(((cls.encoder == "bge") & (cls.selected == "identity")).sum())

    def span(source: str) -> str:
        part = cls[cls.source == source].raw
        return f"{100 * part.min():.1f} to {100 * part.max():.1f}%"

    def gap(source: str) -> float:
        """Largest test gain of any candidate over raw on a source's kept sets."""
        part = cls[(cls.source == source) & (cls.selected == "identity")]
        return float((part.best_value - part.raw).max()) if len(part) else 0.0
    counts = df.selected.map(lambda s: METHOD.get(s, s)).value_counts()
    n = len(df)
    most = ", ".join(ENCODERS[e][0] for e in by_enc.index[::-1][:3])
    least = ", ".join(ENCODERS[e][0] for e in by_enc.index[:2])

    def src_line(s: str) -> str:
        r = by_src.loc[s]
        return (f"{SOURCES[s][1]} (raw median {100 * r.raw:.1f}%, better on {int(r.better)} "
                f"of {int(r.n)}, median gain {100 * r.gain:.1f} points)")

    out = [
        "# Results by dataset", "",
        "What `EmbedKit().fit(X, y)` does to each of the "
        f"{n} data sets of benchmark E10, compared with the embedding as the encoder produced "
        "it. Each data set is one source (for example CIFAR-100) embedded by one pretrained "
        f"encoder (for example DINOv2): {len(cls[cls.modality == 'text'])} text and "
        f"{len(cls[cls.modality == 'image'])} image classification sets, and {len(reg)} "
        "regression sets. The summary of the same runs, with the cost at scale and the "
        "retest of the hubness recommendation, is `docs/benchmarks/BENCHMARK_SELECTION.md`.", "",
        "Generated by `benchmarks/report_datasets.py` from "
        "`benchmarks/results/e10/summary_datasets.csv`; do not edit by hand.", "",
        "## In short", "",
        f"- **Never worse, often better.** The default is significantly better than the raw "
        f"embedding on {len(better)} of {n} sets and significantly worse on "
        f"{int((df.hi < 0).sum())}. Where it is better on a classification set it adds "
        f"{100 * cls_better['diff'].min():.1f} to {100 * cls_better['diff'].max():.1f} points of "
        f"kNN accuracy (median {100 * cls_better['diff'].median():.1f}), which removes a median "
        f"{100 * share.median():.0f}% of the remaining errors.",
        f"- **What it picks.** The refiner on {counts.get('refiner', 0)} sets, the raw "
        f"embedding on {counts.get('raw (kept)', 0)} and PCA on {counts.get('PCA', 0)}. "
        "Centring alone was never picked.",
        f"- **Weaker embeddings gain more.** The lower the raw accuracy, the larger the gain "
        f"(Spearman {rho_all:.2f} over the {len(cls)} classification sets; {rho_img:.2f} on "
        f"images). The encoders that gain most are {most}; those that gain least are "
        f"{least}.",
        f"- **Caution costs little.** On the {len(regret)} sets where another candidate scored "
        f"higher on test than the one picked, the difference is at most "
        f"{100 * regret.regret.max():.1f} points.",
        f"- **kNN is what is optimised.** A linear probe on the output is better on "
        f"{int((cls.probe_lo > 0).sum())} sets but worse on {len(probe_worse)}, by up to "
        f"{100 * -probe_worse.probe_diff.min():.1f} points.", "",
        "## Protocol", "",
        "**Splits.** Every set has a fixed training split and a fixed test split (the sizes are "
        "in the source table below). The test split is used once, to score the final result; "
        "nothing is chosen on it.", "",
        "**What the default does** (`EmbedKit()`, `strategy=\"select\"`, with labels):",
        "1. It holds out a random 20% of the training rows, stratified by class "
        "(`val_fraction=0.2`).",
        "2. It fits each candidate on the other 80%:",
        "   - **raw**: the embedding unchanged (`Identity`);",
        "   - **centre + L2**: subtract the training mean, then scale each row to unit length;",
        "   - **PCA → 64, 128, 256**: the top principal directions, then L2 normalisation "
        "(only the sizes below d);",
        "   - **refiner**: a small network trained on the labels (described below).",
        "3. It scores each candidate on the held-out 20% by cosine 10-NN classification "
        "accuracy (10-NN R² for a continuous target).",
        "4. It keeps **raw** unless a candidate beats it with a paired bootstrap interval (95%, "
        "2,000 resamples) entirely above 0, and then takes the best such candidate.",
        "5. It refits the winner on all the training rows.", "",
        "**The refiner.** There are two refiners, one for classification and one for "
        "regression, each a fixed configuration: the library's defaults, the same on every "
        "set, with nothing tuned per set. Only the weights are learned on each set. The "
        "output dimension is min(128, d / 2), which is 128 on all the sets here (the "
        f"smallest d is {int(df.d.min())}).", "",
        "| | Classification (`mode=\"supervised\"`) | Regression (`mode=\"regression\"`) |",
        "|---|---|---|",
        "| Architecture | residual adapter: z = normalise(W x̃ + g(x̃)), with x̃ the input "
        "standardised, W initialised at the top principal directions and g an MLP (hidden "
        "256, 2 layers, dropout 0.1) whose output starts at 0, so training starts at PCA + "
        "L2 | MLP (hidden 256, 2 layers, dropout 0.1), L2-normalised output |",
        "| Loss | supervised contrastive (SupCon), temperature 0.07 | Rank-N-Contrast, "
        "temperature 0.07 |",
        "| Views per row | the row and one of its 5 nearest neighbours in the input "
        "(`KNNPairs(5)`) | two copies with Gaussian noise, 0.05 of each feature's standard "
        "deviation (`GaussianNoise(0.05)`) |",
        "| Early stopping | on 10-NN accuracy of 10% of the fitting rows held out | on the "
        "R² of a ridge probe on 10% held out |",
        "| Training | Adam, learning rate 3e-4, weight decay 1e-4, cosine schedule, batch 256, "
        "at most 100 epochs; evaluated every 5 epochs, stopped after 10 evaluations without "
        "improvement, best weights restored | the same |", "",
        "**What is measured**, on the test split:",
        "- **kNN accuracy** (primary, classification): the percentage of test rows whose 10 "
        "nearest training rows by cosine distance vote for the correct class.",
        "- **kNN R²** (primary, regression): R² of predicting each test row's target as the "
        "mean target of its 10 nearest training rows by cosine distance. It can be negative "
        "(worse than predicting the mean).",
        "- **Euclidean kNN accuracy**: the same vote with Euclidean distance.",
        "- **Linear probe accuracy**: the test accuracy of a logistic regression (C = 1) "
        "trained on the transformed training rows. Not computed for regression.",
        "- **Δ [95% CI]**: after minus raw, in accuracy points (R² for regression), with a paired "
        "bootstrap interval over test rows (2,000 resamples). **Bold**: the interval excludes "
        "0 (significantly better). In the Euclidean and probe columns, ▲ and ▼ mark a "
        "significant rise or fall; a single number means the raw embedding was kept.", "",
        "**Picked and best in hindsight.** *Picked* is what the default chose, using only its "
        "held-out training rows; it is what a user gets. *Best in hindsight* is the candidate "
        "with the highest test score, each fitted on all the training rows. It cannot be used "
        "in practice (choosing it needs the test labels) and its score is slightly optimistic, "
        "but the gap to it shows what the default leaves on the table. \"same\" means the "
        "picked candidate was also the best on test.", "",
        "**Runs.** EmbedKit 0.7.0, one seed per set, one Quadro RTX 8000 GPU per run on the "
        "OCCIDATA cluster (2026-11-18 and 2026-11-19). *Fit time* is the whole "
        "`fit`: the selection and the refit. Text was embedded with sentence-transformers "
        "(any final normalisation removed, sequences cut at 256 tokens); images with each "
        "model's own preprocessing.", "",
        "## Data sets", "",
        "Each classification source is embedded by every encoder of its modality, so a set "
        "name is `<source>_<encoder>`.", "",
        *sources_table(df), "",
        *encoders_table(df), "",
    ]
    for key, title in MODALITIES:
        part = df[df.modality == key]
        if part.empty:
            continue
        regression = key == "regression"
        out += [f"## {title}", ""]
        sub = part[part.lo > 0]
        out.append(f"Better on {len(sub)} of {len(part)} sets, worse on "
                   f"{int((part.hi < 0).sum())}."
                   + ("" if regression else
                      f" Median gain {100 * part['diff'].median():.1f} points of kNN accuracy; "
                      "largest "
                      f"{100 * part['diff'].max():.1f} "
                      f"(`{part.loc[part['diff'].idxmax(), 'dataset']}`)."))
        out += ["", *results_table(part, regression), ""]

    newsgroups = by_src.loc["newsgroups"] if "newsgroups" in by_src.index else None
    emotion = by_src.loc["emotion"] if "emotion" in by_src.index else None
    out += [
        "## Discussion", "",
        "### Where the default helps", "",
        f"The gain depends on how much room the raw embedding leaves. Over the {len(cls)} "
        f"classification sets the Spearman correlation between raw kNN accuracy and the gain "
        f"is {rho_all:.2f} ({rho_img:.2f} on images, {rho_txt:.2f} on text), and within a "
        f"source the encoder with the weaker raw score gains more in "
        f"{sum(w < 0 for w in within)} of the {len(within)} sources whose gains differ. The "
        "largest gains are on tasks with many classes (Flowers-102, CIFAR-100, MASSIVE) and "
        "on Emotion, which has 6 classes but a low raw accuracy: "
        + "; ".join(src_line(s) for s in ("flowers102", "cifar100", "massive", "emotion")
                    if s in by_src.index)
        + ". Small training sets are not an obstacle: Flowers-102 has 2,040 training rows for "
        "102 classes (20 per class, 4 per class in the held-out part), and its gains are "
        "among the largest.", "",
        "Among encoders, "
        + ", ".join(f"{ENCODERS[e][0]} (median {100 * by_enc[e]:+.1f})"
                    for e in by_enc.index[::-1][:3])
        + " gain most. Averaged GloVe vectors and ImageNet ResNet-50 features are older and "
        "weaker, and CLIP's image features were trained to match images with captions, not "
        "to separate fine-grained categories; this is a plausible reading, not something "
        "these runs test. "
        + " and ".join(f"{ENCODERS[e][0]} (median {100 * by_enc[e]:+.1f})"
                       for e in by_enc.index[:2])
        + f" gain least. DINOv2 is at 98% or more on {dino_high} of its five sources; BGE "
        "is among the strongest text encoders here, and the default kept it raw on "
        f"{bge_kept} of 5.", "",
        "### Where the raw embedding is kept", "",
        f"The default kept the raw embedding on {len(kept)} sets. On most of them no "
        f"candidate does much better even in hindsight: AG News (raw {span('agnews')} with "
        f"every encoder; at most {100 * gap('agnews'):.1f} points from any candidate on "
        f"test) and STL-10 (raw {span('stl10')}; at most {100 * gap('stl10'):.1f}), and "
        "DINOv2 on Flowers-102 (99.6%).", "",
    ]
    if newsgroups is not None and emotion is not None:
        out += [
            f"20 Newsgroups is the exception: its raw accuracy is low (median "
            f"{100 * newsgroups.raw:.1f}%), yet the default improves it on only "
            f"{int(newsgroups.better)} of {int(newsgroups.n)} encoders (GloVe, the weakest), "
            f"and no candidate gains more than {100 * gap('newsgroups'):.1f} points on the "
            "others even in hindsight. Emotion starts as low (median "
            f"{100 * emotion.raw:.1f}%) and improves on {int(emotion.better)} of "
            f"{int(emotion.n)}, by "
            f"{100 * emotion.gain:.1f} points at the median. One reading: sentence encoders "
            "already arrange text by topic, which is what the newsgroup labels are, so "
            "reshaping the space cannot add much; the newsgroup errors may come from what "
            "the embedding does not contain (long posts cut at 256 tokens, overlapping "
            "groups). Emotion is not what these encoders organise by, and supervision can "
            "pull it forward. These runs do not test that explanation.", "",
        ]
    out += [
        "### What the caution costs", "",
        f"On {len(regret)} sets the candidate picked was not the best on test. On "
        f"{len(regret_kept)} of them the raw embedding was kept because no candidate's "
        "interval on the held-out rows lay above 0: with a held-out part of a few hundred to "
        "a few thousand rows, a gain under a point or so cannot be told from noise, and the "
        "rule then keeps the input. The largest such gap is "
        f"{100 * regret.regret.max():.1f} points "
        f"(`{regret.loc[regret.regret.idxmax(), 'dataset']}`)."
        + "".join(f" On `{r.dataset}`, {method(r.selected, int(r.selected_dim), int(r.d))} "
                  f"passed on the held-out rows and was {100 * r.regret:.1f} points below raw on "
                  "test, inside the noise." for r in regret_changed.itertuples())
        + " The other side of that caution is that no set got worse.", "",
        "### kNN against the linear probe", "",
        f"The selection scores candidates by kNN, and Euclidean kNN follows it (better on "
        f"{int((cls.euc_lo > 0).sum())} sets, worse on {int((cls.euc_hi < 0).sum())}). A "
        f"linear probe is better on {int((cls.probe_lo > 0).sum())} sets but worse on "
        f"{len(probe_worse)}: "
        + ", ".join(f"`{r.dataset}` ({delta(r.probe_diff)})" for r in probe_worse.itertuples())
        + f". All are image sets where the raw probe is already at {100 * probe_floor:.0f}% "
        "or more. A likely reason is that the lower-dimensional output (128 or 256) keeps the "
        "neighbourhoods but drops some directions that a probe on the full embedding can "
        "use. If the downstream model is a linear classifier, compare "
        "with the raw embedding on held-out data before switching.", "",
        "### Regression", "",
        "The refiner was picked on all four regression sets and improved each one. The "
        "largest change is on TAPE fluorescence, where 10-NN on raw ESM-2 features scores "
        f"below zero (R² {reg.set_index('source').raw.get('fluorescence', np.nan):.2f}, worse "
        "than predicting the mean; the test variants carry more mutations than the training "
        "ones, so this is extrapolation) and the refiner reaches "
        f"{reg.set_index('source').after.get('fluorescence', np.nan):.2f}. On Yelp the "
        f"gain is {delta(reg.set_index('source')['diff'].get('yelp', np.nan), True)}.", "",
        "### Cost", "",
        f"A whole `fit` took {df.fit_time.min():.0f} s to {df.fit_time.max() / 60:.1f} min "
        f"(median {df.fit_time.median():.0f} s) on one GPU, most of it training the refiner. "
        "The slowest sets have 50,000 training rows. Beyond 50,000 rows the selection runs "
        "on a subsample (`selection_max_rows`); `docs/manual.md` section 3.5 has time and "
        "memory up to a million rows.", "",
        "## Limitations", "",
        "- **One seed per set.** The intervals cover the sampling of test rows, not the "
        "randomness of training or of the held-out split.",
        "- **kNN is the criterion.** The default optimises neighbour-based accuracy; other "
        "uses (linear probes, retrieval with other metrics, clustering) can differ, as the "
        "probe column shows.",
        "- **With labels only.** Without labels the default is centre + L2, which no "
        "label-free score can validate; experiment E4 measured it (`benchmarks/results/e4/"
        "REPORT.md`).",
        "- **Encoders as published.** Each encoder was used frozen with one pooling choice; "
        "fine-tuning the encoder itself is outside this comparison.",
        "- **Geometry not reported.** How hubness, isotropy or intrinsic dimension change "
        "after the transform was not measured on these runs.",
    ]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    main()
