from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset
from datasets import Dataset, IterableDataset, Features, Value
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.data_collator import SentenceTransformerDataCollator
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


DEFAULT_PARAMS = {
    "base_model": "sentence-transformers/all-MiniLM-L6-v2",
    "jitter_path": "synthetic_data/data/jittered_titles.csv",
    "dynamic_negatives": True,
    "num_epochs": 5,
    "train_batch_size": 192,
    "negatives_per_positive": 5,
    "evals_per_epoch": 4,
    "val_fraction": 0.2,
    "test_fraction": 0.1,
    "random_seed": 42,
    "max_steps": None,
    "tsne_subset_size": 500,
    "tsne_label_subset": 20,
    "tsne_perplexity": 20,
    "tsne_learning_rate": 200,
    "tsne_n_iter": 1000,
    "model_save_dir": "fine_tuning/data/trained_models",
    "metrics_path": "fine_tuning/metrics/training.json",
    "plot_path": "fine_tuning/plots/embedding_tsne.png",
}


def load_params(path: Path) -> Dict:
    if not path.exists():
        return DEFAULT_PARAMS.copy()
    import yaml

    params = yaml.safe_load(path.read_text()) or {}
    ft = params.get("fine_tuning", {})
    return DEFAULT_PARAMS | ft


def stratified_split(df: pd.DataFrame, val_frac: float, test_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    val_titles = []
    test_titles = []

    for _, group in df.groupby("onet_code"):
        titles = group["seed_title"].unique()
        rng.shuffle(titles)
        n_val = max(1, int(len(titles) * val_frac))
        n_test = max(1, int(len(titles) * test_frac))
        val_titles.extend(titles[:n_val])
        test_titles.extend(titles[n_val : n_val + n_test])

    val_df = df[df["seed_title"].isin(val_titles)]
    test_df = df[df["seed_title"].isin(test_titles)]
    train_df = df[
        (~df["seed_title"].isin(val_titles)) & (~df["seed_title"].isin(test_titles))
    ]
    return train_df, val_df, test_df


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def clean_title(title: str) -> str:
    # If title contains abbreviation in parentheses, keep the longer string.
    import re

    m = re.search(r"\\(.*\\)$", title)
    if not m:
        return title
    abbreviated_title = m.group()[1:-1]
    un_abbreviated_title = title[: m.span()[0] - 1]
    return max((abbreviated_title, un_abbreviated_title), key=len)


class DynamicTripletDataset(TorchDataset):
    """PyTorch dataset that resamples negatives on every __getitem__ call."""

    def __init__(self, anchors, positives, seed_titles, seed: int):
        self.anchors = anchors
        self.positives = positives
        self.seed_titles = seed_titles
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, idx):
        positive = self.positives[idx]
        neg_pool = self.seed_titles[self.seed_titles != positive]
        negative = self.rng.choice(neg_pool)
        return {"anchor": self.anchors[idx], "positive": positive, "negative": negative}
    
    @property
    def column_names(self):
        # minimal interface expected by SentenceTransformerTrainer
        return ["anchor", "positive", "negative"]


def build_train_dataset(train_df: pd.DataFrame, take_longest_variant: bool = True, dynamic_negatives: bool = True, seed: int = 42):
    df = train_df.copy()
    if take_longest_variant:
        df["seed_title"] = df["seed_title"].apply(clean_title)

    anchors = df["jittered_title"].to_numpy()
    positives = df["seed_title"].to_numpy()
    seed_titles = df["seed_title"].unique()

    if dynamic_negatives:
        return DynamicTripletDataset(anchors, positives, seed_titles, seed=seed)

    rng = np.random.default_rng(seed)
    negatives = [rng.choice(seed_titles[seed_titles != pos]) for pos in positives]

    triplets = pd.DataFrame(
        {"anchor": anchors, "positive": positives, "negative": negatives},
        columns=["anchor", "positive", "negative"],
    )
    return Dataset.from_pandas(triplets, preserve_index=False)


def build_val_dataset(val_df: pd.DataFrame, negatives_per_positive: int, seed: int):
    rng = np.random.default_rng(seed)
    df = val_df.copy()
    df["seed_title"] = df["seed_title"].apply(clean_title)
    anchors = df["jittered_title"].to_numpy()
    positives = df["seed_title"].to_numpy()
    seed_titles = df["seed_title"].unique()

    examples = []
    for anchor, positive in zip(anchors, positives):
        negatives = rng.choice(
            seed_titles[seed_titles != positive],
            size=min(negatives_per_positive, len(seed_titles) - 1),
            replace=False,
        )
        for negative in negatives:
            examples.append({"anchor": anchor, "positive": positive, "negative": negative})

    ds = Dataset.from_list(examples)
    return ds.select_columns(["anchor", "positive", "negative"]).with_format(None)


def compute_triplet_metrics(model: SentenceTransformer, df: pd.DataFrame, seed: int):
    rng = np.random.default_rng(seed)
    sample = df.sample(min(500, len(df)), random_state=seed)
    seed_titles = df["seed_title"].unique()

    anchors = sample["jittered_title"].tolist()
    positives = sample["seed_title"].tolist()
    negatives = [rng.choice(seed_titles[seed_titles != p]) for p in positives]

    all_text = anchors + positives + negatives
    # Compute both normalized (cosine) and raw embeddings for consistency checks
    emb_norm = model.encode(
        all_text, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False
    )
    emb_raw = model.encode(
        all_text, normalize_embeddings=False, convert_to_tensor=True, show_progress_bar=False
    )
    n = len(anchors)
    anchor_norm, positive_norm, negative_norm = emb_norm[:n], emb_norm[n : 2 * n], emb_norm[2 * n :]
    anchor_raw, positive_raw, negative_raw = emb_raw[:n], emb_raw[n : 2 * n], emb_raw[2 * n :]

    pos_sim = torch.sum(anchor_norm * positive_norm, dim=1)
    neg_sim = torch.sum(anchor_norm * negative_norm, dim=1)
    cos_margin = (pos_sim - neg_sim).mean().item()

    pos_dist = torch.linalg.norm(anchor_raw - positive_raw, dim=1)
    neg_dist = torch.linalg.norm(anchor_raw - negative_raw, dim=1)
    euclid_margin = (neg_dist - pos_dist).mean().item()
    return {
        "cosine_positive_mean": pos_sim.mean().item(),
        "cosine_negative_mean": neg_sim.mean().item(),
        "cosine_margin": cos_margin,
        "euclidean_positive_mean": pos_dist.mean().item(),
        "euclidean_negative_mean": neg_dist.mean().item(),
        "euclidean_margin": euclid_margin,
    }


class OrderedTripletCollator:
    """
    Wrapper to guarantee anchor/positive/negative column order before tokenization.
    Prevents datasets with alphabetized columns from swapping positive/negative.
    """

    def __init__(self, base_collator: SentenceTransformerDataCollator, order=("anchor", "positive", "negative")):
        self.base_collator = base_collator
        self.order_index = {name: idx for idx, name in enumerate(order)}
        # passthrough attributes used by Trainer for dataloader setup
        self.valid_label_columns = getattr(base_collator, "valid_label_columns", [])
        self.router_mapping = getattr(base_collator, "router_mapping", {})
        self.prompts = getattr(base_collator, "prompts", {})
        self.include_prompt_lengths = getattr(base_collator, "include_prompt_lengths", False)

    def __call__(self, features):
        def sort_key(key: str):
            return self.order_index.get(key, len(self.order_index))

        reordered = [{k: f[k] for k in sorted(f.keys(), key=sort_key)} for f in features]
        return self.base_collator(reordered)


def tsne_plot(
    base_model: SentenceTransformer,
    tuned_model: SentenceTransformer,
    test_df: pd.DataFrame,
    params: Dict,
):
    rng = np.random.default_rng(params["random_seed"])
    all_labels = test_df["seed_title"].unique().tolist()
    chosen_labels = set(
        rng.choice(all_labels, size=min(params["tsne_label_subset"], len(all_labels)), replace=False)
    )
    subset = test_df[test_df["seed_title"].isin(chosen_labels)]
    if len(subset) > params["tsne_subset_size"]:
        subset = subset.sample(params["tsne_subset_size"], random_state=params["random_seed"])

    titles = subset["jittered_title"].tolist()
    labels = subset["seed_title"].tolist()

    base_embeddings = base_model.encode(
        titles, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    )
    tuned_embeddings = tuned_model.encode(
        titles, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    )

    tsne = TSNE(
        n_components=2,
        perplexity=params["tsne_perplexity"],
        learning_rate=params["tsne_learning_rate"],
        max_iter=params["tsne_n_iter"],
        random_state=params["random_seed"],
        init="random",
    )
    base_2d = tsne.fit_transform(base_embeddings)
    tuned_2d = tsne.fit_transform(tuned_embeddings)

    unique_labels = list(subset["seed_title"].unique())
    colors = plt.cm.get_cmap("tab20", len(unique_labels))
    color_map = {label: colors(i) for i, label in enumerate(unique_labels)}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=False, sharey=False)
    for label in unique_labels:
        idx = subset["seed_title"] == label
        axes[0].scatter(base_2d[idx, 0], base_2d[idx, 1], color=color_map[label], s=20)
        axes[1].scatter(tuned_2d[idx, 0], tuned_2d[idx, 1], color=color_map[label], s=20)

    axes[0].set_title("Base model (t-SNE)")
    axes[1].set_title("Fine-tuned model (t-SNE)")
    for ax in axes:
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")

    # Legend
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[label], markersize=6, label=label)
        for label in unique_labels
    ]
    axes[1].legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")

    plot_path = Path(params["plot_path"])
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def train(params: Dict):
    device = _get_device()
    jitter_df = pd.read_csv(params["jitter_path"])
    train_df, val_df, test_df = stratified_split(
        jitter_df, params["val_fraction"], params["test_fraction"], seed=params["random_seed"]
    )

    train_ds = build_train_dataset(
        train_df,
        dynamic_negatives=params.get("dynamic_negatives", True),
        seed=params["random_seed"],
    )
    # Trainer expects Datasets to expose features/column_names; wrap torch Dataset in iterable if needed
    if isinstance(train_ds, TorchDataset):
        train_ds = Dataset.from_generator(
            lambda: (train_ds[i] for i in range(len(train_ds))),
            features=Features(
                {"anchor": Value("string"), "positive": Value("string"), "negative": Value("string")}
            ),
        )
    val_ds = build_val_dataset(val_df, params["negatives_per_positive"], params["random_seed"])

    base_model = SentenceTransformer(params["base_model"], device=str(device))
    train_loss = losses.TripletLoss(
        model=base_model,
        distance_metric=losses.TripletDistanceMetric.COSINE,
        triplet_margin=0.1,
    )

    steps_per_epoch = max(1, len(train_df) // params["train_batch_size"])
    eval_steps = max(1, steps_per_epoch // max(1, params["evals_per_epoch"]))
    max_steps = params["max_steps"] or steps_per_epoch * params["num_epochs"]

    output_dir = Path(params["model_save_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=params["num_epochs"],
        max_steps=max_steps,
        warmup_ratio=0.1,
        per_device_train_batch_size=params["train_batch_size"],
        per_device_eval_batch_size=params["train_batch_size"],
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=2,
        logging_steps=eval_steps,
        report_to="tensorboard",
        dataloader_pin_memory=torch.cuda.is_available(),
    )

    trainer = SentenceTransformerTrainer(
        model=base_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        loss=train_loss,
        data_collator=OrderedTripletCollator(
            SentenceTransformerDataCollator(tokenize_fn=base_model.tokenize),
            order=("anchor", "positive", "negative"),
        ),
    )

    trainer.train()
    trainer.save_model(str(output_dir))

    tuned_model = SentenceTransformer(str(output_dir), device=str(device))
    triplet_metrics = compute_triplet_metrics(tuned_model, val_df, params["random_seed"])
    plot_path = tsne_plot(
        base_model=SentenceTransformer(params["base_model"], device=str(device)),
        tuned_model=tuned_model,
        test_df=test_df,
        params=params,
    )

    metrics = {
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "final_eval_loss": trainer.state.log_history[-1].get("eval_loss")
        if trainer.state.log_history
        else None,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "triplet_eval": triplet_metrics,
        "plot_path": str(plot_path),
    }

    metrics_path = Path(params["metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    # Persist splits for reproducibility
    ds_dir = Path("fine_tuning/data/datasets")
    ds_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(ds_dir / "train_ds.csv", index=False)
    val_df.to_csv(ds_dir / "val_ds.csv", index=False)
    test_df.to_csv(ds_dir / "test_ds.csv", index=False)

    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train fine-tuned embeddings on jittered titles.")
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    parser.add_argument("--jitter-path", type=Path, help="Override jittered_titles.csv path")
    return parser.parse_args()


def main():
    args = parse_args()
    params = load_params(args.params)
    if args.jitter_path:
        params["jitter_path"] = str(args.jitter_path)
    metrics = train(params)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
