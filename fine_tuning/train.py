from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, IterableDataset, Features, Value
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


DEFAULT_PARAMS = {
    "base_model": "sentence-transformers/all-MiniLM-L6-v2",
    "jitter_path": "synthetic_data/data/jittered_titles.csv",
    "num_epochs": 5,
    "train_batch_size": 192,
    "negatives_per_positive": 5,
    "evals_per_epoch": 4,
    "val_fraction": 0.2,
    "test_fraction": 0.1,
    "random_seed": 42,
    "max_steps": None,
    "tsne_subset_size": 500,
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


def build_train_dataset(train_df: pd.DataFrame, take_longest_variant: bool = True):
    rng = np.random.default_rng()
    df = train_df.copy()
    if take_longest_variant:
        df["seed_title"] = df["seed_title"].apply(clean_title)

    anchors = df["jittered_title"].to_numpy()
    positives = df["seed_title"].to_numpy()
    seed_titles = df["seed_title"].unique()

    negative_indices_list = [
        np.arange(len(seed_titles))[seed_titles != positive] for positive in positives
    ]

    def generator():
        indices = list(range(len(anchors)))
        while True:
            rng.shuffle(indices)
            for idx in indices:
                negative_idx = rng.choice(negative_indices_list[idx])
                yield {
                    "anchor": anchors[idx],
                    "positive": positives[idx],
                    "negative": seed_titles[negative_idx],
                }

    return (
        IterableDataset.from_generator(
            generator,
            features=Features(
                {"anchor": Value("string"), "positive": Value("string"), "negative": Value("string")}
            ),
        ).with_format(None)
    )


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

    return Dataset.from_list(examples).with_format(None)


def compute_triplet_metrics(model: SentenceTransformer, df: pd.DataFrame, seed: int):
    rng = np.random.default_rng(seed)
    sample = df.sample(min(500, len(df)), random_state=seed)
    seed_titles = df["seed_title"].unique()

    anchors = sample["jittered_title"].tolist()
    positives = sample["seed_title"].tolist()
    negatives = [rng.choice(seed_titles[seed_titles != p]) for p in positives]

    all_text = anchors + positives + negatives
    embeddings = model.encode(
        all_text, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False
    )
    n = len(anchors)
    anchor_emb = embeddings[:n]
    positive_emb = embeddings[n : 2 * n]
    negative_emb = embeddings[2 * n :]

    pos_sim = torch.sum(anchor_emb * positive_emb, dim=1)
    neg_sim = torch.sum(anchor_emb * negative_emb, dim=1)
    margin = (pos_sim - neg_sim).mean().item()
    return {
        "positive_similarity_mean": pos_sim.mean().item(),
        "negative_similarity_mean": neg_sim.mean().item(),
        "avg_margin": margin,
    }


def tsne_plot(
    base_model: SentenceTransformer,
    tuned_model: SentenceTransformer,
    test_df: pd.DataFrame,
    params: Dict,
):
    subset = test_df.sample(
        min(params["tsne_subset_size"], len(test_df)), random_state=params["random_seed"]
    )
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
        n_iter=params["tsne_n_iter"],
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

    train_ds = build_train_dataset(train_df)
    val_ds = build_val_dataset(val_df, params["negatives_per_positive"], params["random_seed"])

    base_model = SentenceTransformer(params["base_model"], device=str(device))
    train_loss = losses.TripletLoss(model=base_model, triplet_margin=0.3)

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
    )

    trainer = SentenceTransformerTrainer(
        model=base_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        loss=train_loss,
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
