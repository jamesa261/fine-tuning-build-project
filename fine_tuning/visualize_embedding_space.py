from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer

from fine_tuning.train import load_params, tsne_plot, _get_device


def main():
    parser = argparse.ArgumentParser(description="Regenerate t-SNE plot for embeddings.")
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("fine_tuning/data/datasets/test_ds.csv"),
        help="Path to test dataset CSV",
    )
    args = parser.parse_args()

    params = load_params(args.params)
    device = _get_device()
    test_df = pd.read_csv(args.dataset)

    base_model = SentenceTransformer(params["base_model"], device=str(device))
    tuned_model = SentenceTransformer(str(params["model_save_dir"]), device=str(device))

    plot_path = tsne_plot(base_model, tuned_model, test_df, params)
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
