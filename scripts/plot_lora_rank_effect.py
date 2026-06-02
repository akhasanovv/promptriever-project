from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


OUTPUT_PATH = Path("tex/figures/lora_rank_effect.png")


def main() -> None:
    data = [
        {
            "model": "m-e5-base",
            "rank": 8,
            "alpha": 16,
            "dropout": 0.05,
            "nDCG@10": 61.392,
            "pMRR": -0.09782,
        },
        {
            "model": "m-e5-base",
            "rank": 16,
            "alpha": 32,
            "dropout": 0.05,
            "nDCG@10": 59.894,
            "pMRR": 0.61896,
        },
        {
            "model": "m-e5-base",
            "rank": 32,
            "alpha": 64,
            "dropout": 0.05,
            "nDCG@10": 59.785,
            "pMRR": -0.07843,
        },
        {
            "model": "bge-m3",
            "rank": 8,
            "alpha": 16,
            "dropout": 0.05,
            "nDCG@10": 63.579,
            "pMRR": -0.08945,
        },
        {
            "model": "bge-m3",
            "rank": 16,
            "alpha": 32,
            "dropout": 0.05,
            "nDCG@10": 61.962,
            "pMRR": -0.05129,
        },
    ]
    df = pd.DataFrame(data)
    sns.set_theme(
        style="whitegrid",
        font_scale=1.1,
        rc={
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "font.weight": "bold",
        },
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=False)
    models = ["m-e5-base", "bge-m3"]
    metrics = ["pMRR", "nDCG@10"]
    metric_labels = {
        "pMRR": "pMRR",
        "nDCG@10": "nDCG@10",
    }
    model_labels = {
        "m-e5-base": "Multilingual E5-base",
        "bge-m3": "BGE-M3",
    }
    colors = {
        "pMRR": "#2F4F6F",
        "nDCG@10": "#4F8F62",
    }

    for row, model_name in enumerate(models):
        model_df = df[df["model"] == model_name]
        for col, metric in enumerate(metrics):
            ax = axes[row][col]
            sns.lineplot(
                data=model_df,
                x="rank",
                y=metric,
                marker="o",
                markersize=8,
                linewidth=2.8,
                color=colors[metric],
                ax=ax,
            )
            ax.set_title(f"{model_labels[model_name]}, {metric_labels[metric]}", fontsize=13, pad=10)
            ax.set_xlabel("LoRA rank, r", fontsize=12)
            ax.set_ylabel(metric_labels[metric], fontsize=12)
            ax.set_xticks(sorted(model_df["rank"].unique()))
            ax.tick_params(axis="both", labelsize=11)
            ax.grid(True, alpha=0.35)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
