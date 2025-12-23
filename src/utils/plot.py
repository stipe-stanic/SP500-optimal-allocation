from pathlib import Path

import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(
    history_df: pl.DataFrame,
    filepath: str | Path | None = None,
    figsize: tuple = (12, 10),
) -> None:
    sns.set_style("whitegrid")
    df = history_df.to_pandas()

    fig, ax1 = plt.subplots(1, 1, figsize=figsize)

    color_train = "tab:blue"
    color_val = "tab:orange"
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    line1 = ax1.plot(
        df["epoch"],
        df["train_loss"],
        color=color_train,
        label="Train Loss",
        linewidth=1,
        marker="o",
        markersize=2,
        alpha=0.7,
    )
    line2 = ax1.plot(
        df["epoch"],
        df["val_loss"],
        color=color_val,
        label="Val Loss",
        linewidth=1,
        marker="s",
        markersize=2,
        alpha=0.7,
    )
    ax1.tick_params(axis="y")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color_score = "tab:green"
    ax2.set_ylabel("Val Score", fontsize=12, color=color_score)
    line3 = ax2.plot(
        df["epoch"],
        df["val_score"],
        color=color_score,
        label="Val Score",
        linewidth=1,
        marker="^",
        markersize=2,
    )
    ax2.tick_params(axis="y", labelcolor=color_score)

    best_epochs = df[df["is_best"]]
    if len(best_epochs) > 0:
        ax2.scatter(
            best_epochs["epoch"],
            best_epochs["val_score"],
            color="red",
            s=100,
            zorder=5,
            marker="*",
            label="Best",
        )

    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]  # noqa
    if len(best_epochs) > 0:
        best_marker = plt.Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="red",
            markersize=5,
            label="Best",
        )
        lines.append(best_marker)
        labels.append("Best")
    ax1.legend(lines, labels, loc="upper right", framealpha=0.9)
    ax1.set_title("Training History", fontsize=14, fontweight="bold", pad=10)

    plt.suptitle("Model Training Metrics", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()

    if filepath:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath, dpi=100, bbox_inches="tight")

    plt.show()


def plot_prediction_distribution(
    predictions: np.ndarray,
    targets: np.ndarray,
    filepath: str | Path | None = None,
    figsize: tuple = (10, 6),
) -> None:
    """Plot distribution of predictions and targets."""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(targets, bins=50, alpha=0.6, label="Target", color="tab:blue", density=True)
    ax.hist(predictions, bins=50, alpha=0.6, label="Prediction", color="tab:orange", density=True)

    ax.set_xlabel("Value", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Distribution of Predictions vs Targets", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if filepath:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath, dpi=100, bbox_inches="tight")

    plt.show()


def plot_prediction_variance(
    predictions: np.ndarray,
    prediction_vars: np.ndarray,
    filepath: str | Path | None = None,
    figsize: tuple = (10, 6),
) -> None:
    """Plot scatter of predictions vs prediction variance."""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        predictions,
        prediction_vars,
        alpha=0.5,
        color="tab:blue",
        s=20,
    )

    ax.set_xlabel("Prediction", fontsize=12)
    ax.set_ylabel("Prediction Variance", fontsize=12)
    ax.set_title("Prediction vs Prediction Variance", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if filepath:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath, dpi=100, bbox_inches="tight")

    plt.show()


def plot_position_distribution(
    positions: np.ndarray,
    filepath: str | Path | None = None,
    figsize: tuple = (10, 6),
) -> None:
    """Plot distribution of positions."""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(positions, bins=50, alpha=0.7, color="tab:green", edgecolor="black")

    ax.axvline(positions.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {positions.mean():.3f}")
    ax.axvline(
        np.median(positions), color="orange", linestyle="--", linewidth=2, label=f"Median: {np.median(positions):.3f}"
    )

    ax.set_xlabel("Position", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Distribution of Positions", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()

    if filepath:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath, dpi=100, bbox_inches="tight")

    plt.show()


def plot_prediction_timeseries(
    date_ids: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    filepath: str | Path | None = None,
    figsize: tuple = (14, 6),
) -> None:
    """Plot time series of predictions and targets."""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(date_ids, targets, label="Target", color="tab:blue", alpha=0.7, linewidth=1.5)
    ax.plot(date_ids, predictions, label="Prediction", color="tab:orange", alpha=0.7, linewidth=1.5)

    ax.set_xlabel("Date ID", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title("Predictions vs Targets over Time", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if filepath:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath, dpi=100, bbox_inches="tight")

    plt.show()


def plot_correlation_heatmap(
    predictions: np.ndarray,
    forward_returns: np.ndarray,
    risk_free_rate: np.ndarray,
    market_forward_excess_returns: np.ndarray,
    positions: np.ndarray,
    filepath: str | Path | None = None,
    figsize: tuple = (10, 8),
) -> None:
    """Plot correlation heatmap of predictions and targets."""
    import pandas as pd

    sns.set_style("white")

    # Create dataframe
    df = pd.DataFrame(
        {
            "Prediction": predictions,
            "Forward Returns": forward_returns,
            "Risk Free Rate": risk_free_rate,
            "Market Forward Excess Returns": market_forward_excess_returns,
            "Position": positions,
        }
    )

    # Calculate correlation matrix
    corr = df.corr()

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )

    ax.set_title("Correlation Matrix", fontsize=14, fontweight="bold", pad=10)

    fig.tight_layout()

    if filepath:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath, dpi=100, bbox_inches="tight")

    plt.show()
