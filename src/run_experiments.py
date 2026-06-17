from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize

SEEDS = list(range(42, 47))
TARGET = "queue"
TAG_COLUMNS = [f"tag_{i}" for i in range(1, 9)]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def safe_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()


def make_features(df: pd.DataFrame, mode: str) -> pd.Series:
    subject = safe_text(df["subject"])
    body = safe_text(df["body"])
    tags = safe_text(df[TAG_COLUMNS].fillna("").agg(" ".join, axis=1))

    if mode == "subject":
        return subject
    if mode == "body":
        return body
    if mode == "subject_body":
        return subject + " " + body
    if mode == "subject_body_tags":
        return subject + " " + body + " " + tags
    raise ValueError(f"Unknown feature mode: {mode}")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    feature_mode: str
    max_features: int | None
    ngram_range: Tuple[int, int]
    alpha: float = 0.1
    is_dummy: bool = False


CONFIGS: List[ExperimentConfig] = [
    ExperimentConfig("Dummy most frequent", "subject_body", None, (1, 1), is_dummy=True),
    ExperimentConfig("Subject only", "subject", 20000, (1, 2)),
    ExperimentConfig("Body only", "body", 20000, (1, 2)),
    ExperimentConfig("Subject + Body", "subject_body", 20000, (1, 2)),
    ExperimentConfig("Small vocabulary", "subject_body", 3000, (1, 2)),
    ExperimentConfig("Subject + Body + tags", "subject_body_tags", 20000, (1, 2)),
]


def build_model(cfg: ExperimentConfig):
    if cfg.is_dummy:
        return DummyClassifier(strategy="most_frequent")
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    max_features=cfg.max_features,
                    ngram_range=cfg.ngram_range,
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            ("clf", ComplementNB(alpha=cfg.alpha)),
        ]
    )


def split_data(df: pd.DataFrame, seed: int):
    y = df[TARGET].astype(str)
    train_val, test = train_test_split(
        df,
        test_size=0.15,
        random_state=seed,
        stratify=y,
    )
    y_train_val = train_val[TARGET].astype(str)
    val_size = 0.15 / 0.85
    train, val = train_test_split(
        train_val,
        test_size=val_size,
        random_state=seed,
        stratify=y_train_val,
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def evaluate_once(df: pd.DataFrame, cfg: ExperimentConfig, seed: int) -> Dict[str, float | str | int]:
    set_seed(seed)
    train, val, test = split_data(df, seed)
    X_train = make_features(train, cfg.feature_mode)
    y_train = train[TARGET].astype(str)
    X_test = make_features(test, cfg.feature_mode)
    y_test = test[TARGET].astype(str)

    model = build_model(cfg)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    return {
        "method": cfg.name,
        "seed": seed,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "accuracy": accuracy_score(y_test, pred),
        "macro_f1": f1_score(y_test, pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_test, pred, average="weighted", zero_division=0),
    }


def ci95(values: List[float]) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    if len(arr) < 2:
        return mean, mean, mean
    half = float(stats.t.ppf(0.975, len(arr) - 1) * arr.std(ddof=1) / math.sqrt(len(arr)))
    return mean, mean - half, mean + half


def format_ci(values: List[float]) -> str:
    mean, lo, hi = ci95(values)
    return f"{mean:.3f} [{lo:.3f}; {hi:.3f}]"


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    baseline = runs[runs["method"] == "Dummy most frequent"].sort_values("seed")
    baseline_macro = baseline["macro_f1"].to_numpy()

    for method, group in runs.groupby("method", sort=False):
        group = group.sort_values("seed")
        macro = group["macro_f1"].to_numpy()
        if method == "Dummy most frequent":
            p_value = np.nan
            delta_macro = 0.0
        else:
            test = stats.ttest_rel(macro, baseline_macro)
            p_value = float(test.pvalue)
            delta_macro = float(macro.mean() - baseline_macro.mean())
        rows.append(
            {
                "method": method,
                "accuracy_mean_ci": format_ci(group["accuracy"].tolist()),
                "macro_f1_mean_ci": format_ci(group["macro_f1"].tolist()),
                "weighted_f1_mean_ci": format_ci(group["weighted_f1"].tolist()),
                "macro_f1_delta_vs_dummy": delta_macro,
                "p_value_vs_dummy": p_value,
            }
        )
    return pd.DataFrame(rows)


def plot_boxplot(runs: pd.DataFrame, out_dir: Path) -> None:
    order = [cfg.name for cfg in CONFIGS]
    data = [runs.loc[runs["method"] == m, "macro_f1"].values for m in order]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.boxplot(data, tick_labels=order, vert=False)
    ax.set_xlabel("Macro-F1")
    ax.set_ylabel("Метод")
    ax.set_title("Распределение Macro-F1 по 7 запускам")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "boxplot_macro_f1.png", dpi=200)
    plt.close(fig)


def plot_confusion_and_errors(df: pd.DataFrame, out_dir: Path, seed: int = 42) -> None:
    cfg = next(c for c in CONFIGS if c.name == "Subject + Body")
    train, val, test = split_data(df, seed)
    X_train = make_features(train, cfg.feature_mode)
    y_train = train[TARGET].astype(str)
    X_test = make_features(test, cfg.feature_mode)
    y_test = test[TARGET].astype(str)
    model = build_model(cfg)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    labels = sorted(y_test.unique().tolist())
    cm = confusion_matrix(y_test, pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, aspect="auto")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Предсказанная очередь")
    ax.set_ylabel("Истинная очередь")
    ax.set_title("Нормализованная матрица ошибок: Subject + Body")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    test_errors = test.copy()
    test_errors["true_queue"] = y_test.values
    test_errors["predicted_queue"] = pred
    test_errors["is_error"] = test_errors["true_queue"] != test_errors["predicted_queue"]
    error_rows = test_errors[test_errors["is_error"]].copy()
    error_rows["text_preview"] = (safe_text(error_rows["subject"]) + " | " + safe_text(error_rows["body"])).str.slice(0, 450)
    error_rows[["subject", "true_queue", "predicted_queue", "language", "priority", "text_preview"]].head(20).to_csv(
        out_dir / "error_examples.csv", index=False
    )
    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(out_dir / "classification_report_subject_body.csv")

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        classes = model.classes_
        y_bin = label_binarize(y_test, classes=classes)
        precision, recall, _ = precision_recall_curve(y_bin.ravel(), proba.ravel())
        ap = average_precision_score(y_bin, proba, average="micro")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, label=f"micro-average AP={ap:.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("PR-кривая: Subject + Body")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "pr_curve.png", dpi=200)
        plt.close(fig)


def plot_learning_curve(df: pd.DataFrame, out_dir: Path) -> None:
    cfg = next(c for c in CONFIGS if c.name == "Subject + Body")
    fractions = [0.25, 0.5, 0.75, 1.0]
    rows = []
    for seed in SEEDS[:3]:
        train, val, test = split_data(df, seed)
        y_train_full = train[TARGET].astype(str)
        X_test = make_features(test, cfg.feature_mode)
        y_test = test[TARGET].astype(str)
        for frac in fractions:
            if frac < 1.0:
                sub_train, _ = train_test_split(
                    train,
                    train_size=frac,
                    random_state=seed,
                    stratify=y_train_full,
                )
            else:
                sub_train = train
            model = build_model(cfg)
            model.fit(make_features(sub_train, cfg.feature_mode), sub_train[TARGET].astype(str))
            pred = model.predict(X_test)
            rows.append({"seed": seed, "fraction": frac, "macro_f1": f1_score(y_test, pred, average="macro", zero_division=0)})
    lc = pd.DataFrame(rows)
    lc.to_csv(out_dir / "learning_curve.csv", index=False)
    grouped = lc.groupby("fraction")["macro_f1"].agg(["mean", "std", "count"]).reset_index()
    grouped["ci"] = stats.t.ppf(0.975, grouped["count"] - 1) * grouped["std"] / np.sqrt(grouped["count"])
    fig, ax = plt.subplots(figsize=(8, 6))
    x = grouped["fraction"] * 100
    y = grouped["mean"]
    ci = grouped["ci"]
    ax.plot(x, y, marker="o")
    ax.fill_between(x, y - ci, y + ci, alpha=0.2)
    ax.set_xlabel("Доля обучающей выборки, %")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Кривая обучения: Subject + Body")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "learning_curve.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="LR5 helpdesk ticket routing experiments")
    parser.add_argument("--data", type=Path, required=True, help="Path to CSV dataset")
    parser.add_argument("--out", type=Path, default=Path("outputs"), help="Output directory")
    parser.add_argument("--language", default="en", help="Language filter: en, de, or all")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data)
    if args.language.lower() != "all":
        df = df[df["language"].astype(str).str.lower() == args.language.lower()].copy()
    required = {"subject", "body", TARGET, *TAG_COLUMNS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)

    metadata = {
        "dataset_rows": int(len(df)),
        "language_filter": args.language,
        "target": TARGET,
        "classes": int(df[TARGET].nunique()),
        "class_distribution": df[TARGET].value_counts().to_dict(),
        "seeds": SEEDS,
        "split": "stratified train/val/test = 70/15/15",
        "configs": [cfg.__dict__ for cfg in CONFIGS],
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    for cfg in CONFIGS:
        for seed in SEEDS:
            print(f"Running {cfg.name} seed={seed}", flush=True)
            rows.append(evaluate_once(df, cfg, seed))
            pd.DataFrame(rows).to_csv(args.out / "runs.csv", index=False)
    runs = pd.DataFrame(rows)
    summary = summarize(runs)
    summary.to_csv(args.out / "summary.csv", index=False)
    plot_boxplot(runs, args.out)
    plot_confusion_and_errors(df, args.out, seed=42)
    plot_learning_curve(df, args.out)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
