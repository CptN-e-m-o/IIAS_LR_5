"""
ЛР5. Экспериментальная оценка подхода к маршрутизации helpdesk-тикетов.

Скрипт запускает серию экспериментов на датасете customer-support-tickets:
- ориентиры сравнения: Dummy, Subject-only, Body-only, Subject+Body, Main, Ablation no balancing;
- 7 повторных запусков с разными seed;
- метрики: accuracy, macro-F1, weighted-F1;
- 95% доверительные интервалы и paired t-test относительно основного метода;
- графики: boxplot, confusion matrix, learning curve, PR curve;
- качественный анализ ошибок.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer

RANDOM_SEEDS = [42, 43, 44, 45, 46, 47, 48]
TEXT_COLUMNS = ["subject", "body", "language"]
TARGET_COLUMN = "queue"


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    use_subject: bool = True
    use_body: bool = True
    use_language: bool = True
    use_tags: bool = False
    ngram_max: int = 2
    max_features: int | None = None
    class_weight: str | None = "balanced"
    dummy: bool = False


def ensure_columns(df: pd.DataFrame) -> None:
    required = {"subject", "body", "queue", "priority", "language"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"В датасете отсутствуют обязательные колонки: {missing}")


def make_text(df: pd.DataFrame, cfg: ExperimentConfig) -> pd.Series:
    parts: List[pd.Series] = []
    if cfg.use_subject:
        parts.append(df["subject"].fillna("").astype(str))
    if cfg.use_body:
        parts.append(df["body"].fillna("").astype(str))
    if cfg.use_language:
        parts.append("language_" + df["language"].fillna("unknown").astype(str))
    if cfg.use_tags:
        tag_cols = [c for c in df.columns if c.startswith("tag_")]
        if tag_cols:
            tag_text = df[tag_cols].fillna("").astype(str).agg(" ".join, axis=1)
            parts.append(tag_text)
    if not parts:
        return pd.Series([""] * len(df), index=df.index)
    return pd.concat(parts, axis=1).agg("\n".join, axis=1)


def prepare_data(path: str, quick: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path)
    ensure_columns(df)
    df = df.dropna(subset=[TARGET_COLUMN]).copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str)
    df["subject"] = df["subject"].fillna("").astype(str)
    df["body"] = df["body"].fillna("").astype(str)
    df["text_len"] = (df["subject"] + " " + df["body"]).str.len()
    # Удаляем полностью пустые обращения.
    df = df[df["text_len"] > 0].reset_index(drop=True)
    if quick:
        # Сохраняем стратификацию и оставляем достаточно объектов для быстрой проверки.
        _, df = train_test_split(
            df,
            test_size=min(6000, len(df)) / len(df),
            random_state=42,
            stratify=df[TARGET_COLUMN],
        )
        df = df.reset_index(drop=True)
    return df


def split_data(df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=seed,
        stratify=df[TARGET_COLUMN],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
        stratify=temp_df[TARGET_COLUMN],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_model(cfg: ExperimentConfig, quick: bool = False) -> Pipeline:
    if cfg.dummy:
        return Pipeline([("model", DummyClassifier(strategy="most_frequent"))])
    max_features = cfg.max_features if cfg.max_features is not None else (8000 if quick else 20000)
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, cfg.ngram_max),
                    min_df=2,
                    max_df=0.95,
                    max_features=max_features,
                    strip_accents="unicode",
                ),
            ),
            (
                "clf",
                ComplementNB(alpha=0.1, norm=False),
            ),
        ]
    )


def evaluate(y_true: Iterable[str], y_pred: Iterable[str]) -> Dict[str, float]:
    y_true = list(y_true)
    y_pred = list(y_pred)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }


def run_one(df: pd.DataFrame, cfg: ExperimentConfig, seed: int, quick: bool) -> Dict[str, object]:
    train_df, val_df, test_df = split_data(df, seed)
    X_train = make_text(train_df, cfg)
    X_val = make_text(val_df, cfg)
    X_test = make_text(test_df, cfg)
    y_train = train_df[TARGET_COLUMN]
    y_val = val_df[TARGET_COLUMN]
    y_test = test_df[TARGET_COLUMN]

    model = build_model(cfg, quick=quick)
    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    result = {
        "experiment": cfg.name,
        "seed": seed,
        "val": evaluate(y_val, val_pred),
        "test": evaluate(y_test, test_pred),
    }
    return result


def mean_ci(values: List[float]) -> Tuple[float, float]:
    arr = np.array(values, dtype=float)
    mean = float(np.mean(arr))
    if len(arr) <= 1:
        return mean, 0.0
    se = stats.sem(arr)
    ci = float(se * stats.t.ppf((1 + 0.95) / 2.0, len(arr) - 1))
    return mean, ci


def summarize(results: pd.DataFrame, main_name: str = "Main: subject+body+language") -> pd.DataFrame:
    rows = []
    main = results[results["experiment"] == main_name].sort_values("seed")
    for exp, g in results.groupby("experiment"):
        row = {"experiment": exp}
        for metric in ["accuracy", "macro_f1", "weighted_f1"]:
            vals = g.sort_values("seed")[metric].tolist()
            m, ci = mean_ci(vals)
            row[f"{metric}_mean"] = m
            row[f"{metric}_ci95"] = ci
        # p-value относительно основного метода по macro-F1.
        if exp == main_name:
            row["p_value_vs_main_macro_f1"] = np.nan
        else:
            paired = g.sort_values("seed")[["seed", "macro_f1"]].merge(
                main[["seed", "macro_f1"]], on="seed", suffixes=("_exp", "_main")
            )
            if len(paired) >= 2:
                row["p_value_vs_main_macro_f1"] = float(
                    stats.ttest_rel(paired["macro_f1_main"], paired["macro_f1_exp"]).pvalue
                )
            else:
                row["p_value_vs_main_macro_f1"] = np.nan
        rows.append(row)
    summary = pd.DataFrame(rows)
    return summary.sort_values("macro_f1_mean", ascending=False).reset_index(drop=True)


def save_boxplot(results: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(11, 5))
    order = results.groupby("experiment")["macro_f1"].mean().sort_values(ascending=False).index
    data = [results[results["experiment"] == exp]["macro_f1"].values for exp in order]
    plt.boxplot(data, tick_labels=[e.replace("Main: ", "Main\n").replace("Ablation: ", "Abl.\n") for e in order], showmeans=True)
    plt.ylabel("Macro-F1")
    plt.title("Сравнение подходов по повторным запускам")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "01_boxplot_macro_f1.png", dpi=180)
    plt.close()


def train_main_for_artifacts(df: pd.DataFrame, seed: int, quick: bool, out_dir: Path) -> None:
    cfg = ExperimentConfig("Main: subject+body+language")
    train_df, val_df, test_df = split_data(df, seed)
    X_train = make_text(train_df, cfg)
    X_test = make_text(test_df, cfg)
    y_train = train_df[TARGET_COLUMN]
    y_test = test_df[TARGET_COLUMN]

    model = build_model(cfg, quick=quick)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, pred, labels=labels, normalize="true")
    plt.figure(figsize=(9, 7))
    plt.imshow(cm, aspect="auto")
    plt.colorbar(label="Доля объектов класса")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(len(labels)), labels, fontsize=8)
    plt.xlabel("Предсказанный класс")
    plt.ylabel("Истинный класс")
    plt.title("Нормированная матрица ошибок основного метода")
    plt.tight_layout()
    plt.savefig(out_dir / "02_confusion_matrix.png", dpi=180)
    plt.close()

    # PR curve micro-average для многоклассовой задачи.
    if hasattr(model.named_steps.get("clf"), "predict_proba"):
        proba = model.predict_proba(X_test)
        lb = LabelBinarizer()
        y_bin = lb.fit_transform(y_train)
        # Гарантируем соответствие классов классификатора.
        lb.classes_ = model.named_steps["clf"].classes_
        y_test_bin = lb.transform(y_test)
        precision, recall, _ = precision_recall_curve(y_test_bin.ravel(), proba.ravel())
        ap = average_precision_score(y_test_bin, proba, average="micro")
        plt.figure(figsize=(7, 5))
        plt.plot(recall, precision, label=f"micro-AP = {ap:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("PR-кривая для многоклассовой маршрутизации")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "03_pr_curve_micro.png", dpi=180)
        plt.close()

    # Ошибки.
    errors = test_df.copy()
    errors["predicted_queue"] = pred
    errors = errors[errors[TARGET_COLUMN] != errors["predicted_queue"]]
    errors["text_preview"] = (errors["subject"].fillna("") + " | " + errors["body"].fillna("")).str.slice(0, 260)
    errors[["subject", "queue", "predicted_queue", "priority", "language", "text_preview"]].head(10).to_csv(
        out_dir / "04_error_examples.csv", index=False
    )

    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    with open(out_dir / "05_classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def save_learning_curve(df: pd.DataFrame, out_dir: Path, quick: bool) -> None:
    cfg = ExperimentConfig("Main: subject+body+language")
    fractions = [0.1, 0.25, 0.5, 0.75, 1.0]
    seeds = RANDOM_SEEDS[:5]
    rows = []
    for seed in seeds:
        train_df, _, test_df = split_data(df, seed)
        for frac in fractions:
            if frac < 1.0:
                sample_df, _ = train_test_split(
                    train_df,
                    train_size=frac,
                    random_state=seed,
                    stratify=train_df[TARGET_COLUMN],
                )
            else:
                sample_df = train_df
            X_train = make_text(sample_df, cfg)
            y_train = sample_df[TARGET_COLUMN]
            X_test = make_text(test_df, cfg)
            y_test = test_df[TARGET_COLUMN]
            model = build_model(cfg, quick=quick)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            rows.append({"seed": seed, "fraction": frac, "macro_f1": f1_score(y_test, pred, average="macro")})
    lc = pd.DataFrame(rows)
    lc.to_csv(out_dir / "06_learning_curve_raw.csv", index=False)
    grouped = lc.groupby("fraction")["macro_f1"].agg(["mean", "std", "count"]).reset_index()
    grouped["ci95"] = grouped.apply(lambda r: stats.t.ppf(0.975, r["count"] - 1) * r["std"] / math.sqrt(r["count"]), axis=1)
    plt.figure(figsize=(7, 5))
    plt.plot(grouped["fraction"], grouped["mean"], marker="o")
    plt.fill_between(grouped["fraction"], grouped["mean"] - grouped["ci95"], grouped["mean"] + grouped["ci95"], alpha=0.25)
    plt.xlabel("Доля обучающей выборки")
    plt.ylabel("Macro-F1")
    plt.title("Кривая обучения основного метода, среднее ± 95% CI")
    plt.tight_layout()
    plt.savefig(out_dir / "07_learning_curve.png", dpi=180)
    plt.close()


def save_environment(out_dir: Path) -> None:
    import sklearn
    import scipy
    import matplotlib

    env = {
        "python": platform.python_version(),
        "os": "Windows 10/11 x64 или локальное окружение Python",
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "cpu": "локальный персональный компьютер",
        "ram": "не менее 8 GB",
    }
    with open(out_dir / "environment.json", "w", encoding="utf-8") as f:
        json.dump(env, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Путь к customer_support_tickets.csv")
    parser.add_argument("--output-dir", default="lr5_outputs", help="Папка для результатов")
    parser.add_argument("--quick", action="store_true", help="Быстрая проверка на подвыборке")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = prepare_data(args.data, quick=args.quick)
    configs = [
        ExperimentConfig("Dummy: most frequent", dummy=True),
        ExperimentConfig("Subject only", use_subject=True, use_body=False, use_language=False, class_weight="balanced"),
        ExperimentConfig("Body only", use_subject=False, use_body=True, use_language=False, class_weight="balanced"),
        ExperimentConfig("Subject+Body", use_subject=True, use_body=True, use_language=False, class_weight="balanced"),
        ExperimentConfig("Main: subject+body+language", use_subject=True, use_body=True, use_language=True, class_weight="balanced"),
        ExperimentConfig("Ablation: unigrams only", use_subject=True, use_body=True, use_language=True, ngram_max=1, class_weight="balanced"),
        ExperimentConfig("Ablation: small vocabulary", use_subject=True, use_body=True, use_language=True, max_features=3000, class_weight="balanced"),
        ExperimentConfig("Ablation: +tags", use_subject=True, use_body=True, use_language=True, use_tags=True, class_weight="balanced"),
    ]

    rows = []
    for cfg in configs:
        for seed in RANDOM_SEEDS:
            print(f"Running {cfg.name} seed={seed}")
            result = run_one(df, cfg, seed, quick=args.quick)
            rows.append(
                {
                    "experiment": result["experiment"],
                    "seed": result["seed"],
                    "val_accuracy": result["val"]["accuracy"],
                    "val_macro_f1": result["val"]["macro_f1"],
                    "val_weighted_f1": result["val"]["weighted_f1"],
                    "accuracy": result["test"]["accuracy"],
                    "macro_f1": result["test"]["macro_f1"],
                    "weighted_f1": result["test"]["weighted_f1"],
                }
            )
    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "lr5_experiment_runs.csv", index=False)
    summary = summarize(results)
    summary.to_csv(out_dir / "lr5_experiment_summary.csv", index=False)

    save_boxplot(results, out_dir)
    train_main_for_artifacts(df, seed=42, quick=args.quick, out_dir=out_dir)
    save_learning_curve(df, out_dir, quick=args.quick)
    save_environment(out_dir)

    print("\nSummary:")
    print(summary.to_string(index=False))
    print(f"\nSaved outputs to {out_dir}")


if __name__ == "__main__":
    main()
