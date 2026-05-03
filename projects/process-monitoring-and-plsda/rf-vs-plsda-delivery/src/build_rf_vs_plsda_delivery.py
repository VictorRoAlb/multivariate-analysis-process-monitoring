from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "genz_mental_wellness_synthetic_dataset.csv"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"

TARGET = "Burnout_Risk"
FEATURES = [
    "Anxiety_Score",
    "Mood_Stability_Score",
    "Sleep_Quality_Score",
    "Motivation_Level",
    "Emotional_Fatigue_Score",
    "Wellbeing_Index",
]
CLASS_ORDER = ["Low", "Medium", "High"]
RANDOM_STATE = 123
N_SPLITS = 7


@dataclass
class ModelOutputs:
    name: str
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray
    metrics: dict[str, float]


def ensure_dirs() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_PATH)
    df = df[df[TARGET].isin(CLASS_ORDER)].copy()
    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    return X, y


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, labels: list[str]) -> dict[str, float]:
    y_true_bin = label_binarize(y_true, classes=labels)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1 macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Cohen kappa": cohen_kappa_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "AUROC OvR macro": roc_auc_score(y_true_bin, y_proba, average="macro", multi_class="ovr"),
    }


def align_proba_to_labels(y_proba: np.ndarray, model_classes: np.ndarray, labels: list[str]) -> np.ndarray:
    order = [list(model_classes).index(label) for label in labels]
    return y_proba[:, order]


def softmax_rows(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def vip_scores(pls: PLSRegression) -> np.ndarray:
    t = pls.x_scores_
    w = pls.x_weights_
    q = pls.y_loadings_
    p, h = w.shape
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1).sum(axis=1)
    total_s = np.sum(s)
    vip = np.zeros((p,))
    for j in range(p):
        weight = np.array([(w[j, a] ** 2) / np.sum(w[:, a] ** 2) for a in range(h)])
        vip[j] = np.sqrt(p * np.sum(s * weight) / total_s)
    return vip


def fit_plsda(X_train: np.ndarray, y_train: np.ndarray, n_components: int) -> tuple[PLSRegression, StandardScaler]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    Y = label_binarize(y_train, classes=CLASS_ORDER)
    pls = PLSRegression(n_components=n_components, scale=False)
    pls.fit(Xs, Y)
    return pls, scaler


def predict_plsda(pls: PLSRegression, scaler: StandardScaler, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Xs = scaler.transform(X)
    raw = pls.predict(Xs)
    proba = softmax_rows(raw)
    pred = np.array(CLASS_ORDER)[np.argmax(proba, axis=1)]
    return pred, proba


def select_plsda_components(X_train: np.ndarray, y_train: np.ndarray, max_components: int) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rows: list[dict[str, float]] = []
    for a in range(1, max_components + 1):
        bal_scores = []
        f1_scores = []
        auc_scores = []
        for train_idx, valid_idx in cv.split(X_train, y_train):
            X_tr, X_va = X_train[train_idx], X_train[valid_idx]
            y_tr, y_va = y_train[train_idx], y_train[valid_idx]
            pls, scaler = fit_plsda(X_tr, y_tr, n_components=a)
            pred, proba = predict_plsda(pls, scaler, X_va)
            bal_scores.append(balanced_accuracy_score(y_va, pred))
            f1_scores.append(f1_score(y_va, pred, average="macro", zero_division=0))
            y_va_bin = label_binarize(y_va, classes=CLASS_ORDER)
            auc_scores.append(roc_auc_score(y_va_bin, proba, average="macro", multi_class="ovr"))
        rows.append(
            {
                "Components": a,
                "Balanced accuracy": float(np.mean(bal_scores)),
                "F1 macro": float(np.mean(f1_scores)),
                "AUROC OvR macro": float(np.mean(auc_scores)),
            }
        )
    return pd.DataFrame(rows)


def run_models() -> tuple[ModelOutputs, ModelOutputs, pd.DataFrame, pd.Series, np.ndarray]:
    X_df, y_series = load_data()
    X = X_df.to_numpy(dtype=float)
    y = y_series.to_numpy(dtype=str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pls_cv = select_plsda_components(X_train, y_train, max_components=len(FEATURES))
    best_components = int(pls_cv.sort_values(["Balanced accuracy", "F1 macro"], ascending=False).iloc[0]["Components"])

    pls, pls_scaler = fit_plsda(X_train, y_train, n_components=best_components)
    y_pred_pls, y_proba_pls = predict_plsda(pls, pls_scaler, X_test)
    pls_metrics = compute_metrics(y_test, y_pred_pls, y_proba_pls, CLASS_ORDER)
    pls_output = ModelOutputs("PLS-DA", y_test, y_pred_pls, y_proba_pls, pls_metrics)

    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=24,
        min_samples_split=8,
        min_samples_leaf=4,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = align_proba_to_labels(rf.predict_proba(X_test), rf.classes_, CLASS_ORDER)
    rf_metrics = compute_metrics(y_test, y_pred_rf, y_proba_rf, CLASS_ORDER)
    rf_output = ModelOutputs("Random Forest", y_test, y_pred_rf, y_proba_rf, rf_metrics)

    rf_perm = permutation_importance(
        rf,
        X_test,
        y_test,
        scoring="balanced_accuracy",
        n_repeats=12,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    rf_importance = pd.Series(rf_perm.importances_mean, index=FEATURES).sort_values(ascending=False)
    pls_vip = pd.Series(vip_scores(pls), index=FEATURES).sort_values(ascending=False)

    RESULTS.joinpath("plsda_cv_selection.csv").write_text(pls_cv.to_csv(index=False), encoding="utf-8")
    results_table = pd.DataFrame(
        [
            {"Model": rf_output.name, **rf_output.metrics},
            {"Model": pls_output.name, **pls_output.metrics},
        ]
    )
    results_table.to_csv(RESULTS / "rf_vs_plsda_metrics.csv", index=False)

    importance_table = pd.DataFrame(
        {
            "Feature": FEATURES,
            "RandomForest permutation importance": [rf_importance.get(feature, np.nan) for feature in FEATURES],
            "PLSDA VIP": [pls_vip.get(feature, np.nan) for feature in FEATURES],
        }
    )
    importance_table.to_csv(RESULTS / "rf_vs_plsda_feature_scores.csv", index=False)

    return rf_output, pls_output, pls_cv, rf_importance, pls_vip.to_numpy()


def build_metric_plot(rf_output: ModelOutputs, pls_output: ModelOutputs, pls_cv: pd.DataFrame) -> None:
    metrics_to_plot = ["Accuracy", "Balanced accuracy", "F1 macro", "AUROC OvR macro", "MCC"]
    rf_values = [rf_output.metrics[m] for m in metrics_to_plot]
    pls_values = [pls_output.metrics[m] for m in metrics_to_plot]
    x = np.arange(len(metrics_to_plot))
    width = 0.34

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), facecolor="white")

    axes[0].bar(x - width / 2, rf_values, width, label="Random Forest", color="#245a86")
    axes[0].bar(x + width / 2, pls_values, width, label="PLS-DA", color="#c8744a")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics_to_plot, rotation=18, ha="right")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(axis="y", linestyle="--", alpha=0.25)
    axes[0].set_title("Test-set comparison", fontsize=13, fontweight="bold")
    axes[0].legend(frameon=False)

    axes[1].plot(pls_cv["Components"], pls_cv["Balanced accuracy"], marker="o", linewidth=2.2, color="#245a86", label="Balanced accuracy")
    axes[1].plot(pls_cv["Components"], pls_cv["F1 macro"], marker="o", linewidth=2.2, color="#c8744a", label="F1 macro")
    axes[1].plot(pls_cv["Components"], pls_cv["AUROC OvR macro"], marker="o", linewidth=2.2, color="#5f8f66", label="AUROC OvR macro")
    best_row = pls_cv.sort_values(["Balanced accuracy", "F1 macro"], ascending=False).iloc[0]
    axes[1].axvline(best_row["Components"], color="#7c60b8", linestyle="--", linewidth=1.6, label=f"Selected A = {int(best_row['Components'])}")
    axes[1].set_xticks(pls_cv["Components"])
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(linestyle="--", alpha=0.25)
    axes[1].set_title("PLS-DA component selection", fontsize=13, fontweight="bold")
    axes[1].legend(frameon=False, fontsize=9)

    fig.suptitle("Random Forest vs PLS-DA", fontsize=17, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.84, bottom=0.24, wspace=0.20)
    fig.savefig(FIGURES / "rf_vs_plsda_metrics.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def build_confusion_plot(rf_output: ModelOutputs, pls_output: ModelOutputs) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), facecolor="white")
    for ax, output, cmap in zip(axes, [rf_output, pls_output], ["Blues", "Oranges"]):
        cm = confusion_matrix(output.y_true, output.y_pred, labels=CLASS_ORDER, normalize="true")
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_ORDER)
        disp.plot(ax=ax, cmap=cmap, colorbar=False, values_format=".2f")
        ax.set_title(output.name, fontsize=13, fontweight="bold")
    fig.suptitle("Normalized confusion matrices", fontsize=16, fontweight="bold", y=0.97)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.84, bottom=0.12, wspace=0.24)
    fig.savefig(FIGURES / "rf_vs_plsda_confusion_matrices.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def build_importance_plot(rf_importance: pd.Series, pls_vip: np.ndarray) -> None:
    vip_series = pd.Series(pls_vip, index=FEATURES).sort_values(ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), facecolor="white")

    axes[0].barh(rf_importance.index[::-1], rf_importance.values[::-1], color="#245a86")
    axes[0].set_title("Random Forest permutation importance", fontsize=13, fontweight="bold")
    axes[0].grid(axis="x", linestyle="--", alpha=0.25)

    axes[1].barh(vip_series.index[::-1], vip_series.values[::-1], color="#c8744a")
    axes[1].axvline(1.0, color="#7c60b8", linestyle="--", linewidth=1.4)
    axes[1].set_title("PLS-DA VIP profile", fontsize=13, fontweight="bold")
    axes[1].grid(axis="x", linestyle="--", alpha=0.25)

    fig.suptitle("Variable importance and latent relevance", fontsize=16, fontweight="bold", y=0.97)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.82, bottom=0.12, wspace=0.28)
    fig.savefig(FIGURES / "rf_vs_plsda_importance.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    rf_output, pls_output, pls_cv, rf_importance, pls_vip = run_models()
    build_metric_plot(rf_output, pls_output, pls_cv)
    build_confusion_plot(rf_output, pls_output)
    build_importance_plot(rf_importance, pls_vip)

    summary_lines = [
        "Random Forest vs PLS-DA delivery summary",
        "=======================================",
        f"Random Forest accuracy: {rf_output.metrics['Accuracy']:.4f}",
        f"Random Forest balanced accuracy: {rf_output.metrics['Balanced accuracy']:.4f}",
        f"PLS-DA accuracy: {pls_output.metrics['Accuracy']:.4f}",
        f"PLS-DA balanced accuracy: {pls_output.metrics['Balanced accuracy']:.4f}",
        f"Selected latent components (PLS-DA): {int(pls_cv.sort_values(['Balanced accuracy', 'F1 macro'], ascending=False).iloc[0]['Components'])}",
    ]
    (RESULTS / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    for path in [FIGURES / "rf_vs_plsda_metrics.png", FIGURES / "rf_vs_plsda_confusion_matrices.png", FIGURES / "rf_vs_plsda_importance.png"]:
        print(path)


if __name__ == "__main__":
    main()
