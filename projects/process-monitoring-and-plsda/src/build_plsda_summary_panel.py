from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"


PANELS = [
    ("01_cv_metricas_vs_A.png", "Cross-validation", "Model selection across latent components."),
    ("11_matriz_confusion_normalizada.png", "Confusion matrix", "Class-oriented evaluation in the final PLS-DA setting."),
    ("12_vip.png", "VIP profile", "Variable-importance summary for the discriminant model."),
    ("13_permutation_test_mcc.png", "Permutation test", "Stability check against random label structure."),
]


def load_image(name: str):
    return mpimg.imread(FIGURES / name)


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14.8, 11.2), facecolor="white")
    axes = axes.ravel()

    for ax, (filename, title, subtitle) in zip(axes, PANELS):
        img = load_image(filename)
        ax.imshow(img)
        ax.set_title(title, fontsize=14.5, fontweight="bold", pad=10)
        ax.text(
            0.5,
            -0.07,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9.8,
            color="#555555",
        )
        ax.axis("off")

    fig.suptitle("PLS-DA monitoring summary", fontsize=22, fontweight="bold", y=0.98)
    fig.text(
        0.5,
        0.02,
        "Representative portfolio view combining model selection, class-level behaviour, variable importance and validation.",
        ha="center",
        fontsize=10.5,
        color="#555555",
    )
    fig.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.08, hspace=0.20, wspace=0.08)
    out_path = FIGURES / "plsda_summary_panel.png"
    fig.savefig(out_path, dpi=320, bbox_inches="tight")
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()
