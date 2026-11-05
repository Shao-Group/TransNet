"""Average precision and precision-recall curves against a reference annotation.

A candidate is a true positive when its stranded intron chain equals that of a
reference transcript (outer transcript ends are ignored). Recall is over the
positives present in each GTF's candidate set, not over all reference
transcripts. Two rankings are compared per GTF: the TransNet score and the
assembler's native abundance.
"""

import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]


def evaluate(tables, destination, title):
    """tables: {gtf name: transcript table with label}. Writes evaluation.tsv and pr_curves.pdf."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    for k, (name, t) in enumerate(tables.items()):
        color = COLORS[k % len(COLORS)]
        for column, method, style in (("score", "TransNet", "-"), ("native_abundance", "native abundance", ":")):
            both = t.label.nunique() == 2
            ap = float(average_precision_score(t.label, t[column])) if both else float("nan")
            rows.append(dict(gtf=name, ranking=method, average_precision=ap,
                             candidates=len(t), positives=int(t.label.sum())))
            if both:
                p, r, _ = precision_recall_curve(t.label, t[column])
                ax.plot(r, p, color=color, ls=style, lw=1.6, label=f"{name} {method} (AP {ap:.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", title=title, xlim=(0, 1), ylim=(0, 1.02))
    ax.set_box_aspect(1)
    ax.grid(alpha=0.2)
    ax.legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    fig.savefig(destination / "pr_curves.pdf")
    plt.close(fig)
    frame = pd.DataFrame(rows)
    frame.to_csv(destination / "evaluation.tsv", sep="\t", index=False)
    return frame
