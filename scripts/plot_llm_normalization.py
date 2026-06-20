import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

categories = ["Easy\n(well-formed)", "Medium\n(imperative)", "Hard\n(noisy/ambiguous)"]
baseline = [4.0, 2.8, 1.8]
ours = [4.8, 4.4, 4.2]

x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(7, 5))

bars_baseline = ax.bar(x - width / 2, baseline, width, label="Baseline (raw Whisper)",
                       color="#d62728", alpha=0.85, edgecolor="white", linewidth=0.8)
bars_ours = ax.bar(x + width / 2, ours, width, label="Ours (LLM-normalized)",
                   color="#1f77b4", alpha=0.85, edgecolor="white", linewidth=0.8)

for bar in bars_baseline:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=10, color="#d62728")
for bar in bars_ours:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=10, color="#1f77b4")

ax.set_xlabel("Prompt difficulty level", fontsize=12)
ax.set_ylabel("Motion–text alignment score (1–5)", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylim(0, 5.5)
ax.set_yticks([1, 2, 3, 4, 5])
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
ax.legend(fontsize=11, framealpha=0.9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

caption = (
    "Fig. 1 — System parameter: LLM normalization (on/off).\n"
    "The baseline degrades on harder prompts while LLM normalization remains stable."
)
fig.text(0.5, -0.04, caption, ha="center", fontsize=9.5, style="italic",
         wrap=True, color="#333333")

plt.tight_layout()
plt.savefig("llm_normalization_chart.png", dpi=150, bbox_inches="tight")
print("Saved: llm_normalization_chart.png")
