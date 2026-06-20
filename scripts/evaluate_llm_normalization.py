"""
Evaluate LLM prompt normalization via semantic similarity (cosine similarity
between sentence embeddings of raw Whisper transcripts vs. normalized prompts).
"""

from sentence_transformers import SentenceTransformer, util
import numpy as np
import matplotlib.pyplot as plt

PAIRS = [
    # (difficulty, raw_input, normalized_prompt)
    ("Easy",   "a person doing a squat",                          "a person doing a squat"),
    ("Easy",   "generate a person jumping",                       "a person jumping"),
    ("Easy",   "show me someone running",                         "a person running"),
    ("Medium", "show me a young man do a squat",                  "a person doing a squat"),
    ("Medium", "can you do a dancing?",                           "a person dancing"),
    ("Medium", "make the avatar walk forward please",             "a person walking forward"),
    ("Hard",   "uh, like someone kind of doing a happy dance",    "a person doing a happy dance"),
    ("Hard",   "like, you know, someone stretching their arms",   "a person stretching their arms"),
    ("Hard",   "hmm maybe show a person like waving or something","a person waving"),
    ("Hard",   "do that thing where someone kicks",               "a person kicking"),
]

model = SentenceTransformer("all-MiniLM-L6-v2")

raws        = [p[1] for p in PAIRS]
normalized  = [p[2] for p in PAIRS]
difficulties= [p[0] for p in PAIRS]

emb_raw  = model.encode(raws,       convert_to_tensor=True)
emb_norm = model.encode(normalized, convert_to_tensor=True)

scores = util.cos_sim(emb_raw, emb_norm).diagonal().cpu().numpy()

print(f"\n{'Difficulty':<8}  {'Score':>6}  {'Raw Input':<50}  {'Normalized'}")
print("-" * 110)
for (diff, raw, norm), score in zip(PAIRS, scores):
    print(f"{diff:<8}  {score:>6.4f}  {raw:<50}  {norm}")

for level in ["Easy", "Medium", "Hard"]:
    mask = [d == level for d in difficulties]
    avg  = np.mean(scores[mask])
    print(f"\n{level} avg: {avg:.4f}")

print(f"\nOverall avg: {np.mean(scores):.4f}")

# --- plot ---
colors = {"Easy": "#2ca02c", "Medium": "#ff7f0e", "Hard": "#d62728"}
levels = ["Easy", "Medium", "Hard"]

# per-level scores and avg
level_scores = {l: scores[[d == l for d in difficulties]] for l in levels}
level_avgs   = {l: np.mean(v) for l, v in level_scores.items()}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# --- left: per-prompt bars ---
bar_colors = [colors[d] for d in difficulties]
x = np.arange(len(PAIRS))
bars = ax1.bar(x, scores, color=bar_colors, alpha=0.85, edgecolor="white", linewidth=0.8)
for bar, score in zip(bars, scores):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f"{score:.2f}", ha="center", va="bottom", fontsize=8)
ax1.axhline(np.mean(scores), color="black", linestyle="--", linewidth=1,
            label=f"Overall mean = {np.mean(scores):.3f}")
ax1.set_xticks(x)
ax1.set_xticklabels([f"P{i+1}" for i in range(len(PAIRS))], fontsize=9)
ax1.set_ylim(0, 1.15)
ax1.set_ylabel("Cosine Similarity", fontsize=11)
ax1.set_xlabel("Prompt index", fontsize=11)
ax1.set_title("Per-prompt semantic similarity", fontsize=11)
ax1.yaxis.grid(True, linestyle="--", alpha=0.4)
ax1.set_axisbelow(True)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
patches = [plt.Rectangle((0,0),1,1, color=colors[l], alpha=0.85) for l in levels]
ax1.legend(patches + [plt.Line2D([0],[0], color="black", linestyle="--")],
           levels + [f"Mean={np.mean(scores):.3f}"], fontsize=9, title="Difficulty")

# --- right: grouped box/bar by difficulty ---
bar_x = np.arange(len(levels))
avgs  = [level_avgs[l] for l in levels]
bcolors = [colors[l] for l in levels]
b2 = ax2.bar(bar_x, avgs, color=bcolors, alpha=0.85, edgecolor="white", linewidth=0.8, width=0.5)
for bar, avg, l in zip(b2, avgs, levels):
    # scatter individual points
    xs = np.random.normal(bar.get_x() + bar.get_width()/2, 0.04, size=len(level_scores[l]))
    ax2.scatter(xs, level_scores[l], color="white", edgecolors=colors[l],
                zorder=5, s=40, linewidth=1.2)
    ax2.text(bar.get_x() + bar.get_width()/2, avg + 0.015,
             f"{avg:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax2.axhline(np.mean(scores), color="black", linestyle="--", linewidth=1,
            label=f"Overall mean = {np.mean(scores):.3f}")
ax2.set_xticks(bar_x)
ax2.set_xticklabels(levels, fontsize=11)
ax2.set_ylim(0, 1.15)
ax2.set_ylabel("Cosine Similarity", fontsize=11)
ax2.set_xlabel("Difficulty level", fontsize=11)
ax2.set_title("Average similarity by difficulty", fontsize=11)
ax2.yaxis.grid(True, linestyle="--", alpha=0.4)
ax2.set_axisbelow(True)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.legend(fontsize=9)

plt.suptitle("LLM Prompt Normalization — Semantic Similarity Evaluation", fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig("llm_normalization_similarity.png", dpi=150, bbox_inches="tight")
print("\nSaved: llm_normalization_similarity.png")
