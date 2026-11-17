import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from scipy.special import softmax

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import data  # noqa: E402
import metrics  # noqa: E402

OUT = ROOT / "presentation"
OUT.mkdir(exist_ok=True)
LOGITS = ROOT / "results" / "extra_logits"
RES = pd.read_csv(ROOT / "results" / "results.csv")
COST = pd.read_csv(ROOT / "results" / "cost.csv")
REL = pd.read_csv(ROOT / "results" / "reliability.csv")

INK, MUTED, GRID = "#1f2937", "#64748b", "#e2e8f0"
COLOR = {"rf": "#8a8f98", "cnn": "#2a78d6", "resnet": "#eb6834", "wav2vec": "#1baf7a"}
NAME = {"rf": "Random Forest", "cnn": "CNN", "resnet": "ResNet-18", "wav2vec": "Wav2Vec2"}
DSET = {"esc10": "ESC-10 (nature sounds)", "sc": "Speech Commands (full)", "sc_small": "Speech Commands, 320 clips"}
SIZE = (11, 6.2)

plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def load(tag, dataset, model, seed=0):
    files = sorted(LOGITS.glob(f"{tag}_{dataset}_{model}_{seed}_f*.npz"))
    parts = [np.load(f) for f in files]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}


def calib(logits, y, t=1.0):
    p = softmax(logits / t, axis=1)
    acc = float((p.argmax(1) == y).mean())
    return {"acc": acc, "ece": float(metrics.ece(p, y)), "overconf": float(p.max(1).mean() - acc)}


def res(dataset, model, condition, col):
    r = RES[(RES.dataset == dataset) & (RES.model == model) & (RES.condition == condition)]
    return float(r[col].iloc[0])


# ------------------------------------------------------------ slide 2
def slide_02():
    esc = data.esc10_table()
    dog = data.fix_length(data.load_wav(esc[esc.category == "dog"].path.iloc[3]), 3 * data.SR)
    yes = data.fix_length(data.load_wav(sorted((data.SC_ROOT / "yes").glob("*.wav"))[10]), data.SR)
    import torch
    logmel = data.LogMel()
    fig, axes = plt.subplots(2, 2, figsize=SIZE, gridspec_kw={"height_ratios": [1, 1.5]})
    for col, (x, label) in enumerate([(dog, "A dog barking (ESC-10), 3 s"), (yes, 'The word "yes" (Speech Commands), 1 s')]):
        t = np.arange(len(x)) / data.SR
        axes[0, col].plot(t, x, color=MUTED, lw=0.6)
        axes[0, col].set_title(label)
        axes[0, col].set_xlim(0, t[-1]); axes[0, col].set_yticks([]); axes[0, col].grid(False)
        axes[0, col].set_ylabel("loudness" if col == 0 else "")
        spec = logmel(torch.from_numpy(x)[None])[0, 0].numpy()
        axes[1, col].imshow(spec, origin="lower", aspect="auto", cmap="magma", extent=[0, t[-1], 0, 8])
        axes[1, col].set_xlabel("time (s)"); axes[1, col].grid(False)
        axes[1, col].set_ylabel("pitch (low to high)" if col == 0 else ""); axes[1, col].set_yticks([])
    fig.text(0.5, 1.0, "Top: the raw sound wave. Bottom: the same sound as a picture (spectrogram)", ha="center", fontsize=15, fontweight="bold")
    fig.tight_layout()
    save(fig, "slide_02_sound_as_picture.png")


# ------------------------------------------------------------ slide 3
def box(ax, x, y, w, h, text, face, edge, size=13, color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=face, ec=edge, lw=1.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=color, fontweight="bold")


def arrow(ax, x0, x1, y):
    ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=2))


def slide_03():
    fig = plt.figure(figsize=SIZE)
    ax = fig.add_axes([0, 0.45, 1, 0.55]); ax.set_xlim(0, 12); ax.set_ylim(0, 4.6); ax.axis("off")
    ax.text(0, 4.4, "Two ways to feed sound to a network", fontsize=15, fontweight="bold", va="center")
    box(ax, 0.0, 2.55, 1.9, 0.9, "sound wave", "#f1f5f9", MUTED)
    box(ax, 2.7, 2.55, 2.2, 0.9, "spectrogram\n(a picture)", "#f1f5f9", MUTED)
    box(ax, 5.7, 3.1, 2.2, 0.8, "CNN\n(small, from scratch)", "#dbeafe", COLOR["cnn"], 11)
    box(ax, 5.7, 1.95, 2.2, 0.8, "ResNet-18\n(pre-trained on photos)", "#ffedd5", COLOR["resnet"], 11)
    box(ax, 9.3, 2.55, 2.3, 0.9, "answer + confidence", "#f1f5f9", MUTED)
    arrow(ax, 1.9, 2.7, 3.0); arrow(ax, 4.9, 5.7, 3.5); arrow(ax, 4.9, 5.7, 2.35); arrow(ax, 7.9, 9.3, 3.5); arrow(ax, 7.9, 9.3, 2.35)
    box(ax, 0.0, 0.7, 1.9, 0.9, "sound wave", "#f1f5f9", MUTED)
    box(ax, 4.0, 0.55, 3.9, 1.2, "Wav2Vec2\n(huge, pre-trained on speech)", "#d1fae5", COLOR["wav2vec"], 11)
    box(ax, 9.3, 0.7, 2.3, 0.9, "answer + confidence", "#f1f5f9", MUTED)
    arrow(ax, 1.9, 4.0, 1.15); arrow(ax, 7.9, 9.3, 1.15)
    ax.text(0.0, 0.05, "Random Forest (baseline): hand-made sound features, no neural network", fontsize=12, color=MUTED)
    bx = fig.add_axes([0.2, 0.06, 0.62, 0.3])
    models = ["cnn", "resnet", "wav2vec"]
    params = [0.24, 11.2, 94.4]
    bx.barh([NAME[m] for m in models][::-1], params[::-1], color=[COLOR[m] for m in models][::-1], height=0.55)
    bx.set_xscale("log"); bx.set_xlim(0.1, 600); bx.set_xlabel("size of the model (million parameters, log scale)")
    for i, (m, p) in enumerate(zip(models[::-1], params[::-1])):
        bx.text(p * 1.15, i, f"{p:g} M", va="center", fontsize=13, fontweight="bold")
    bx.grid(axis="y", visible=False)
    save(fig, "slide_03_models.png")


# ------------------------------------------------------------ slide 4
def slide_04a():
    yes = data.fix_length(data.load_wav(sorted((data.SC_ROOT / "yes").glob("*.wav"))[10]), data.SR)
    X = yes[None]
    fig, axes = plt.subplots(4, 1, figsize=SIZE, sharex=True, sharey=True)
    items = [(X, "no noise"), (data.add_noise(X, "white", 20), "SNR 20 dB: weak noise"), (data.add_noise(X, "white", 10), "SNR 10 dB: medium noise"), (data.add_noise(X, "white", 0), "SNR 0 dB: noise as loud as the sound")]
    t = np.arange(len(yes)) / data.SR
    for ax, (x, label), c in zip(axes, items, [COLOR["wav2vec"], MUTED, MUTED, COLOR["resnet"]]):
        ax.plot(t, x[0], color=c, lw=0.6); ax.set_yticks([]); ax.grid(False)
        ax.text(1.0, 0.82, label, transform=ax.transAxes, ha="right", fontsize=14, fontweight="bold", color=INK)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle('The same word "yes" with more and more noise added', fontsize=15, fontweight="bold")
    fig.tight_layout()
    save(fig, "slide_04a_noise_levels.png")


def slide_04b():
    fig, axes = plt.subplots(1, 2, figsize=SIZE, sharey=True)
    for ax, cond, label in zip(axes, ["clean", "white_0"], ["No noise", "Strong noise (0 dB)"]):
        part = REL[(REL.dataset == "esc10") & (REL.model == "resnet") & (REL.condition == cond)].sort_values("bin")
        keep = part[part.weight > 0.02]
        centers = (keep.bin + 0.5) / 10
        ax.bar(centers, keep.accuracy, width=0.085, color=COLOR["resnet"], label="how often it is right")
        ax.plot([0, 1], [0, 1], "--", color=INK, lw=1.5, label="honest model")
        ax.set_title(label); ax.set_xlabel("how sure the model says it is"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0%", "50%", "100%"]); ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(["0%", "50%", "100%"])
    axes[0].set_ylabel("how often it is actually right"); axes[0].legend(loc="upper left", fontsize=12, frameon=False)
    axes[1].text(0.74, 0.42, "bars far below the\ndashed line:\nsays 90% sure,\nis right ~15%", ha="center", fontsize=13, color=INK, fontweight="bold")
    fig.suptitle("ResNet-18 on ESC-10: when is its confidence honest?", fontsize=15, fontweight="bold")
    fig.tight_layout()
    save(fig, "slide_04b_honest_confidence.png")


# ------------------------------------------------------------ slide 5
def slide_05():
    fig, axes = plt.subplots(1, 2, figsize=SIZE, sharey=False)
    offsets = {
        "esc10": {"rf": (1.0, -0.016, "center"), "cnn": (1.0, -0.017, "center"), "resnet": (1.0, 0.012, "center"), "wav2vec": (1.0, -0.017, "center")},
        "sc": {"rf": (1.0, -0.035, "center"), "cnn": (0.8, -0.035, "right"), "resnet": (1.0, 0.02, "center"), "wav2vec": (1.0, 0.02, "center")},
    }
    for ax, dataset in zip(axes, ["esc10", "sc"]):
        for m in ["rf", "cnn", "resnet", "wav2vec"]:
            c = COST[(COST.dataset == dataset) & (COST.model == m)]
            secs, acc = max(float(c.train_seconds.mean()), 0.3), res(dataset, m, "clean", "acc")
            ax.scatter(secs, acc, s=190, color=COLOR[m], zorder=3, edgecolor="white", linewidth=2)
            fx, dy, ha = offsets[dataset][m]
            ax.text(secs * fx, acc + dy, NAME[m], fontsize=13, fontweight="bold", ha=ha)
        ax.set_xscale("log"); ax.set_xlabel("training time (seconds, log scale)"); ax.set_title(DSET[dataset])
        ax.set_xlim(0.25, 3e4)
    axes[0].set_ylabel("accuracy on clean sound"); axes[0].set_ylim(0.74, 0.9); axes[1].set_ylim(0.58, 1.05)
    fig.suptitle("Is the huge model worth its price?", fontsize=15, fontweight="bold")
    fig.tight_layout()
    save(fig, "slide_05_accuracy_vs_cost.png")


# ------------------------------------------------------------ slide 6
def slide_06():
    fig, axes = plt.subplots(1, 2, figsize=SIZE, sharey=True)
    for ax, dataset, models in [(axes[0], "esc10", ["rf", "cnn", "resnet"]), (axes[1], "sc_small", ["rf", "cnn", "resnet", "wav2vec"])]:
        for i, m in enumerate(models):
            vals = []
            for seed in [0, 1, 2]:
                d = load("base", dataset, m, seed)
                vals.append(calib(d["test_clean"], d["y_test"])["acc"])
            ax.scatter([i] * 3 + np.array([-0.08, 0, 0.08]), vals, s=120, color=COLOR[m], edgecolor="white", linewidth=1.5, zorder=3)
            ax.hlines(np.mean(vals), i - 0.25, i + 0.25, color=INK, lw=2)
            if min(vals) < 0.3:
                ax.annotate("one run\nfailed", xy=(i + 0.08, min(vals)), xytext=(i - 0.9, 0.28), fontsize=13, fontweight="bold", color=INK, arrowprops=dict(arrowstyle="->", color=INK))
        ax.set_xticks(range(len(models))); ax.set_xticklabels([NAME[m] for m in models], fontsize=12); ax.set_title(DSET[dataset]); ax.set_xlim(-0.6, len(models) - 0.4)
    axes[0].set_ylabel("accuracy on clean sound"); axes[0].set_ylim(0, 1)
    axes[0].text(0.5, 0.06, "Wav2Vec2 was run once here: 0.82 and 0.85 in two runs", transform=axes[0].transAxes, ha="center", fontsize=11, color=MUTED)
    fig.suptitle("Same experiment repeated 3 times (each dot = one run, line = average)", fontsize=15, fontweight="bold")
    fig.tight_layout()
    save(fig, "slide_06_seeds.png")


# ------------------------------------------------------------ slide 7
def slide_07():
    steps_per_epoch = {"cnn": 10, "resnet": 10, "wav2vec": 20}
    grid = {"cnn": [20, 60, 150], "resnet": [15, 60, 150], "wav2vec": [10, 30]}
    fig, ax = plt.subplots(figsize=SIZE)
    for m, epochs in grid.items():
        xs, ys = [], []
        for e in epochs:
            accs = []
            for seed in [0, 1]:
                d = load(f"ep{e}", "sc_small", m, seed)
                accs.append(calib(d["test_clean"], d["y_test"])["acc"])
            xs.append(e * steps_per_epoch[m]); ys.append(np.mean(accs))
        ax.plot(xs, ys, "-o", color=COLOR[m], lw=3, ms=10)
        ax.scatter(xs[0], ys[0], s=330, facecolor="none", edgecolor=INK, linewidth=2, zorder=4)
        ax.text(xs[-1] * 1.07, ys[-1], f"{NAME[m]}\n{ys[-1]:.2f}", color=INK, fontsize=14, fontweight="bold", va="center")
    ax.set_xscale("log"); ax.set_xlim(120, 4000); ax.set_ylim(0.4, 1.0)
    ax.minorticks_off(); ax.set_xticks([200, 600, 1500]); ax.set_xticklabels(["200", "600", "1500"])
    ax.set_xlabel("training steps (log scale)"); ax.set_ylabel("accuracy on clean sound")
    ax.set_title("Only 320 training clips: what if the small models train much longer?\n(circled = the amount of training used in the first run)", loc="left")
    save(fig, "slide_07_undertraining.png")


# ------------------------------------------------------------ slide 8
def slide_08():
    fig, axes = plt.subplots(1, 2, figsize=SIZE, sharey=True)
    for ax, dataset in zip(axes, ["esc10", "sc"]):
        for m in ["rf", "cnn", "resnet", "wav2vec"]:
            y = [res(dataset, m, c, "overconf") for c in ["clean", "white_20", "white_10", "white_0"]]
            ax.plot(range(4), y, "-o", color=COLOR[m], lw=3, ms=8, label=NAME[m])
        ax.axhline(0, color=INK, lw=1)
        ax.set_xticks(range(4)); ax.set_xticklabels(["clean", "20 dB", "10 dB", "0 dB"]); ax.set_title(DSET[dataset]); ax.set_xlabel("more noise to the right")
    axes[0].set_ylabel("overconfidence\n(confidence minus accuracy)"); axes[0].legend(fontsize=12, frameon=False, loc="upper left")
    fig.suptitle("Do models stay honest when noise grows? (white noise)", fontsize=15, fontweight="bold")
    fig.tight_layout()
    save(fig, "slide_08_overconfidence_noise.png")


# ------------------------------------------------------------ slide 9
def slide_09():
    fig, ax = plt.subplots(figsize=SIZE)
    dc = {"esc10": "#e34948", "sc": "#2a78d6", "sc_small": "#1baf7a"}
    mk = {"cnn": "o", "resnet": "s", "wav2vec": "^"}
    for (dataset, model), part in RES[RES.model.isin(mk)].groupby(["dataset", "model"]):
        ax.scatter(part.acc, part.overconf, color=dc[dataset], marker=mk[model], s=110, alpha=0.9, edgecolor="white", linewidth=1.2)
    ax.axhline(0, color=INK, lw=1)
    for d, lab in DSET.items():
        ax.scatter([], [], color=dc[d], s=110, label=lab)
    for m, mm in mk.items():
        ax.scatter([], [], color=MUTED, marker=mm, s=90, label=NAME[m])
    ax.legend(fontsize=12, frameon=False, loc="upper right")
    ax.set_xlabel("accuracy"); ax.set_ylabel("overconfidence (confidence minus accuracy)")
    ax.set_title("At the same accuracy, is nature sound (ESC-10) more overconfident than speech?", loc="left")
    save(fig, "slide_09_same_accuracy.png")


# ------------------------------------------------------------ slide 10
def slide_10():
    kinds = [("white_0", "white\nnoise"), ("real_0", "speech-data\nbackground"), ("esc_0", "other nature\nsounds")]
    fig, axes = plt.subplots(1, 2, figsize=SIZE)
    for ax, key, title in [(axes[0], "acc", "accuracy"), (axes[1], "overconf", "overconfidence")]:
        for j, m in enumerate(["cnn", "resnet", "wav2vec"]):
            d = load("base", "esc10", m)
            vals = [calib(d[f"test_{c}"], d["y_test"])[key] for c, _ in kinds]
            ax.bar(np.arange(3) + (j - 1) * 0.26, vals, width=0.24, color=COLOR[m], label=NAME[m])
        ax.set_xticks(range(3)); ax.set_xticklabels([k for _, k in kinds], fontsize=12); ax.set_title(title + ", noise at 0 dB")
        ax.axhline(0, color=INK, lw=1); ax.grid(axis="x", visible=False)
    axes[0].legend(fontsize=12, frameon=False)
    fig.suptitle("ESC-10 under the same strength of noise, but three different kinds", fontsize=15, fontweight="bold")
    fig.tight_layout()
    save(fig, "slide_10_noise_source.png")


# ------------------------------------------------------------ slide 11
SEEN = ["white_20", "white_10", "white_0", "real_20", "real_10", "real_0"]
UNSEEN = ["esc_20", "esc_10", "esc_0"]
ALLC = ["clean"] + SEEN + UNSEEN


def temperature_summary():
    out = {}
    for dataset, models in [("esc10", ["rf", "cnn", "resnet", "wav2vec"]), ("sc", ["rf", "cnn", "resnet"])]:
        acc = {g: {k: [] for k in ["none", "clean", "pooled", "oracle"]} for g in ["clean", "seen", "unseen"]}
        for m in models:
            d = load("base", dataset, m)
            names = ["clean"] + SEEN
            pooled = np.concatenate([d[f"val_{c}"] for c in names]); y = np.tile(d["y_val"], len(names))
            t = {"clean": metrics.fit_temperature(d["val_clean"], d["y_val"]), "pooled": metrics.fit_temperature(pooled, y)}
            for c in ALLC:
                t["oracle"] = metrics.fit_temperature(d[f"val_{c}"], d["y_val"])
                g = "clean" if c == "clean" else ("unseen" if c.startswith("esc") else "seen")
                for k, temp in [("none", 1.0), ("clean", t["clean"]), ("pooled", t["pooled"]), ("oracle", t["oracle"])]:
                    acc[g][k].append(calib(d[f"test_{c}"], d["y_test"], temp)["ece"])
        out[dataset] = {g: {k: float(np.mean(v)) for k, v in kk.items()} for g, kk in acc.items()}
    return out


def slide_11():
    s = temperature_summary()
    print("temperature summary", {k: {g: {m: round(v, 3) for m, v in x.items()} for g, x in d.items()} for k, d in s.items()})
    cols = {"none": "#b6bcc6", "clean": "#2a78d6", "pooled": "#eb6834", "oracle": "#1baf7a"}
    labels = {"none": "no fix", "clean": "fitted on clean sound", "pooled": "fitted on all noise levels (one number)", "oracle": "knows the noise (best case)"}
    fig, axes = plt.subplots(1, 2, figsize=SIZE, sharey=True)
    for ax, dataset in zip(axes, ["esc10", "sc"]):
        for j, k in enumerate(["none", "clean", "pooled", "oracle"]):
            ax.bar(np.arange(3) + (j - 1.5) * 0.2, [s[dataset][g][k] for g in ["clean", "seen", "unseen"]], width=0.19, color=cols[k], label=labels[k])
        ax.set_xticks(range(3)); ax.set_xticklabels(["clean\nsound", "noise seen\nwhen fitting", "new kind\nof noise"], fontsize=12)
        ax.set_title(DSET[dataset]); ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("dishonesty (ECE, lower = better)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=12)
    fig.suptitle("Can one simple correction (temperature scaling) fix overconfidence?", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0.1, 1, 0.97])
    save(fig, "slide_11_temperature.png")


# ------------------------------------------------------------ slide 12
def group_means(tag, dataset, model):
    d = load(tag, dataset, model)
    g = {"clean": ["clean"], "seen": SEEN, "unseen": UNSEEN}
    return {k: {m: float(np.mean([calib(d[f"test_{c}"], d["y_test"])[m] for c in cs])) for m in ["acc", "ece"]} for k, cs in g.items()}


def slide_12():
    models = ["cnn", "resnet", "wav2vec"]
    before = {m: group_means("base", "esc10", m) for m in models}
    after = {m: group_means("aug", "esc10", m) for m in models}
    fig, axes = plt.subplots(1, 3, figsize=SIZE)
    panels = [("clean", "acc", "accuracy on clean sound"), ("seen", "ece", "dishonesty on noise\nseen in training (ECE)"), ("unseen", "ece", "dishonesty on a NEW\nkind of noise (ECE)")]
    for ax, (grp, key, title) in zip(axes, panels):
        for j, m in enumerate(models):
            ax.bar(j - 0.21, before[m][grp][key], width=0.4, color=COLOR[m], alpha=0.4)
            ax.bar(j + 0.21, after[m][grp][key], width=0.4, color=COLOR[m])
        ax.set_xticks(range(3)); ax.set_xticklabels([NAME[m] for m in models], fontsize=11, rotation=0); ax.set_title(title, fontsize=13); ax.grid(axis="x", visible=False)
    axes[0].set_ylim(0, 1)
    fig.suptitle("ESC-10: pale bar = trained on clean sound, solid bar = trained with noise", fontsize=15, fontweight="bold")
    fig.tight_layout()
    save(fig, "slide_12_noise_training.png")

    fig, ax = plt.subplots(figsize=(8, 5.2))
    for j, m in enumerate(["cnn", "resnet", "wav2vec"]):
        b = group_means("base", "sc_small", m)["clean"]["acc"]; a = group_means("aug", "sc_small", m)["clean"]["acc"]
        ax.bar(j - 0.21, b, width=0.4, color=COLOR[m], alpha=0.4); ax.bar(j + 0.21, a, width=0.4, color=COLOR[m])
        ax.text(j + 0.21, a + 0.02, f"{a:.2f}", ha="center", fontsize=13, fontweight="bold"); ax.text(j - 0.21, b + 0.02, f"{b:.2f}", ha="center", fontsize=13, alpha=0.7)
    ax.set_xticks(range(3)); ax.set_xticklabels([NAME[m] for m in ["cnn", "resnet", "wav2vec"]]); ax.set_ylim(0, 1.05); ax.grid(axis="x", visible=False)
    ax.set_ylabel("accuracy on clean sound")
    ax.set_title("With only 320 clips, training with noise hurts\n(pale = clean training, solid = noise training)", loc="left")
    save(fig, "slide_12b_small_data_warning.png")


if __name__ == "__main__":
    for f in [slide_02, slide_03, slide_04a, slide_04b, slide_05, slide_06, slide_07, slide_08, slide_09, slide_10, slide_11, slide_12]:
        f()
