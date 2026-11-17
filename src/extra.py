import os

import numpy as np
from scipy.special import softmax

import metrics
from data import CONDITIONS, EXTRA_CONDITIONS, ROOT, apply_condition, esc_noise_bank, get_split
from models import train_model

QUICK = bool(os.environ.get("EXTRA_QUICK"))
LOGITS = ROOT / "results" / ("extra_logits_quick" if QUICK else "extra_logits")
FIGURES = ROOT / "results" / ("figures_quick" if QUICK else "figures")
MODELS = ["rf", "cnn", "resnet", "wav2vec"]
ALL_CONDITIONS = CONDITIONS + EXTRA_CONDITIONS


# train on one fold and save logits for every noise condition
def fold_logits(dataset, model, seed, fold, epochs, augment, tag):
    path = LOGITS / f"{tag}_{dataset}_{model}_{seed}_f{fold}.npz"
    if path.exists():
        return dict(np.load(path))
    LOGITS.mkdir(parents=True, exist_ok=True)
    train, val, test = get_split(dataset, fold, seed)
    if QUICK:
        epochs = 1
        val, test = (val[0][::20], val[1][::20]), (test[0][::20], test[1][::20])
    predict, _ = train_model(model, train, val, dataset, seed, epochs, augment)
    out = {"y_test": test[1], "y_val": val[1]}
    for condition in ALL_CONDITIONS:
        out[f"test_{condition}"] = predict(apply_condition(test[0], condition))
        out[f"val_{condition}"] = predict(apply_condition(val[0], condition))
    np.savez_compressed(path, **out)
    return out


# logits of one model on one dataset, trained once and cached on disk
def collect_logits(dataset, model, seed=0, epochs=None, augment=None, tag="base"):
    esc_noise_bank()
    folds = [1] if QUICK or dataset != "esc10" else [1, 2, 3, 4, 5]
    parts = [fold_logits(dataset, model, seed, fold, epochs, augment, tag) for fold in folds]
    return {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}


# accuracy, ece and confidence minus accuracy, optionally with a temperature
def calib(logits, y, temperature=1.0):
    probs = softmax(logits / temperature, axis=1)
    acc = metrics.accuracy(probs, y)
    return {"acc": acc, "ece": metrics.ece(probs, y), "overconf": probs.max(1).mean() - acc}


# metrics for every condition of one logits file
def calib_table(data, temperature=1.0):
    return {c: calib(data[f"test_{c}"], data["y_test"], temperature) for c in ALL_CONDITIONS}


# save a figure into results/figures
def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=150, bbox_inches="tight")
