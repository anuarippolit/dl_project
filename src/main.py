import argparse
import time

import numpy as np
import pandas as pd
from scipy.special import softmax

import metrics
from data import CONDITIONS, ROOT, apply_condition, get_split
from models import predict, train_model

RESULTS = ROOT / "results"
DATASETS = ["esc10", "sc", "sc_small"]
MODELS = ["rf", "cnn", "resnet", "wav2vec"]
SEEDS = [0]
RELIABILITY_CONDITIONS = ["clean", "white_0", "real_0"]


def latency_ms(model, X, runs=20):
    predict(model, X[:1], cpu=True)
    start = time.time()
    for _ in range(runs):
        predict(model, X[:1], cpu=True)
    return (time.time() - start) / runs * 1000


def run_experiment(dataset, model_name, seed, epochs):
    folds = [1, 2, 3, 4, 5] if dataset == "esc10" else [1]
    logits = {c: [] for c in CONDITIONS + ["val"]}
    y_test, y_val, cost = [], [], []

    for fold in folds:
        print(f"  fold {fold}", flush=True)
        train, val, test = get_split(dataset, fold, seed)
        model, info = train_model(model_name, train, val, dataset, seed, epochs)
        for condition in CONDITIONS:
            logits[condition].append(predict(model, apply_condition(test[0], condition)))
        logits["val"].append(predict(model, val[0]))
        y_test.append(test[1])
        y_val.append(val[1])
        info["latency_ms"] = latency_ms(model, test[0])
        info.update({"dataset": dataset, "model": model_name, "seed": seed, "fold": fold})
        cost.append(info)

    logits = {c: np.concatenate(v) for c, v in logits.items()}
    y_test, y_val = np.concatenate(y_test), np.concatenate(y_val)

    temperature = metrics.fit_temperature(logits["val"], y_val)
    scores, bins = [], []
    for condition in CONDITIONS:
        row = {"dataset": dataset, "model": model_name, "seed": seed, "condition": condition}
        score = dict(row)
        score.update(metrics.evaluate(logits[condition], y_test, temperature))
        scores.append(score)
        if condition in RELIABILITY_CONDITIONS:
            table = metrics.reliability(softmax(logits[condition], axis=1), y_test)
            for i, (weight, conf, acc) in enumerate(table):
                bin_row = dict(row)
                bin_row.update({"bin": i, "weight": weight, "confidence": conf, "accuracy": acc})
                bins.append(bin_row)
    print(f"  clean acc {scores[0]['acc']:.3f}", flush=True)
    return scores, bins, cost


def read_csv(name):
    path = RESULTS / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--epochs", type=int, default=None, help="override the epochs in models.py (for quick tests)")
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    scores, bins, cost = read_csv("results.csv"), read_csv("reliability.csv"), read_csv("cost.csv")

    for dataset in args.datasets:
        for model in args.models:
            for seed in SEEDS:
                done = not scores.empty and ((scores.dataset == dataset) & (scores.model == model) & (scores.seed == seed)).any()
                if done:
                    print("skip", dataset, model, seed)
                    continue
                print(dataset, model, "seed", seed, flush=True)
                new_scores, new_bins, new_cost = run_experiment(dataset, model, seed, args.epochs)
                scores = pd.concat([scores, pd.DataFrame(new_scores)], ignore_index=True)
                bins = pd.concat([bins, pd.DataFrame(new_bins)], ignore_index=True)
                cost = pd.concat([cost, pd.DataFrame(new_cost)], ignore_index=True)
                scores.to_csv(RESULTS / "results.csv", index=False)
                bins.to_csv(RESULTS / "reliability.csv", index=False)
                cost.to_csv(RESULTS / "cost.csv", index=False)


if __name__ == "__main__":
    main()
