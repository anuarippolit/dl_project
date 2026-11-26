import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import soxr
import torch
import torchaudio

SR = 16_000
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = DATA / "cache"
ESC_ROOT = DATA / "ESC-50-master"
SC_ROOT = DATA / "speech_commands_v0.02"
SC_CLASSES = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
SECONDS = {"esc10": 5, "sc": 1}

SNRS = [20, 10, 0]
CONDITIONS = ["clean"] + [f"{kind}_{snr}" for kind in ("white", "real") for snr in SNRS]
EXTRA_CONDITIONS = [f"esc_{snr}" for snr in SNRS]
ALL_CONDITIONS = CONDITIONS + EXTRA_CONDITIONS
REAL_NOISE = ["doing_the_dishes", "dude_miaowing", "exercise_bike", "running_tap"]
ESC_NOISE_CLIPS = 300


def load_wav(path):
    x, sr = sf.read(str(path), dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    if sr != SR:
        x = soxr.resample(x, sr, SR)
    return x


def fix_length(x, n):
    if len(x) < n:
        return np.pad(x, (0, n - len(x)))
    return x[:n]


def esc10_table():
    df = pd.read_csv(ESC_ROOT / "meta" / "esc50.csv")
    df = df[df.esc10].copy()
    classes = sorted(df.category.unique())
    df["label"] = df.category.map({c: i for i, c in enumerate(classes)})
    df["path"] = df.filename.map(lambda f: ESC_ROOT / "audio" / f)
    df["group"] = df.fold
    return df[["path", "label", "group", "category"]].reset_index(drop=True)


def sc_table():
    val = set((SC_ROOT / "validation_list.txt").read_text().split())
    test = set((SC_ROOT / "testing_list.txt").read_text().split())
    rows = []
    for label, word in enumerate(SC_CLASSES):
        for path in sorted((SC_ROOT / word).glob("*.wav")):
            name = f"{word}/{path.name}"
            group = "val" if name in val else "test" if name in test else "train"
            rows.append((path, label, group, word))
    return pd.DataFrame(rows, columns=["path", "label", "group", "category"])


def load_dataset(name):
    CACHE.mkdir(exist_ok=True)
    x_path = CACHE / f"{name}_X.npy"
    meta_path = CACHE / f"{name}_meta.csv"
    if not x_path.exists():
        df = esc10_table() if name == "esc10" else sc_table()
        n = SECONDS[name] * SR
        X = np.stack([fix_length(load_wav(p), n) for p in df.path])
        np.save(x_path, X)
        df[["label", "group"]].to_csv(meta_path, index=False)
    X = np.load(x_path, mmap_mode="r")
    meta = pd.read_csv(meta_path)
    return X, meta.label.values, meta.group.values


def stratified(y, idx, n, seed):
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    keep = [rng.choice(idx[y[idx] == c], n // len(classes), replace=False) for c in classes]
    return np.sort(np.concatenate(keep))


def get_split(name, fold=1, seed=0):
    small = name == "sc_small"
    if small:
        name = "sc"
    X, y, group = load_dataset(name)
    if name == "esc10":
        val_fold = fold % 5 + 1
        train = ~np.isin(group, [fold, val_fold])
        val = group == val_fold
        test = group == fold
    else:
        train, val, test = group == "train", group == "val", group == "test"
    train, val, test = np.where(train)[0], np.where(val)[0], np.where(test)[0]
    if small:
        train, val = stratified(y, train, 320, seed), stratified(y, val, 80, seed)
    return [(np.array(X[i]), y[i].astype(np.int64)) for i in (train, val, test)]


def esc_noise_files():
    df = pd.read_csv(ESC_ROOT / "meta" / "esc50.csv")
    df = df[~df.esc10]
    return [ESC_ROOT / "audio" / f for f in df.filename if (ESC_ROOT / "audio" / f).exists()]


def esc_noise_bank(seed=0):
    path = CACHE / "esc_noise.npy"
    if not path.exists():
        files = esc_noise_files()
        if len(files) < ESC_NOISE_CLIPS:
            raise FileNotFoundError(f"need the full ESC-50 audio in {ESC_ROOT / 'audio'}, found only {len(files)} non-ESC-10 clips")
        CACHE.mkdir(exist_ok=True)
        pick = np.random.default_rng(seed).choice(len(files), ESC_NOISE_CLIPS, replace=False)
        np.save(path, np.stack([fix_length(load_wav(files[i]), 5 * SR) for i in pick]))
    return np.load(path)


def esc_crop(bank, n, rng):
    for _ in range(50):
        clip = bank[rng.integers(len(bank))]
        start = rng.integers(0, bank.shape[1] - n + 1)
        crop = clip[start : start + n]
        if (crop**2).mean() > 1e-6:
            break
    return crop


def add_noise(X, kind, snr_db, seed=0):
    rng = np.random.default_rng(seed)
    n_clips, n = X.shape
    if kind == "white":
        noise = rng.standard_normal((n_clips, n)).astype(np.float32)
    elif kind == "esc":
        bank = esc_noise_bank()
        noise = np.stack([esc_crop(bank, n, rng) for _ in range(n_clips)])
    else:
        files = [load_wav(SC_ROOT / "_background_noise_" / f"{f}.wav") for f in REAL_NOISE]
        crops = []
        for _ in range(n_clips):
            f = files[rng.integers(len(files))]
            start = rng.integers(0, len(f) - n)
            crops.append(f[start : start + n])
        noise = np.stack(crops)
    voiced = np.abs(X) > 1e-4
    signal_power = (X**2 * voiced).sum(1) / np.maximum(voiced.sum(1), 1)
    noise_power = np.maximum((noise**2).mean(1), 1e-12)
    scale = np.sqrt(signal_power / (10 ** (snr_db / 10) * noise_power))
    return (X + scale[:, None] * noise).astype(np.float32)


def apply_condition(X, condition):
    if condition == "clean":
        return X
    kind, snr = condition.split("_")
    return add_noise(X, kind, int(snr))


def load_noise_bank():
    return [torch.from_numpy(load_wav(SC_ROOT / "_background_noise_" / f"{f}.wav")) for f in REAL_NOISE]


def add_train_noise(x, bank, gen):
    n_clips, n = x.shape
    noise = torch.randn(n_clips, n, generator=gen)
    for i in range(n_clips):
        if torch.rand(1, generator=gen) < 0.5:
            f = bank[int(torch.randint(len(bank), (1,), generator=gen))]
            start = int(torch.randint(0, len(f) - n, (1,), generator=gen))
            noise[i] = f[start : start + n]
    snr = 20 * torch.rand(n_clips, generator=gen)
    voiced = (x.abs() > 1e-4).float()
    signal_power = (x**2 * voiced).sum(1) / voiced.sum(1).clamp(min=1)
    noise_power = (noise**2).mean(1).clamp(min=1e-12)
    scale = torch.sqrt(signal_power / (10 ** (snr / 10) * noise_power))
    return x + scale[:, None] * noise


class LogMel(torch.nn.Module):
    def __init__(self, n_mels=64, n_fft=1024, hop=160):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            SR, n_fft=n_fft, hop_length=hop, n_mels=n_mels, f_min=20, f_max=SR // 2
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB("power", top_db=80)

    def forward(self, wav):
        return self.to_db(self.mel(wav)).unsqueeze(1)
