# Generalist vs. Specialist: Wav2Vec2 vs. spectrogram models

CSCI 494 course project. We compare MFCC + Random Forest, a log-mel CNN, ResNet-18 (ImageNet) and Wav2Vec2 (speech pretrained, raw waveform) on ESC-10 (environmental sounds, mismatched domain for Wav2Vec2) and Speech Commands (speech, matched domain). We look at clean accuracy, cost, and accuracy and calibration (ECE) under white and real noise at 20, 10 and 0 dB SNR. `sc_small` is Speech Commands with only 320 training clips, close to the 240 training clips that ESC-10 has in each fold.

### Results

Clean accuracy (first full run, one seed):

| | ESC-10 | Speech Commands | SC, 320 clips |
|---|---|---|---|
| RF | 0.782 | 0.633 | 0.416 |
| CNN | 0.832 | 0.970 | 0.598 |
| ResNet-18 | 0.872 | 0.980 | 0.494 |
| Wav2Vec2 | 0.822 | 0.984 | 0.950 |

Wav2Vec2 is not better than ResNet on ESC-10, and it is only clearly better with little speech data. It has 94M parameters (CNN: 0.24M) and takes about 94 min to train on the full Speech Commands (CNN: 8 min). Under strong noise all neural networks become overconfident on ESC-10 (ECE 0.5-0.8) but stay calibrated on Speech Commands (ECE below 0.1). Temperature scaling fitted on clean validation data does not fix this.

### Extra checks

After the first run we checked ourselves with six notebooks in `notebooks/` (`q1_seeds`, `q3_undertraining`, `q5_confidence_vs_accuracy`, `q6_noise_source`, `q7_temperature`, `q8_noise_training`).

- **Are the gaps real? (q1)** With 3 seeds, ResNet is reliably better than the CNN on ESC-10 (0.877 against 0.834). On 320 clips Wav2Vec2 reached 0.943 and 0.950 in two runs but failed in the third (0.101, one class for every clip). On ESC-10 two runs of Wav2Vec2 gave 0.822 and 0.850.
- **Were the small models under-trained? (q3)** Partly. With 320 clips the CNN went from 0.603 to 0.789 and ResNet from 0.521 to 0.655 when trained about 7 to 10 times longer. Wav2Vec2 stays higher (0.954-0.957), so the gap is about 17 points, not 35.
- **Domain or just a broken model? (q5)** At the same accuracy (below 40%) ESC-10 models are more overconfident than speech models (confidence minus accuracy +0.77 and +0.41 against +0.24 and +0.10). Few points per range, and full Speech Commands never gets that low.
- **Does the noise source matter? (q6)** Yes. On ESC-10 at 0 dB, accuracy is 0.11-0.13 with white noise, 0.24-0.25 with the Speech Commands background and 0.51-0.56 with noise made of other ESC-50 sounds. Overconfidence falls from about +0.75 to between -0.01 and +0.21.
- **Wav2Vec2 on ESC-10 (q6, q7, q8):** it behaves like ResNet. At 0 dB it is at 13-25% accuracy and overconfident by +0.71 (white) and +0.52 (real).
- **Can temperature scaling fix it? (q7)** One temperature fitted on clean data does not. One temperature fitted on all noise levels helps on noise it has seen (ESC-10 mean ECE 0.28 to 0.21) but hurts clean sound (0.12 to 0.32) and does not transfer to a new noise kind. A temperature per noise condition works (ECE 0.07) but needs the noise to be known.
- **Can training with noise fix it? (q8)** On seen noise yes (ESC-10 ResNet ECE 0.34 to 0.04, Wav2Vec2 0.33 to 0.07). The price is clean accuracy (ESC-10: 4 to 12 points lower, Wav2Vec2 0.85 to 0.73), and with 320 clips it hurts badly (Wav2Vec2 0.94 to 0.61). On a new noise kind the gain is smaller.

### Conclusion

Wav2Vec2 turned out not to be a universal model. On environmental sounds (ESC-10) it is no better than a plain ResNet-18, and on speech it is clearly better only when there is little data (320 clips: 0.95 against 0.60 for the CNN, or 0.79 after the CNN trains much longer). With the full dataset its advantage almost disappears (0.984 against 0.970 for the CNN), while it costs 390 times more parameters and about 11 times more training time, so it is only worth it when data is scarce and the domain matches its pretraining. Fine-tuning it on 320 clips is also unstable (one run in three failed). Under strong noise every neural network, Wav2Vec2 included, becomes confidently wrong on ESC-10 (ECE 0.5-0.8) while on Speech Commands they stay calibrated. At the same accuracy ESC-10 is still more overconfident than speech, so the effect is not only "the model is broken", but it depends a lot on the kind of noise. A single temperature does not fix it, and training with noise fixes it for seen noise at the cost of clean accuracy. The results agree with the idea that a domain mismatch breaks calibration under noise, but we cannot prove it.

### Look at the results

Open `notebooks/results.ipynb` (first run: tables and plots) and the `q*.ipynb` notebooks (extra checks). `notebooks/eda.ipynb` shows the data.

- `results/results.csv`: one row per dataset, model and noise condition (acc, f1, ece, nll, overconf, ece_ts, temperature)
- `results/reliability.csv`: confidence bins for reliability diagrams
- `results/cost.csv`: parameters, training time and latency per run
- `results/figures/`: plots made by the extra notebooks
- `results/extra_logits/`: raw model outputs of the extra notebooks (not in git, made by `make extra`)

### Reproduce

Put the full ESC-50 audio (2000 clips) in `data/ESC-50-master/` and Speech Commands v0.02 in `data/speech_commands_v0.02/` (the data is not in git), then:

    uv venv --python 3.12 .venv
    uv pip install --python .venv/bin/python -r requirements.txt
    make train

`make train` takes a few hours (Wav2Vec2 on Speech Commands is most of it). Finished experiments are skipped, so it can be stopped and restarted. `make run` executes the results notebook.

`make extra` runs the six extra notebooks (a few more hours, also resumable). `make extra-quick` is a fast check with 1 epoch and a small test set, and it does not change the notebooks. Wav2Vec2 on ESC-10 and on the full Speech Commands is switched off by flags at the top of some notebooks because it takes about an hour or more per run.

### Code

- `src/data.py`: loading, splits, noise (white, real, ESC-50), log-mel, noise for training
- `src/models.py`: the four models and the training loop
- `src/metrics.py`: accuracy, F1, ECE, NLL, temperature scaling
- `src/main.py`: runs the first full run and writes `results/`
- `src/extra.py`: trains a model once and saves its raw outputs for every noise condition, used by the extra notebooks
- `report/`: scripts that build the figures and PDFs for the presentation

ESC-10 uses 5 folds (test fold, next fold for validation, rest for training), Speech Commands uses the official lists. Models are trained on clean audio only (except in `q8`), noise is added to the test clips, and temperature is fitted on clean validation data (except in `q7`).

### Limitations

Most experiments use one seed, and ESC-10 has only 400 test clips, so differences of a few points are not reliable. ESC-10 and Speech Commands differ in more than domain (clip length, class types, noise source), so the results agree with the domain hypothesis but do not prove it. We tested one speech model only; models pre-trained on many kinds of sound were not tried. In `q8` the "seen" noise comes from the same background files as the "real" test noise, so only the ESC-50 noise is a fair test of new noise.
