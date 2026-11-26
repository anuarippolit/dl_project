# Generalist vs. Specialist: Wav2Vec2 vs. spectrogram models

We compare MFCC + Random Forest, a log-mel CNN, ResNet-18 (ImageNet) and Wav2Vec2 (speech pretrained) on ESC-10 (environmental sounds) and Speech Commands (speech). We look at accuracy, cost, and calibration (ECE) on clean audio and under white and real noise at 20, 10 and 0 dB SNR. `sc_small` is Speech Commands with only 320 training clips, close to the size of ESC-10.

## Results

Clean accuracy (one seed):

| | ESC-10 | Speech Commands | SC, 320 clips |
|---|---|---|---|
| RF | 0.782 | 0.633 | 0.416 |
| CNN | 0.832 | 0.970 | 0.598 |
| ResNet-18 | 0.872 | 0.980 | 0.494 |
| Wav2Vec2 | 0.822 | 0.984 | 0.950 |

- Wav2Vec2 is not better than ResNet on ESC-10, and it is clearly better than the small models only with little speech data.
- It has 94M parameters (CNN: 0.24M) and trains about 11 times longer on full Speech Commands.
- Under strong noise all neural networks become overconfident on ESC-10 (ECE 0.5-0.8) but stay calibrated on Speech Commands.
- Temperature scaling fitted on clean data does not fix this. Training with noise fixes it on seen noise but lowers clean accuracy.

The extra experiments are in the notebooks `q1` to `q8` (seeds, longer training, noise source, temperature, training with noise).

## Files

- `src/data.py`: data loading, splits, noise, log-mel
- `src/models.py`: the four models, training and prediction
- `src/metrics.py`: accuracy, F1, ECE, NLL, temperature scaling
- `src/main.py`: first full run, writes `results/*.csv`
- `notebooks/`: `results.ipynb` (main run), `eda.ipynb`, `q*.ipynb` (extra experiments)
- `results/`: csv files and figures

## How to run

    pip install -r requirements.txt
    make data
    make train
    make notebooks

`make data` downloads ESC-50 and Speech Commands v0.02 (about 2.5 GB). `make train` takes a few hours and can be stopped and started again. `make notebooks` runs all notebooks (the extra ones train more models and take a few more hours).

## Limitations

Most experiments use one seed, and ESC-10 has only 400 test clips, so differences of a few points are not reliable. ESC-10 and Speech Commands differ in more than domain (clip length, class types, noise source), so the results agree with the domain hypothesis but do not prove it.
