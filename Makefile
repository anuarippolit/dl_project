PY = .venv/bin/python

# train
train:
	PYTORCH_ENABLE_MPS_FALLBACK=1 $(PY) src/main.py

# run
run:
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/results.ipynb

NOTEBOOKS = q1_seeds q3_undertraining q5_confidence_vs_accuracy q6_noise_source q7_temperature q8_noise_training
NBRUN = $(PY) -m jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1

# extra experiments, can be stopped and restarted
extra:
	for n in $(NOTEBOOKS); do PYTORCH_ENABLE_MPS_FALLBACK=1 $(NBRUN) --inplace notebooks/$$n.ipynb || exit 1; done

# fast check of the extra notebooks (1 epoch, tiny test set), notebooks are not changed
extra-quick:
	for n in $(NOTEBOOKS); do EXTRA_QUICK=1 PYTORCH_ENABLE_MPS_FALLBACK=1 $(NBRUN) --output-dir results/quick_check notebooks/$$n.ipynb || exit 1; done
