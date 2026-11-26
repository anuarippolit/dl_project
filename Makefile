data:
	mkdir -p data/speech_commands_v0.02
	curl -L -o data/esc50.zip https://github.com/karoldvl/ESC-50/archive/master.zip
	unzip -q data/esc50.zip -d data
	curl -L https://download.tensorflow.org/data/speech_commands_v0.02.tar.gz | tar -xz -C data/speech_commands_v0.02

train:
	python src/main.py

notebooks:
	jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/*.ipynb
