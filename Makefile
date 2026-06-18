.PHONY: all data prepare topics macro analysis advanced final \
        smoke clean coherence-grid full-corpus-transform

PYTHON := python3
SRC := src
CONFIG := config.yaml

all: data prepare topics macro analysis advanced final

data:
	$(PYTHON) $(SRC)/01_fetch_data.py --config $(CONFIG)

prepare:
	$(PYTHON) $(SRC)/02_prepare_data.py --config $(CONFIG)

topics:
	$(PYTHON) $(SRC)/03_train_bertopic.py --config $(CONFIG)

macro:
	$(PYTHON) $(SRC)/04_macro_topics.py --config $(CONFIG)

analysis:
	$(PYTHON) $(SRC)/05_comprehensive_analysis.py --config $(CONFIG)

advanced:
	$(PYTHON) $(SRC)/06_advanced_analytics.py --config $(CONFIG)

final:
	$(PYTHON) $(SRC)/07_final_analytics.py --config $(CONFIG)

coherence-grid:
	$(PYTHON) $(SRC)/experiments/topic_coherence_grid.py

full-corpus-transform:
	$(PYTHON) $(SRC)/experiments/full_corpus_transform.py

smoke:
	$(PYTHON) -c "import sys; sys.path.insert(0,'src'); from utils import load_config; c=load_config(); print(f'OK: {len(c[\"categories\"])} categories, seed={c[\"random_seed\"]}')"
	$(PYTHON) -m py_compile src/utils.py src/02_prepare_data.py src/03_train_bertopic.py src/04_macro_topics.py src/05_comprehensive_analysis.py src/06_advanced_analytics.py src/07_final_analytics.py src/experiments/topic_coherence_grid.py src/experiments/full_corpus_transform.py

clean:
	rm -rf analysis_output/* logs/* data/processed/* data/results/*
