PY ?= python3

.PHONY: test repro data run run_real judge plots banner app demo all

test:            ## fast checks on the committed real data and results (no torch)
	$(PY) -m skilladmit.ltt && $(PY) test_skilladmit.py

repro: plots     ## regenerate every plot from the committed results/*.json

all: data run run_real judge plots   ## the whole CPU run (~1 h on 8 cores)

data:            ## clone the 9 pinned skill repos into data/src and rebuild data/skills.jsonl.gz
	$(PY) -m skilladmit.data
run:             ## 3 seeds x 1000 streams per split, real + SYNTH near-duplicate candidates
	$(PY) -m skilladmit.run --seeds 0,1,2 --n 1000 --n_train 400
run_real:        ## same, real candidates only
	$(PY) -m skilladmit.run --seeds 0,1,2 --n 1000 --n_train 400 --mix real_only
judge:           ## small-scale prompted Qwen2.5-0.5B judge arm (needs torch + transformers)
	$(PY) -m skilladmit.run --seeds 0 --judge 100
plots:
	$(PY) scripts/plots.py
banner:
	$(PY) scripts/banner.py
app:
	$(PY) app/server.py 8766
demo:            ## needs `make app` running
	$(PY) scripts/demo_gif.py
