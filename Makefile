.PHONY: eval eval-retrieval eval-generation update-baselines

eval:
	python3 scripts/run_eval.py

eval-retrieval:
	python3 scripts/run_eval.py --only retrieval

eval-generation:
	python3 scripts/run_eval.py --only generation

update-baselines:
	python3 scripts/update_baselines.py
