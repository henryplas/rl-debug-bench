.PHONY: install test docker-build repro

install:
	pip install -e ".[dev]"

test:
	pytest -q

docker-build:
	docker build -t rl-debug-bench .

# TODO(build order step 9): wire up eval/run_eval.py + eval/analyze.py here
# once the harness and scoring components exist.
repro:
	@echo "not implemented yet: see README.md build order, step 9"
	@exit 1
