.PHONY: help install test test-py test-js test-all bench bench-quick bench-full clean

help:
	@echo "fmc-core targets:"
	@echo "  install        editable install with test extras"
	@echo "  test           pytest (Python only)"
	@echo "  test-py        alias for test"
	@echo "  test-js        node js test runner"
	@echo "  test-all       both Python and JS test suites"
	@echo "  bench          rocket alpha-beta sweep, default size"
	@echo "  bench-quick    rocket sweep, 3 seeds (smoke test)"
	@echo "  bench-full     rocket sweep, 20 seeds (slow, ~minutes)"
	@echo "  clean          remove build artifacts and bench outputs"

install:
	pip install -e ".[test]"

test test-py:
	pytest

test-js:
	node js/_test.js

test-all: test-py test-js

bench: bench-rocket bench-nav2d bench-pendulum bench-cK bench-M bench-N

bench-M:
	python3 -m bench.M_dependence

bench-N:
	python3 -m bench.N_dependence

bench-rocket:
	python3 -m bench.rocket_sweep

bench-nav2d:
	python3 -m bench.navigation2d_sweep

bench-pendulum:
	python3 -m bench.pendulum_sweep

bench-cK:
	python3 -m bench.c_K_shape

bench-quick:
	python3 -m bench.rocket_sweep --quick
	python3 -m bench.navigation2d_sweep --quick
	python3 -m bench.pendulum_sweep --quick

bench-full:
	python3 -m bench.rocket_sweep --full
	python3 -m bench.navigation2d_sweep --full
	python3 -m bench.pendulum_sweep --full
	python3 -m bench.c_K_shape

clean:
	rm -rf build dist *.egg-info src/*.egg-info
	rm -rf .pytest_cache **/__pycache__ **/**/__pycache__
	rm -f bench/results/*.jsonl
	rm -f tests/_fixture_relativize.json
