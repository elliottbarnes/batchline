.PHONY: run test load docker-build

run:
	PYTHONPATH=src python3 -m inference_service.server

test:
	PYTHONPATH=src python3 -m unittest discover -s tests -v

load:
	python3 scripts/load_test.py

docker-build:
	docker build -t batchline:local .
