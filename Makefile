SHELL := bash

.PHONY: check clean reformat dist venv docker docker-nobuildx

all: dist

check:
	scripts/check-code.sh

reformat:
	scripts/format-code.sh

venv:
	scripts/create-venv.sh

dist:
	python3 setup.py sdist

docker:
	scripts/build-docker.sh

docker-nobuildx:
	DOCKER_BUILDX='' scripts/build-docker.sh
