PYTHON=3.9
BASENAME=$(shell basename $(CURDIR))

env:
	conda create -n $(BASENAME) -y python=$(PYTHON)

setup:
	pip install -r requirements.txt