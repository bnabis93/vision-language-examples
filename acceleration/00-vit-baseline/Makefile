PYTHON=3.10
BASENAME=$(shell basename $(CURDIR))
CURRENT_DIR = $(shell pwd)

env:
	conda create -n $(BASENAME) -y python=$(PYTHON)

setup:
	pip install -r requirements.txt