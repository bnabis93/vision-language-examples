PYTHON=3.9
BASENAME=$(shell basename $(CURDIR))
CURRENT_DIR = $(shell pwd)

env:
	conda create -n $(BASENAME) -y python=$(PYTHON)

setup:
	pip install ftfy regex tqdm
	pip install git+https://github.com/openai/CLIP.git