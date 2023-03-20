PYTHON=3.9
BASENAME=$(shell basename $(CURDIR))
CURRENT_DIR = $(shell pwd)

env:
	conda create -n $(BASENAME) -y python=$(PYTHON)

setup:
	pip install -r requirements.txt

build:
	docker build -t 00-vit-huggingface . 

run-docker:
	docker run -it --rm --gpus all \
		--network host \
		-v $(CURRENT_DIR):/workspace \
		--name 00-vit-huggingface 00-vit-huggingface
