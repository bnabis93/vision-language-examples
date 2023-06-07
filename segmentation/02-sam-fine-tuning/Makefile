PYTHON=3.10
BASENAME=$(shell basename $(CURDIR))
CURRENT_DIR = $(shell pwd)

env:
	conda create -n $(BASENAME) -y python=$(PYTHON)

setup:
	pip install -r requirements.txt
	pip install git+https://github.com/facebookresearch/segment-anything.git

download-dataset:
	gdown https://drive.google.com/uc?id=18GhVEODbTi17jSeBXdeLQ7vHPdtlTYXK
	mkdir -p dataset
	unzip MedSAMDemo_2D.zip -d dataset
	rm -r MedSAMDemo_2D.zip


