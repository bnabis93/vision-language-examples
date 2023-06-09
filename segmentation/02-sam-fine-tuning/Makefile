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
	mkdir -p data
	unzip MedSAMDemo_2D.zip -d data
	rm -r MedSAMDemo_2D.zip

download-model:
	mkdir -p models
	curl -O https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth && mv sam_vit_b_01ec64.pth models/sam_vit_b_01ec64.pth