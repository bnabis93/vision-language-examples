PYTHON=3.10
BASENAME=$(shell basename $(CURDIR))
CURRENT_DIR = $(shell pwd)

env:
	conda create -n $(BASENAME) -y python=$(PYTHON)

setup:
	pip install -r requirements.txt

convert-trt:
	mkdir -p $(CURDIR)/output
	docker run --gpus all -it \
		-v $(CURDIR):/workspace/src \
		-v $(CURDIR)/output:/workspace/output \
		--rm nvcr.io/nvidia/pytorch:22.03-py3 \
		/bin/bash -c "python src/onnx_export.py && \
			trtexec --onnx=./model.onnx \
					--saveEngine=./output/model.trt"