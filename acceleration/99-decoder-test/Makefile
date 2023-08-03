PYTHON=3.10
BASENAME=$(shell basename $(CURDIR))
CURRENT_DIR = $(shell pwd)

env:
	conda create -n $(BASENAME) -y python=$(PYTHON)

setup:
	pip install -r requirements.txt


convert-trt:
	mkdir -p $(CURDIR)/output
	docker run --gpus device=0 -it \
		-v $(CURDIR):/workspace/src \
		-v $(CURDIR)/output:/workspace/output \
		--rm nvcr.io/nvidia/pytorch:23.06-py3 \
		/bin/bash -c "pip install timm onnxruntime-gpu && python src/onnx_export.py && \
			trtexec --onnx=./output/model.onnx \
					--minShapes=input0:1x256x32x32 \
					--optShapes=input0:1x256x32x32 \
					--maxShapes=input0:64x256x32x32 \
					--explicitBatch \
					--saveEngine=./output/model.plan"

convert-fp16-trt:
	mkdir -p $(CURDIR)/output
	docker run --gpus device=0 -it \
		-v $(CURDIR):/workspace/src \
		-v $(CURDIR)/output:/workspace/output \
		--rm nvcr.io/nvidia/pytorch:23.06-py3 \
		/bin/bash -c "pip install timm onnxruntime-gpu && python src/onnx_export.py && \
			trtexec --onnx=./output/model.onnx \
					--minShapes=input0:1x256x32x32 \
					--optShapes=input0:1x256x32x32 \
					--maxShapes=input0:64x256x32x32 \
					--explicitBatch \
					--fp16 \
					--saveEngine=./output/model_fp16.plan"

benchmark:
	docker run --gpus device=0 -it \
		-v $(CURDIR)/output:/workspace/output \
		--rm nvcr.io/nvidia/pytorch:23.06-py3 \
		/bin/bash -c "trtexec \
			--loadEngine=./output/model.plan \
			--shapes=input0:32x256x32x32"

benchmark-fp16:
	docker run --gpus device=0 -it \
		-v $(CURDIR)/output:/workspace/output \
		--rm nvcr.io/nvidia/pytorch:23.06-py3 \
		/bin/bash -c "trtexec \
			--loadEngine=./output/model_fp16.plan \
			--shapes=input0:32x256x32x32"