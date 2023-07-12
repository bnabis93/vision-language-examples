convert-trt:
	mkdir -p $(CURDIR)/output
	docker run --gpus all -it \
		-v $(CURDIR):/workspace/src \
		-v $(CURDIR)/output:/workspace/output \
		--rm nvcr.io/nvidia/pytorch:23.06-py3 \
		/bin/bash -c "pip install timm onnxruntime-gpu && python src/onnx_export.py && \
			trtexec --onnx=./output/model.onnx \
					--minShapes=input:1x3x224x224 \
					--optShapes=input:1x3x224x224 \
					--maxShapes=input:64x3x1024x1024 \
					--explicitBatch \
					--saveEngine=./output/model.plan"

benchmark:
	docker run --gpus device=0 -it \
		-v $(CURDIR)/output:/workspace/output \
		--rm nvcr.io/nvidia/pytorch:23.06-py3 \
		/bin/bash -c "trtexec \
			--loadEngine=./output/model.trt \
			--batch=1"
