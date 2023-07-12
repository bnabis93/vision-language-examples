convert-trt:
	mkdir -p $(CURDIR)/output
	docker run --gpus device=0 -it \
		-v $(CURDIR):/workspace/src \
		-v $(CURDIR)/output:/workspace/output \
		--rm nvcr.io/nvidia/pytorch:23.06-py3 \
		/bin/bash -c "pip install timm onnx onnxruntime-gpu && python src/onnx_export.py && python src/check_onnx.py && \
			trtexec --onnx=./output/model.onnx \
					--minShapes=input0:1x3x224x224 \
					--optShapes=input0:1x3x224x224 \
					--maxShapes=input0:64x3x224x224 \
					--explicitBatch \
					--saveEngine=./output/model.plan"

benchmark:
	docker run --gpus device=7 -it \
		-v $(CURDIR)/output:/workspace/output \
		--rm nvcr.io/nvidia/pytorch:23.06-py3 \
		/bin/bash -c "trtexec \
			--loadEngine=./output/model.plan \
			--shapes=input0:64x3x224x224"
