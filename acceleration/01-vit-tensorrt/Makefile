convert-trt:
	mkdir -p $(CURDIR)/output
	docker run --gpus all -it \
		-v $(CURDIR):/workspace/src \
		-v $(CURDIR)/output:/workspace/output \
		--rm nvcr.io/nvidia/pytorch:23.06-py3 \
		/bin/bash -c "pip install timm onnxruntime-gpu && python src/onnx_export.py && \
			trtexec --onnx=./output/model.onnx \
					--saveEngine=./output/model.trt"

benchmark:
	docker run --gpus device=0 -it \
		-v $(CURDIR)/output:/workspace/output \
		--rm nvcr.io/nvidia/pytorch:23.06-py3 \
		/bin/bash -c "trtexec \
			--loadEngine=./output/model.trt \
			--batch=1"
