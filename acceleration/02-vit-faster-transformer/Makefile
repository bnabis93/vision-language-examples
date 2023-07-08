docker-setup:
	git clone https://github.com/NVIDIA/FasterTransformer.git
	docker run \
		-it \
		--rm \
		--gpus="device=0" \
		"--cap-add=SYS_ADMIN" \
		--shm-size=16g \
		--ulimit memlock=-1 \
		--ulimit stack=67108864 \
		-v $(CURDIR)/{FasterTransformer}:/workspace/FasterTransformer \
		--workdir /workspace/FasterTransformer \
		nvcr.io/nvidia/pytorch:22.09-py3 bash -c "export WORKSPACE = /workspace/FasterTransformer"