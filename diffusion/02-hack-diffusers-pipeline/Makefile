PYTHON=3.10
BASENAME=$(shell basename $(CURDIR))

env:
	conda create -n $(BASENAME) -y python=$(PYTHON)
setup:
	pip install -r requirements.txt
	pip install --upgrade diffusers[torch]
