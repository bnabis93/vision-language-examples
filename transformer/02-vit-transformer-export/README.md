# Transformer ViT Model to Onnx model
- I use the ViT model from transformer framework.
    - Note that we converted the weights from Ross Wightman’s [timm library](https://github.com/huggingface/pytorch-image-models), who already converted the weights from JAX to PyTorch. Credits go to him!

## ONNX
> ONNX is an open standard that defines a common set of operators and a common file format to represent deep learning models in a wide variety of frameworks, including PyTorch and TensorFlow.
> When a model is exported to the ONNX format, these operators are used to construct a computational graph (often called an intermediate representation) which represents the flow of data through the neural network.
> By exposing a graph with standardized operators and data types, ONNX makes it easy to switch between frameworks. 
> For example, a model trained in PyTorch can be exported to ONNX format and then imported in TensorFlow (and vice versa).

## Transformer model export to ONNX format
- Check the transformer model supported onnx export. https://huggingface.co/docs/transformers/serialization
- ViT is supported

### Local setup
```
# Create virtual env
$ make env
$ conda activate 02-vit-transformer-export

# Install dependencies
$ make setup
```

### Transformer.onnx (Kind of cli tool)
- You can use transformer's onnx cli tool. `python -m transformers.onnx`
```
$ python -m transformers.onnx --help
None of PyTorch, TensorFlow >= 2.0, or Flax have been found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
The ONNX export features are only supported for PyTorch or TensorFlow. You will not be able to export models without one of these libraries installed.
usage: Hugging Face Transformers ONNX exporter [-h] -m MODEL [--feature FEATURE] [--opset OPSET] [--atol ATOL] [--framework {pt,tf}] [--cache_dir CACHE_DIR]
                                               [--preprocessor {auto,tokenizer,feature_extractor,processor}] [--export_with_transformers]
                                               output

positional arguments:
  output                Path indicating where to store generated ONNX model.

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Model ID on huggingface.co or path on disk to load model from.
  --feature FEATURE     The type of features to export the model with.
  --opset OPSET         ONNX opset version to export the model with.
  --atol ATOL           Absolute difference tolerance when validating the model.
  --framework {pt,tf}   The framework to use for the ONNX export. If not provided, will attempt to use the local checkpoint's original framework or what is available in the environment.
  --cache_dir CACHE_DIR
                        Path indicating where to store cache.
  --preprocessor {auto,tokenizer,feature_extractor,processor}
                        Which type of preprocessor to use. 'auto' tries to automatically detect it.
  --export_with_transformers
                        Whether to use transformers.onnx instead of optimum.exporters.onnx to perform the ONNX export. It can be useful when exporting a model supported in transformers
                        but not in optimum, otherwise it is not recommended.
```
### Simple ViT onnx export 
- If you already know model ID, you can export onnx model easily.
- Model ID : google/vit-base-patch16-224, https://huggingface.co/google/vit-base-patch16-224
- atol is the tolerance. 
- Device : 2080ti
- torch model average inference time: 0.006316 seconds
```
$ python -m transformers.onnx --model=google/vit-base-patch16-224 --atol 1e-3 onnx/

# Measure inference time.
$ python inference.py 
['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
2023-04-05 01:24:35.075534148 [W:onnxruntime:Default, onnxruntime_pybind_state.cc:515 CreateExecutionProviderInstance] Failed to create TensorrtExecutionProvider. Please reference https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#requirements to ensure all dependencies are met.
Input name: pixel_values, output name: last_hidden_state
Average Inference time: 0.006433 seconds
```
