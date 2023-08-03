import torch
from decoder import Decoder
from torch.autograd import Variable

# Create a random input tensor
x = Variable(torch.randn(1, 512, 32, 32))

# Create an instance of the decoder
decoder = Decoder(num_classes=10)

# Specify the name of the output ONNX file
onnx_file_name = "./output/model.onnx"

# Export the model to an ONNX file
torch.onnx.export(
    decoder,  # model being run
    x,  # model input (or a tuple for multiple inputs)
    onnx_file_name,  # where to save the model (can be a file or file-like object)
    export_params=True,  # store the trained parameter weights inside the model file
    opset_version=17,  # the ONNX version to export the model to
    do_constant_folding=True,  # whether to execute constant folding for optimization
    input_names=["input"],  # the model's input names
    output_names=["output"],
)  # the model's output names
