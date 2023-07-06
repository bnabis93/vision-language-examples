""" Vit export to onnx format.
- Author: Bono (qhsh9713@gmail.com)
"""
import timm
from timm.utils.onnx import onnx_export


# Export vit
model = timm.create_model(
    "vit_base_patch16_224",
    pretrained=True,
    exportable=True,
)
onnx_export(
    model,
    "./output/model.onnx",
    opset=17,
    dynamic_size=False,
)
