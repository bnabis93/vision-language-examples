import timm
from xformers.components.attention import ScaledDotProduct
from xformers.helpers.timm_sparse_attention import TimmSparseAttention
import torch.nn as nn
import torch

# Define global variables
img_size = 224
patch_size = 16
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == "cpu":
    print("This code only supports GPU.")
    exit(-1)
print(f"Device : {device}")

# Get a reference ViT model
model = timm.create_model("vit_base_patch16_224", pretrained=True)
model = model.to(device)

# Define a recursive monkey patching function
def replace_attn_with_xformers_one(module, att_mask):
    module_output = module
    if isinstance(module, timm.models.vision_transformer.Attention):
        qkv = module.qkv
        dim = qkv.weight.shape[1] * module.num_heads
        # Extra parameters can be exposed in TimmSparseAttention, this is a minimal example
        module_output = TimmSparseAttention(dim, module.num_heads, attn_mask=att_mask)
    for name, child in module.named_children():
        module_output.add_module(name, replace_attn_with_xformers_one(child, att_mask))
    del module

    return module_output


# Now we can just patch our reference model, and get a sparse-aware variation
## Attention: https://facebookresearch.github.io/xformers/components/attentions.html
### Scaled Dot Product Attention
model_sdp_attn = replace_attn_with_xformers_one(model, ScaledDotProduct)
