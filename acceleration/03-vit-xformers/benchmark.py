import time
import timm
import torch
import copy
import xformers.components.attention.attention_patterns as AP
from torch.utils import benchmark
from xformers.helpers.timm_sparse_attention import TimmSparseAttention
from xformers.components.attention._sputnik_sparse import SparseCS

# Define global variables
img_size = 224
patch_size = 16
batch_size = 1
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == "cpu":
    print("This code only supports GPU.")
    exit(-1)
print(f"Device : {device}")


# Get Sparse Attention mask
def get_sparse_attention_mask(img_size: int, patch_size: int, sparsity: float = 0.97):
    """
    https://github.com/facebookresearch/xformers/blob/main/docs/source/vision_transformers.ipynb
    """
    H, W = img_size // patch_size, img_size // patch_size
    print(f"Sequence length: {H}x{W} = {H * W}")

    axial_pattern = AP.axial_2d_pattern(H, W)
    loc_2d_dist = AP.local_2d_pattern(H, W, distance=2, p=2.0)
    rand_pattern = torch.rand((H * W) ** 2).reshape(H * W, H * W) > 0.99

    gaus_2d_dist = AP.local_2d_gausian_distribution(H, W, sigma=5)
    num_non_zeros = int((H * W) ** 2 * (1 - sparsity))
    random_gaus_2d_pattern = AP.random_pattern_from_probability_matrix(
        gaus_2d_dist, num_non_zeros
    )

    t_mask = axial_pattern | loc_2d_dist | rand_pattern | random_gaus_2d_pattern

    # and let's not forget to add a global attention for the cls_token
    mask = torch.ones((H * W + 1, H * W + 1), dtype=torch.bool)
    mask[1:, 1:] = t_mask

    print(f"Sparsity: {1 - mask.float().mean().item()}, nnz={num_non_zeros}")
    return mask


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


# Get a reference ViT model
model = timm.create_model("vit_base_patch16_224", pretrained=True).to(device)
model_sparse = copy.deepcopy(model)

sparse_attn = get_sparse_attention_mask(img_size, patch_size, sparsity=0.97)
sparse_attn = SparseCS(sparse_attn, torch.device("cuda"))
model_sparse = replace_attn_with_xformers_one(model_sparse, sparse_attn)

img = torch.rand(batch_size, 3, img_size, img_size)
for _ in range(10):
    model(img.to(device))

inference_times = []
with torch.no_grad():
    for _ in range(100):
        torch.cuda.synchronize()
        start = time.time()
        model(img.to(device))
        torch.cuda.synchronize()
        end = time.time()
        inference_times.append((end - start) * 1000)

print(f"ViT average inference time : {sum(inference_times)/len(inference_times)}ms")

inference_times = []
with torch.no_grad():
    for _ in range(100):
        torch.cuda.synchronize()
        start = time.time()
        model_sparse(img.to(device))
        torch.cuda.synchronize()
        end = time.time()
        inference_times.append((end - start) * 1000)

print(
    f"Sparse ViT average inference time : {sum(inference_times)/len(inference_times)}ms"
)
