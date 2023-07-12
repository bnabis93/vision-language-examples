import copy
import time
import torch
from torch import nn
from torch.utils import benchmark

import xformers.components.attention.attention_patterns as AP
from xformers.components.attention.core import scaled_dot_product_attention
from xformers.components.attention._sputnik_sparse import SparseCS
from xformers.ops import memory_efficient_attention

import timm
from timm.models.vision_transformer import VisionTransformer


def profile_model(fn, min_run_time=2):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    res = benchmark.Timer(
        stmt="fn()", globals={"fn": fn}, label="profile", sub_label="", description=""
    ).blocked_autorange(min_run_time=min_run_time)
    torch.cuda.synchronize()
    memory = torch.cuda.max_memory_allocated() / 2**20
    memory = f"Memory used: {memory} MB"
    print(res)
    print(memory)


class Attention(torch.nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        attn_mask=None,
    ):
        super().__init__()
        self.num_heads = num_heads

        self.qkv = torch.nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = torch.nn.Linear(dim, dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.attn_mask = attn_mask

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )

        qkv = qkv.flatten(1, 2)

        q, k, v = qkv.unbind()

        x = scaled_dot_product_attention(
            q, k, v, self.attn_mask, dropout=self.attn_drop
        )
        x = x.reshape(B, self.num_heads, N, C // self.num_heads)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffiAttention(torch.nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads

        self.qkv = torch.nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = torch.nn.Linear(dim, dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )

        qkv = qkv.flatten(1, 2)

        q, k, v = qkv.unbind()

        x = memory_efficient_attention(q, k, v, dropout=self.attn_drop)
        x = x.reshape(B, self.num_heads, N, C // self.num_heads)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def replace_attn_with_xformers_one(module, att_mask, mem_effi=False):
    module_output = module
    if isinstance(module, timm.models.vision_transformer.Attention):
        qkv = module.qkv
        dim = qkv.weight.shape[1] * module.num_heads
        if mem_effi:
            module_output = MemEffiAttention(dim, module.num_heads)
        else:
            module_output = Attention(dim, module.num_heads, attn_mask=att_mask)
    for name, child in module.named_children():
        module_output.add_module(name, replace_attn_with_xformers_one(child, att_mask))
    del module
    return module_output


img_size = 224
patch_size = 16

model = VisionTransformer(
    img_size=img_size,
    patch_size=patch_size,
    embed_dim=96,
    depth=8,
    num_heads=8,
    mlp_ratio=3.0,
    qkv_bias=False,
    norm_layer=nn.LayerNorm,
).cuda()
model_sparse = copy.deepcopy(model)
model_memory_efficient = copy.deepcopy(model)


H, W = img_size // patch_size, img_size // patch_size
print(f"Sequence length: {H}x{W} = {H * W}")

axial_pattern = AP.axial_2d_pattern(H, W)
loc_2d_dist = AP.local_2d_pattern(H, W, distance=2, p=2.0)
rand_pattern = torch.rand((H * W) ** 2).reshape(H * W, H * W) > 0.99

gaus_2d_dist = AP.local_2d_gausian_distribution(H, W, sigma=5)
sparsity = 0.97
num_non_zeros = int((H * W) ** 2 * (1 - sparsity))
random_gaus_2d_pattern = AP.random_pattern_from_probability_matrix(
    gaus_2d_dist, num_non_zeros
)


t_mask = axial_pattern | loc_2d_dist | rand_pattern | random_gaus_2d_pattern

# and let's not forget to add a global attention for the cls_token
mask = torch.ones((H * W + 1, H * W + 1), dtype=torch.bool)
mask[1:, 1:] = t_mask
mask = SparseCS(mask, torch.device("cuda"))
print(1 - mask.values.shape[1] / (mask.shape[0] * mask.shape[1]))


model_sparse = replace_attn_with_xformers_one(model_sparse, mask)
model_memory_efficient = replace_attn_with_xformers_one(
    model_memory_efficient, att_mask=None, mem_effi=True
)
i = torch.rand(1, 3, img_size, img_size).cuda()

# Autocast to fp16
with torch.cuda.amp.autocast(device_type="cuda", dtype=torch.float16):
    print("ViT Forward only")
    profile_model(lambda: model(i))
    print("Sparse ViT Forward only")
    profile_model(lambda: model_sparse(i))
    print("Mem efficient ViT Forward only")
    profile_model(lambda: model_memory_efficient(i))


inference_times = []
with torch.no_grad():
    with torch.cuda.amp.autocast(device_type="cuda", dtype=torch.float16):
        for _ in range(100):
            torch.cuda.synchronize()
            start = time.time()
            model(i)
            end = time.time()
            torch.cuda.synchronize()
            inference_times.append((end - start) * 1000)

print(f"ViT average inference time : {sum(inference_times)/len(inference_times)}ms")


inference_times = []
with torch.no_grad():
    with torch.cuda.amp.autocast(device_type="cuda", dtype=torch.float16):
        for _ in range(100):
            torch.cuda.synchronize()
            start = time.time()
            model_sparse(i)
            end = time.time()
            torch.cuda.synchronize()
            inference_times.append((end - start) * 1000)

print(f"ViT average inference time : {sum(inference_times)/len(inference_times)}ms")


inference_times = []
with torch.no_grad():
    with torch.cuda.amp.autocast(device_type="cuda", dtype=torch.float16):
        for _ in range(100):
            torch.cuda.synchronize()
            start = time.time()
            model_memory_efficient(i)
            end = time.time()
            torch.cuda.synchronize()
            inference_times.append((end - start) * 1000)

print(f"ViT average inference time : {sum(inference_times)/len(inference_times)}ms")
