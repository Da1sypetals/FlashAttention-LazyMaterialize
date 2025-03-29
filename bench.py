import torch
from torch.utils.cpp_extension import load
import einops as ein

# Load the CUDA kernel as a python module
minimal_attn = load(
    name="minimal_attn", sources=["main.cpp", "flash.cu"], extra_cuda_cflags=["-O2"]
)

# Use small model params, otherwise slower than manual attention. See caveats in README.
batch_size = 1
n_head = 12
seq_len = 64
head_embd = 64
mask_embed = 8

q = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
k = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
v = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
mf = (torch.rand(batch_size, seq_len, mask_embed) < 0.5).to(dtype=torch.int32)
# mf = torch.ones(batch_size, seq_len, seq_len, dtype=torch.int32)
mask = (ein.einsum(mf, mf, "b l1 d, b l2 d -> b l1 l2") > 0).to(torch.bool)
mf = mf.cuda()
mask = ein.repeat(mask, "b l1 l2 -> b h l1 l2", h=n_head).cuda()
print(mask.sum().item())
print(batch_size * n_head * seq_len * seq_len)


# Our minimal flash attention aims to be faster than this by avoiding HBM read/writes of N^2 matrices.
def manual_attn(q, k, v, mask):
    res = torch.nn.functional.scaled_dot_product_attention(q, k, v, mask)
    return res


print("=== profiling manual attention ===")


with torch.autograd.profiler.profile(use_cuda=True) as prof:
    manual_result = manual_attn(q, k, v, mask)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print("=== profiling minimal flash attention === ")

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    minimal_result = minimal_attn.forward(q, k, v, mf)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


diff = minimal_result - manual_result
print(f"Max manual: {manual_result.abs().max()}")
print(f"Max diff: {diff.abs().mean() / manual_result.abs().mean()}")

print(
    "attn values sanity check:",
    torch.allclose(minimal_result, manual_result, rtol=0, atol=1e-2),
)
