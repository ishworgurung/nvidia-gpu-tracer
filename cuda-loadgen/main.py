# Sample torch app for exercising GPU tracer

import torch
import time

N = 4096

# Multiple large buffers to stress both compute and memory bandwidth
a = torch.randn(N, N, device='cuda', dtype=torch.float32)
b = torch.randn(N, N, device='cuda', dtype=torch.float32)
v = torch.randn(N, N, device='cuda', dtype=torch.float32)

print(f"Starting heavy GPU compute ({N}x{N} matmuls + elementwise + reductions) ...")
torch.cuda.synchronize()

start = time.perf_counter()
iterations = 0
while True:
    # Tensor core GEMM — compute bound
    c = torch.mm(a, b)

    # Transcendental elementwise ops — ALU + memory bandwidth bound
    v = torch.tanh(c) + torch.sin(v) * torch.cos(v)
    v = torch.exp(-v.abs()) + torch.sqrt(v.abs() + 1e-6)

    # Softmax + layer-norm style reductions — reduction bound
    v = torch.nn.functional.softmax(v, dim=-1)
    v = (v - v.mean(dim=-1, keepdim=True)) / (v.std(dim=-1, keepdim=True) + 1e-5)

    # Another GEMM with the result to chain dependency
    a = torch.mm(v, b)
    a = a / (a.norm() + 1e-6)  # renorm to keep values stable

    iterations += 1
    if iterations % 10 == 0:
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        print(f"  iter {iterations:4d} | elapsed {elapsed:.1f}s | a[0,0]={a[0,0].item():.4f}")
        if elapsed >= 30.0:
            break

torch.cuda.synchronize()
total = time.perf_counter() - start
print(f"Done. {iterations} iterations in {total:.2f}s")
