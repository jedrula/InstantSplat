# InstantSplat TODOs

## Performance
- [ ] Compile RoPE2D CUDA kernel to speed up MASt3R inference (~20-40% faster on that step).
      Currently falls back to slow PyTorch version: "cannot find cuda-compiled version of RoPE2D".
      Requires building the CUDA extension inside the mast3r/dust3r submodule.
