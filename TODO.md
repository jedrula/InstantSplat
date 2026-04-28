# InstantSplat TODOs

## Pipeline choice

See [PIPELINE_COMPARISON.md](PIPELINE_COMPARISON.md) for a full write-up of why
InstantSplat outperforms the GLOMAP+gsplat approach for our climbing-wall use case,
and guidance on how to improve quality further.

## Performance
- [ ] Compile RoPE2D CUDA kernel to speed up MASt3R inference (~20-40% faster on that step).
      Currently falls back to slow PyTorch version: "cannot find cuda-compiled version of RoPE2D".
      Requires building the CUDA extension inside the mast3r/dust3r submodule.

- [ ] Evaluate Fast3R as a drop-in replacement for MASt3R pose estimation.
      Feed-forward architecture — no global alignment step, so no O(N²) memory problem.
      Claims 4-10× speedup at similar quality. Could unlock more frames on 8GB GPU.

## Evaluated and ruled out

### GaussianObject (SIGGRAPH Asia 2024)
**Verdict: not applicable to our use case.**
GaussianObject reconstructs isolated 3D *objects* (vases, toys, bonsai) from only 4 views using
visual hull construction + a diffusion model trained to "repair" unseen object faces.
It requires masked foreground images (object segmented from background) and its diffusion prior
is trained on compact objects, not large architectural surfaces. A climbing wall IS the background —
there's nothing to segment, no unseen "back face" to inpaint, and the diffusion model would not
generalise to a flat textured plane. Not worth pursuing.

### InstantSplat++ (phai-lab/InstantSplatPP)
**Verdict: not a new algorithm — but 2D-GS mode is worth a future experiment.**
InstantSplat++ is an extension of the exact same MASt3R → 3DGS pipeline we already run.
The main additions are: support for **2D Gaussian Splatting** and Mip-Splatting as drop-in
replacements for the standard 3D-GS trainer, plus optional VGGT/MapAnything priors.

2D-GS (Huang et al., SIGGRAPH 2024) uses flat disc-shaped Gaussians instead of 3D ellipsoids.
For near-planar scenes like a climbing wall, 2D-GS captures the flat surface geometry more
accurately and produces fewer floaters perpendicular to the wall. This *could* directly improve
our results. Swapping in the 2D-GS trainer from InstantSplat++ is a contained change (replace
train.py + gaussian_renderer) with meaningful upside for our planar use case.
- [ ] Try 2D-GS trainer from InstantSplat++ as a drop-in for train.py on a fpinka scene.
