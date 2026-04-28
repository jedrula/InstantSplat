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
**Verdict: misleading marketing — the repo does not actually implement 2DGS.**
Inspected the actual code: both `gaussian_renderer/__init__.py` and `__init__3dgs.py` import
`diff_gaussian_rasterization` (standard 3DGS kernel). There is no `diff-surfel-rasterization`
submodule, which is the CUDA kernel that real 2D Gaussian Splatting requires. The README claim
of "2DGS and Mip-Splatting support" is not backed by the code. Not worth migrating to.

## Future quality experiments

### Real 2D Gaussian Splatting (hbb1/2d-gaussian-splatting)
2D-GS (Huang et al., SIGGRAPH 2024) uses flat disc-shaped Gaussians instead of 3D ellipsoids.
For near-planar scenes like a climbing wall, discs align with the wall surface naturally and
produce fewer floaters perpendicular to it. This is the correct implementation to try.

Integration plan (init_geo.py is unchanged — poses are backend-agnostic):
- Add `submodules/diff-surfel-rasterization` (the 2DGS CUDA kernel)
- Pull `train_2dgs.py` + `scene/gaussian_model_2dgs.py` from hbb1/2d-gaussian-splatting
- Add `--gs-type 3dgs|2dgs` flag to `video_to_splat.sh` to pick the trainer
- Expose engine choice in Vue UI alongside image size
- Risk: two CUDA submodules may conflict; may need a separate conda env

- [ ] Try real 2DGS from hbb1/2d-gaussian-splatting on a fpinka scene.
