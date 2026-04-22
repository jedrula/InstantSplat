# Setup Notes

## CUDA 12.9 / GCC compile fix

When building `submodules/diff-gaussian-rasterization` with CUDA 12.9 and GCC,
you may hit:

```
error: namespace "std" has no member "uintptr_t"
error: identifier "uint32_t" is undefined
```

Fix: add `#include <cstdint>` to
`submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h`
after the existing `#include <vector>` line.

## Converting trained .ply to browser-viewable .splat

After training, convert the output PLY to the compact `.splat` binary format:

```bash
python ply2splat.py \
  output_infer/<scene>/point_cloud/iteration_1000/point_cloud.ply \
  output_infer/<scene>/<scene>.splat
```

Drop the `.splat` file onto https://antimatter15.com/splat/ or
https://playcanvas.com/supersplat/editor for interactive browser viewing.

## Example scene: climbing_section1

6 consecutive frames from a bouldering gym walkthrough video (fpinka.mp4),
covering the first wall section. Frames extracted at 1 FPS starting at 00:01.

Run the full pipeline:
```bash
bash run_infer.sh assets/examples/climbing_section1 6 output_infer/climbing_section1
```
