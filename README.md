# Radiance — Bible Worlds · Unified 3DGS Asset PoC

Turn **one turntable MP4 of Peter's head** into a standard **3D Gaussian Splatting** asset
(`.ply`) that opens directly in viewers like SuperSplat. Part of the **Bible Worlds** program
(historically and scripturally accurate 3D environments for media production).

This README summarizes the pipeline. The full step-by-step document is
[`docs/head_mp4_to_3dgs_pipeline.org`](./docs/head_mp4_to_3dgs_pipeline.org).

## Key idea

A turntable is a **known circular orbit**, so we *assign* camera poses analytically instead of
*estimating* them with COLMAP (which fails on a background-removed, low-texture subject). We then
train plain 3DGS and apply a post-hoc proportion fix.

```
mp4 → frames → masks → analytic poses (★rotation direction) → 3DGS → Z-rescale → .ply
```

## Pipeline

| # | Step | Command | Output |
|---|------|---------|--------|
| 1 | Extract frames | `ffmpeg -i turntable.mp4 frames_raw/f_%04d.png` | `frames_raw/*.png` |
| 2 | Analyze rotation, trim still tail | `python src/analyze_turntable.py frames_raw` | valid range + `_contact.png` |
| 3 | Foreground masks → RGBA | `python src/make_masks.py frames_raw rgba --bg auto` | `rgba/*.png` |
| 4 | **Analytic turntable poses** ★ | `python src/poses_turntable.py rgba out --direction -1` | `transforms_{train,test}.json` |
| 5 | 3DGS training (free, no prior) | stock r3dg `train.py`, 30k iters | `point_cloud.ply` + `chkpnt30000.pth` |
| 6 | Post-hoc Z-rescale | `python src/zrescale_ply.py in.ply out.ply <ZF>` | `*_zcorr.ply` (final) |
| 7 | Inspect / render | `python src/render_turntable.py out.ply frames 300 0 800` | demo `*.mp4` |

**Critical config:** `direction=-1`, `elevation=0`, `radius=3`, `fov=40`, `test-every=8`.
A wrong rotation `direction` keeps only the front sharp and smears/concaves everything else.

## Software & hardware

- **Core engine:** Relightable3DGaussian (r3dg) — 3DGS-native, and decomposes PBR material
  (albedo/roughness/normal) + environment light in the same pipeline, so one asset carries
  geometry → relight → edit → compose.
- **Stack:** PyTorch 1.12.1+cu116, CUDA extensions (r3dg-rasterization, simple-knn, bvh,
  nvdiffrast), numpy/scipy/Pillow(<10)/opencv/plyfile/ffmpeg.
- **GPU:** prebuilt extensions target `sm_80/86`, so use an **Ampere** GPU (RTX 3090 / A5000 /
  A6000 / A40 / A100). Avoid 4090/L40 (sm_89), H100 (sm_90), T4/2080 (sm_75) — they recompile.
- **License:** r3dg is research code (non-commercial, INRIA-rasterizer-based) → **PoC / internal
  demo only**. Production target: reimplement on gsplat (Apache 2.0).

## Environment setup

`scripts/runpod_launcher.py` bootstraps the full stack onto a RunPod network volume once
(env, repos, CUDA extensions, Jupyter kernel). Interactive/visual work runs locally; batch
compute runs on the Pod.

## Lessons learned

- **Rotation direction is the single most important setting.** If unsure, train both signs and
  keep the higher test-PSNR / lower sliver ratio.
- **Validate geometry, not PSNR** — a high PSNR can hide a badly distorted reconstruction.
- **Don't fight the optimizer with strong priors** — fix the data/poses, train plainly, correct
  proportions afterward.
- **Single-elevation capture cannot fix vertical scale** — the result is stretched by design;
  correct it post-hoc with a silhouette-matched Z-rescale.

## Repository layout

| Path | Contents |
|------|----------|
| `src/` | per-step pipeline scripts (analyze, masks, poses, zrescale, render, glb) |
| `scripts/runpod_launcher.py` | RunPod environment bootstrap / lifecycle |
| `docs/` | `head_mp4_to_3dgs_pipeline.org` (full pipeline doc) |
| `third_party/` | stock Relightable3DGaussian (not committed; cloned per-environment) |

Large artifacts (`data/`, `outputs/`, `*.ply`, `*.mp4`, `experiments/demo/`) are git-ignored and
live on the Pod network volume.
