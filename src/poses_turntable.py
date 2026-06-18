"""Turntable pose synthesis for background-removed rotating captures.

Rationale: a background-removed turntable is the
"rotating object on transparent background" = NeRF-synthetic capture scenario.
COLMAP feature points vanish without background, so analytic circular poses are
the most robust choice. Relightable3DGaussian's blender loader
(scene/dataset_readers.py :: readCamerasFromTransforms) is the consumer.

Loader contract we adhere to (verified against the repo):
  * file_path is written WITHOUT extension; the loader appends `extension`
    (default ".png") via os.path.join(path, frame["file_path"] + extension).
  * transform_matrix is camera-to-world in OpenGL/Blender axes (X right, Y up,
    Z back); the loader converts to COLMAP with c2w[:3, 1:3] *= -1.
  * Masking is via the RGBA alpha channel (image[:, :, 3]); there is no separate
    mask directory on the blender path. Prepare RGBA PNGs (see make_rgba_dataset).
  * The loader reads transforms_train.json (always) and transforms_test.json
    (when --eval). We emit both, holding out every Nth view for NVS sanity.
"""
import json
import math
import os

import numpy as np


def look_at_c2w(eye, target=(0, 0, 0), up=(0, 0, 1)):
    """Camera-to-world 4x4 in OpenGL/Blender axes (cols = right, up, back, eye)."""
    eye, target, up = map(lambda v: np.array(v, float), (eye, target, up))
    f = (target - eye)
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = r, u, -f, eye
    return c2w


def turntable_transforms(image_dir, out_dir, rel_subdir="rgba", radius=3.0,
                         elevation_deg=0.0, fov_x_deg=40.0, img_ext=".png",
                         test_every=8, direction=-1):
    """Write transforms_train.json + transforms_test.json for a turntable.

    Angles are derived from each frame's ABSOLUTE index over the full rotation
    (theta = direction * 2*pi*i/N), so the held-out test views keep their true
    pose. Frames with index % test_every == 0 go to the test split.

    CRITICAL (confirmed 2026-06-14): `direction` MUST match the subject's real
    rotation. For the Peter turntable captures the correct value is **-1**.
    A wrong sign makes only the front sharp and smears/concaves everything else
    (free-3DGS test PSNR drops ~3, gaussian slivers explode). If unsure, train
    free with both signs and keep the one with higher PSNR / lower axis-ratio.

    `elevation_deg=0` matches an eye-level turntable. NOTE: single-elevation
    capture leaves the recon vertically stretched (depth ambiguity) — correct it
    post-hoc with a silhouette-matched Z-rescale (see experiments/zrescale_ply.py).

    Returns (n_total, n_train, n_test).
    """
    files = sorted(f for f in os.listdir(image_dir) if f.endswith(img_ext))
    n = len(files)
    el = math.radians(elevation_deg)
    train, test = [], []
    for i, fn in enumerate(files):
        th = direction * 2 * math.pi * i / n
        eye = (radius * math.cos(th) * math.cos(el),
               radius * math.sin(th) * math.cos(el),
               radius * math.sin(el))
        frame = {
            # NO extension: the loader appends it.
            "file_path": os.path.join(".", rel_subdir, os.path.splitext(fn)[0]),
            "transform_matrix": look_at_c2w(eye).tolist(),
        }
        (test if (test_every and i % test_every == 0) else train).append(frame)

    cam_angle_x = math.radians(fov_x_deg)
    os.makedirs(out_dir, exist_ok=True)
    for name, frames in (("transforms_train.json", train),
                         ("transforms_test.json", test)):
        with open(os.path.join(out_dir, name), "w") as fh:
            json.dump({"camera_angle_x": cam_angle_x, "frames": frames}, fh, indent=2)
    return n, len(train), len(test)


def make_rgba_dataset(image_dir, out_subdir, th=12):
    """Composite RGB frames + luminance mask into RGBA PNGs the loader expects.

    alpha = 255 where mean(RGB) > th else 0 (clean black background => SAM-free).
    Returns (n_written, mean_foreground_fraction).
    """
    from PIL import Image
    os.makedirs(out_subdir, exist_ok=True)
    files = sorted(f for f in os.listdir(image_dir) if f.endswith(".png"))
    fracs = []
    for fn in files:
        rgb = np.array(Image.open(os.path.join(image_dir, fn)).convert("RGB"))
        alpha = ((rgb.mean(-1) > th) * 255).astype("uint8")
        rgba = np.dstack([rgb, alpha])
        Image.fromarray(rgba, "RGBA").save(os.path.join(out_subdir, fn))
        fracs.append((alpha > 0).mean())
    return len(files), float(np.mean(fracs))


# NOTE: make_rgba_dataset above is the legacy luminance-only masker (black bg).
# The current Step-3 masker is experiments/make_masks.py (black/gray/auto). Kept here
# only for the original NeRF-synthetic path.


if __name__ == "__main__":
    # [Step 4] CLI: turntable analytic camera poses → transforms_train/test.json
    #   python src/poses_turntable.py <rgba_dir> <out_dir> [--direction -1] [--elevation 0] ...
    import argparse
    ap = argparse.ArgumentParser(description="Turntable analytic poses → transforms_*.json")
    ap.add_argument("image_dir", help="RGBA frames dir (e.g. data/peter_face/rgba)")
    ap.add_argument("out_dir", help="transforms_*.json output location (e.g. data/peter_face)")
    ap.add_argument("--rel-subdir", default="rgba", help="file_path prefix in json (image folder name)")
    ap.add_argument("--radius", type=float, default=3.0)
    ap.add_argument("--elevation", type=float, default=0.0)
    ap.add_argument("--fov", type=float, default=40.0)
    ap.add_argument("--test-every", type=int, default=8)
    ap.add_argument("--direction", type=int, default=-1, choices=[-1, 1],
                    help="rotation direction (critical). Peter captures = -1 (wrong sign collapses shape)")
    a = ap.parse_args()
    n, ntr, nte = turntable_transforms(
        a.image_dir, a.out_dir, rel_subdir=a.rel_subdir, radius=a.radius,
        elevation_deg=a.elevation, fov_x_deg=a.fov, test_every=a.test_every, direction=a.direction)
    print(f"transforms written: {n} frames → train {ntr} / test {nte}  "
          f"(direction={a.direction}, elev={a.elevation}, radius={a.radius}, fov={a.fov})")
