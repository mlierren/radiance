"""[Step 3] Foreground mask → RGBA. Remove the background, keep only the subject as alpha.

Keying modes per background type:
  - black: black background → luminance threshold (mean(RGB) > th, default th=15)
  - gray : solid/gray background → distance from background color (dist > th, default th=35;
           background color = average of the four corners)
  - green: green chroma-key → green excess (foreground where G - max(R,B) < th, default th=40)
           + despill: clamp foreground G to <= max(R,B) → removes green halo on hair/edges
  - auto : auto-detect from corner color (default). green-dominant→green / dark→black / else→gray.
  - hex  : --bg-hex RRGGBB sets the background color explicitly → key by distance to it (same
           formula as gray, only the color is manual). Useful for real workflows with a fixed
           chroma-key color like studio specs. Despill auto-ON if the color is green-dominant.
Common post-processing: fill holes → keep largest connected component → closing. Output is RGBA PNG.

Usage:
  python src/make_masks.py <frames_dir> <out_rgba_dir>
                           [--bg auto|black|gray|green] [--th N] [--frames a:b] [--no-despill]
  If --frames is omitted, the range from Step 2's <frames_dir>/../trim.json is applied
  automatically (organic pipeline link: no copying numbers by hand). An explicit --frames wins.
"""
import argparse, json, os
import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def load_trim_slice(frames_dir):
    """Read Step 2's <frames_dir>/../trim.json automatically and return (start, end).

    Organic pipeline link: inherits the analyze result without passing --frames by hand.
    Missing → (None, None) → use all frames.
    """
    p = os.path.join(os.path.dirname(frames_dir.rstrip('/')), "trim.json")
    if not os.path.exists(p):
        return None, None
    with open(p) as fh:
        t = json.load(fh)
    return int(t["start"]), int(t["end"])


def detect_bg(img):
    """Average color of the four 20x20 corners and whether it is dark."""
    c = np.concatenate([img[:20, :20].reshape(-1, 3), img[:20, -20:].reshape(-1, 3),
                        img[-20:, :20].reshape(-1, 3), img[-20:, -20:].reshape(-1, 3)])
    bg = c.mean(0)
    return bg, float(bg.mean())


def hex_to_rgb(s):
    """'#00B140' / '00b140' → (R,G,B) float. For manual background color."""
    s = s.lstrip("#").strip()
    if len(s) != 6:
        raise ValueError(f"--bg-hex must be a 6-digit RRGGBB hex: {s!r}")
    return tuple(float(int(s[i:i + 2], 16)) for i in (0, 2, 4))


def green_excess(img):
    """Green excess = G - max(R,B). Larger positive = greener background (chroma-key)."""
    return img[:, :, 1] - np.maximum(img[:, :, 0], img[:, :, 2])


def despill_green(img):
    """Remove green spill: clamp each pixel's G to <= max(R,B) (removes green edge tint)."""
    out = img.copy()
    out[:, :, 1] = np.minimum(out[:, :, 1], np.maximum(out[:, :, 0], out[:, :, 2]))
    return out


def mask_of(img, mode, bg, th):
    if mode == "black":
        m = img.mean(2) > th
    elif mode == "green":
        m = green_excess(img) < th                 # foreground = pixels that are not green-dominant
    else:  # gray
        m = np.linalg.norm(img - bg, axis=2) > th
    m = ndi.binary_fill_holes(m)
    lbl, n = ndi.label(m)
    if n > 0:
        sizes = ndi.sum(np.ones_like(lbl), lbl, range(1, n + 1))
        m = lbl == (1 + int(np.argmax(sizes)))     # largest connected component only
    m = ndi.binary_closing(m, iterations=2)
    m = ndi.binary_fill_holes(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--bg", choices=["auto", "black", "gray", "green"], default="auto")
    ap.add_argument("--bg-hex", default=None,
                    help="set background color as RRGGBB hex (e.g. 00B140). Key by distance to it "
                         "(overrides auto). Despill auto-ON if green-dominant.")
    ap.add_argument("--th", type=float, default=None, help="threshold (default: black=15, gray/hex=35, green=40)")
    ap.add_argument("--frames", default=None, help="valid frame slice, e.g. 0:214 (Step 2 result)")
    ap.add_argument("--no-despill", action="store_true", help="disable despill in green mode")
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.frames_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    if args.frames:                                       # explicit slice wins
        a, b = args.frames.split(":")
        files = files[int(a or 0):int(b) if b else None]
        print(f"frame slice (manual): {args.frames} → {len(files)} frames")
    else:                                                 # else apply Step 2's trim.json automatically
        s, e = load_trim_slice(args.frames_dir)
        if s is not None:
            files = files[s:e]
            print(f"frame slice (trim.json auto): {s}:{e} → {len(files)} frames")
        else:
            print(f"no trim.json → using all {len(files)} frames")
    os.makedirs(args.out_dir, exist_ok=True)

    # decide mode/threshold (based on first frame)
    img0 = np.asarray(Image.open(os.path.join(args.frames_dir, files[0])).convert("RGB"), np.float32)
    bg, bg_lum = detect_bg(img0)
    if args.bg_hex:                                       # manual background color wins
        bg = np.array(hex_to_rgb(args.bg_hex), np.float32)
        bg_lum = float(bg.mean())
        mode = "hex"                                      # use mask_of's distance keying (else branch)
    bg_gex = float(bg[1] - max(bg[0], bg[2]))
    if args.bg_hex:
        pass                                             # mode already set above
    elif args.bg != "auto":
        mode = args.bg
    else:
        mode = "green" if bg_gex > 20 else ("black" if bg_lum < 25 else "gray")
    defaults = {"black": 15.0, "gray": 35.0, "green": 40.0, "hex": 35.0}
    th = args.th if args.th is not None else defaults[mode]
    # despill: auto-ON for green mode, or when a hex background is green-dominant (not for blue/other)
    despill = (mode == "green" or (mode == "hex" and bg_gex > 20)) and not args.no_despill
    src = f"hex #{args.bg_hex.lstrip('#').upper()}" if args.bg_hex else "corner auto"
    print(f"background mode: {mode} ({src}, color {np.round(bg,1).tolist()}, luma {bg_lum:.1f}, "
          f"green-excess {bg_gex:.1f}), threshold {th}" + (", despill ON" if despill else ""))

    fracs = []
    for f in files:
        img = np.asarray(Image.open(os.path.join(args.frames_dir, f)).convert("RGB"), np.float32)
        m = mask_of(img, mode, bg, th)
        rgb = despill_green(img) if despill else img
        fracs.append(float(m.mean()))
        rgba = np.dstack([rgb, (m * 255)]).astype(np.uint8)
        Image.fromarray(rgba, "RGBA").save(os.path.join(args.out_dir, f))
    print(f"saved {len(files)} RGBA frames → {args.out_dir}  (mean foreground fraction {np.mean(fracs):.3f})")
    print("⚠️ Eyeball a few masks for quality (hair edges / missing holes / residual green, etc.).")


if __name__ == "__main__":
    main()
