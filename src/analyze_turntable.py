"""[Step 2] Turntable rotation-structure analysis + automatic trim of still/duplicate ranges.

Takes the extracted-frames folder, *deterministically* computes the valid one-revolution
range, and exports a machine-readable =trim.json= (the next step, make_masks, reads it
automatically). Human/Claude eyes only do 'post-hoc verification' via the saved montage
(the script makes the decision).

What it detects:
  1) leading-still : still (frontal hold) range at the start — must be removed.
     (Otherwise the same front is trained at 0°,Δ,2Δ… different angles → front smear/ghost.)
  2) trailing-still: still tail at the end — removed.
  3) wrap-closure  : the point that returns to the front after one revolution (≈360°) —
     removed from that frame on.
     (Analytic poses use the [0°,360°) uniform-spacing convention, so a duplicated front = double layer/ghost.)

Still detection is *adaptive* (no fixed threshold). For adjacent ZNCC correlation adj:
  m = median(adj)                      # typical 'mid-rotation' correlation (robust by majority)
  still ⇔ (1-adj) < alpha*(1-m)        # still if the change is below alpha of the usual rotation change
It measures contrast, so it adapts even when rotation speed / detail / compression noise vary per clip.
(A fixed threshold breaks both ways: too tight mistakes slow rotation for still and over-trims;
 too loose misses noisy stills and leaves a ghost.)

Usage:
  python src/analyze_turntable.py <frames_dir> [--alpha 0.25] [--wrap-th 0.98]
                                  [--abs-still-th 0.9999] [--montage out.png]
Outputs:
  <frames_dir>/../trim.json   {"start":S,"end":E,...}  (end is exclusive = slice S:E)
  <frames_dir>/../_contact.png  verification montage (DROP/KEEP/DUP cells marked)
  one line 'TRIM S:E' to stdout (for parsing).
"""
import argparse, json, os
import numpy as np
from PIL import Image, ImageDraw


def load_gray(frames_dir, size=(160, 90)):
    files = sorted(f for f in os.listdir(frames_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    g = np.stack([np.asarray(Image.open(os.path.join(frames_dir, f)).convert("L").resize(size), np.float32)
                  for f in files])
    return files, g


def zncc_matrix(g):
    """Flatten → mean-subtract → normalize. corr(i,j)=gf[i]·gf[j] (Pearson = ZNCC)."""
    n = len(g)
    gf = g.reshape(n, -1)
    gf = gf - gf.mean(1, keepdims=True)
    gf = gf / (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-8)
    return gf


def adaptive_still_th(adj, alpha, abs_floor=None):
    """Derive the still threshold adaptively from the 'mid-rotation correlation median'.

    m = median(adj) = typical adjacent correlation during rotation (rotation frames dominate,
    so the median sits there).
    still_th = 1 - alpha*(1-m).  If abs_floor is given, clamp to at least that (lower bound).
    Returns (still_th, m).
    """
    m = float(np.median(adj))
    gap = max(1.0 - m, 1e-5)                              # contrast span between still (≈1) and rotation (m)
    still_th = 1.0 - alpha * gap
    if abs_floor is not None:
        still_th = max(still_th, abs_floor)
    return still_th, m


def detect_leading_still(adj, still_th):
    """Count of consecutive still pairs (adj>still_th) at the start = first moving-frame index (= keep start)."""
    k = 0
    while k < len(adj) and adj[k] > still_th:
        k += 1
    return k                                             # frames 0..k-1 are the frontal still (duplicate) → drop; keep from frame k


def detect_trailing_still(adj, still_th):
    """Last moving-frame index after removing the still tail (adj>still_th) at the end."""
    last = len(adj)                                      # = n-1
    while last > 0 and adj[last - 1] > still_th:
        last -= 1
    return last


def detect_wrap_closure(c0, wrap_th):
    """The 'one-revolution closure' point that returns to frame 0 (front).

    In the back half, the first frame where c0 rises past wrap_th again = frontal re-match (≈360°).
    If none, return the back-half max-correlation position for reference only. If strong=True,
    trim from that frame on.
    Returns (idx, corr, strong).
    """
    n = len(c0)
    lo = max(2, n // 2)
    back = np.arange(lo, n)
    cb = c0[back]
    strong = back[cb >= wrap_th]
    if len(strong):
        idx = int(strong[0])
        return idx, float(c0[idx]), True
    idx = int(back[int(np.argmax(cb))])
    return idx, float(c0[idx]), False


def save_montage(frames_dir, files, start, end, out, wrap_idx=None, cols=5, cell=(200, 200)):
    """Verification montage: uniform samples of the keep range + DROP (leading still f0) + KEEP start + DUP (wrap) cells."""
    last = end - 1
    idx = list(range(start, last + 1, max(1, (last - start + 1) // 12))) + [last]
    labels = {i: (f"f{i}", (255, 255, 0)) for i in idx}
    if start > 0:                                        # representative of the dropped leading still (first frame)
        labels[0] = ("DROP lead f0", (255, 80, 80)); idx.append(0)
        labels[start] = (f"KEEP start f{start}", (80, 255, 80))
    if wrap_idx is not None and 0 <= wrap_idx < len(files):
        labels[wrap_idx] = (f"DUP f{wrap_idx} (drop)", (255, 80, 80)); idx.append(wrap_idx)
    idx = sorted(set(idx))
    W, H = cell
    rows = (len(idx) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * W, rows * H), (30, 30, 30))
    d = ImageDraw.Draw(sheet)
    for k, i in enumerate(idx):
        im = Image.open(os.path.join(frames_dir, files[i])).convert("RGB").resize((W, H))
        r, c = divmod(k, cols)
        sheet.paste(im, (c * W, r * H))
        text, color = labels[i]
        d.text((c * W + 4, r * H + 4), text, fill=color)
    sheet.save(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames_dir")
    ap.add_argument("--montage", default=None, help="verification montage path (default: <frames_dir>/../_contact.png)")
    ap.add_argument("--trim-json", default=None, help="trim-range output path (default: <frames_dir>/../trim.json)")
    ap.add_argument("--alpha", type=float, default=0.25,
                    help="still-detection sensitivity. still ⇔ (1-adj) < alpha*(1-median(adj)). Smaller = stricter")
    ap.add_argument("--abs-still-th", type=float, default=None,
                    help="absolute lower bound for the still threshold (optional). Combined with the adaptive value via max")
    ap.add_argument("--wrap-th", type=float, default=0.98,
                    help="correlation threshold vs frame 0 to count as frontal re-match (one-revolution closure)")
    args = ap.parse_args()

    files, g = load_gray(args.frames_dir)
    n = len(files)
    gf = zncc_matrix(g)
    adj = (gf[:-1] * gf[1:]).sum(1)                      # adjacent-frame correlation
    c0 = gf @ gf[0]                                      # correlation vs frame 0 (front)

    still_th, m = adaptive_still_th(adj, args.alpha, args.abs_still_th)
    lead = detect_leading_still(adj, still_th)           # keep-start index
    last_still = detect_trailing_still(adj, still_th)    # last moving frame after removing the still tail
    wrap_idx, wrap_corr, wrap_strong = detect_wrap_closure(c0, args.wrap_th)

    end = last_still + 1                                 # exclusive
    if wrap_strong:
        end = min(end, wrap_idx)                         # only up to just before frontal re-match = [0°,360°)
    start = lead
    if start >= end:                                     # safety guard (degenerate)
        start, end = 0, n

    n_keep = end - start
    print(f"total frames: {n}")
    print(f"[adaptive threshold] rotation median m={m:.5f} → still threshold still_th={still_th:.5f} (alpha={args.alpha})")
    if lead > 0:
        print(f"[leading still] frame 0..{lead - 1} frontal hold (duplicate) → removed. keep start = frame {lead}.")
    else:
        print(f"[leading still] none (rotating from the start).")
    print(f"[still tail] last moving frame {last_still} (then {last_still + 1}..{n - 1} still)")
    if wrap_strong:
        print(f"[wrap closure] ⚠️ frame {wrap_idx} re-matches the front (corr={wrap_corr:.4f} ≥ {args.wrap_th}) "
              f"→ removed from that frame on.")
    else:
        print(f"[wrap closure] no strong frontal re-match (back-half max corr={wrap_corr:.4f} @f{wrap_idx} < {args.wrap_th}).")
    print(f"  ✅ valid range: frame {start}..{end - 1}  (slice {start}:{end}, {n_keep} frames)")
    print(f"[note] lowest correlation (back ≈180°): frame {int(c0.argmin())}/{n} — may be inaccurate due to back-view self-similarity.")

    trim_path = args.trim_json or os.path.join(os.path.dirname(args.frames_dir.rstrip('/')), "trim.json")
    with open(trim_path, "w") as fh:
        json.dump({"start": int(start), "end": int(end), "n_total": int(n), "n_keep": int(n_keep),
                   "leading_still": int(lead), "trailing_last_move": int(last_still),
                   "wrap_idx": int(wrap_idx), "wrap_strong": bool(wrap_strong),
                   "wrap_corr": round(float(wrap_corr), 4),
                   "moving_median": round(m, 5), "still_th": round(still_th, 5),
                   "alpha": args.alpha}, fh, indent=2)
    print(f"trim.json saved: {trim_path}")

    out = args.montage or os.path.join(os.path.dirname(args.frames_dir.rstrip('/')), "_contact.png")
    save_montage(args.frames_dir, files, start, end, out,
                 wrap_idx=(wrap_idx if wrap_strong else None))
    print(f"verification montage saved: {out}")
    print(f"TRIM {start}:{end}")                          # one line for parsing


if __name__ == "__main__":
    main()
