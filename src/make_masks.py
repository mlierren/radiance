"""[Step 3] 전경 마스크 → RGBA. 배경을 지우고 인물만 알파(투명도)로 남긴다.

배경 종류에 맞춰 두 방식 지원:
  - black: 검정 배경 → 휘도 임계 (mean(RGB) > th,  기본 th=15)
  - gray : 단색/회색 배경 → 배경색과의 거리 (dist > th, 기본 th=35; 배경색은 네 모서리 평균)
  - auto : 모서리 색으로 자동 판별(기본). 모서리가 어두우면 black, 아니면 gray.
공통 후처리: 구멍 메우기 → 최대 연결성분만 유지 → closing. 결과는 RGBA PNG.

사용법:
  python experiments/make_masks.py <frames_dir> <out_rgba_dir> [--bg auto|black|gray] [--th N] [--frames a:b]
"""
import argparse, os
import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def detect_bg(img):
    """네 모서리(20x20) 평균색과 어두움 여부."""
    c = np.concatenate([img[:20, :20].reshape(-1, 3), img[:20, -20:].reshape(-1, 3),
                        img[-20:, :20].reshape(-1, 3), img[-20:, -20:].reshape(-1, 3)])
    bg = c.mean(0)
    return bg, float(bg.mean())


def mask_of(img, mode, bg, th):
    if mode == "black":
        m = img.mean(2) > th
    else:  # gray
        m = np.linalg.norm(img - bg, axis=2) > th
    m = ndi.binary_fill_holes(m)
    lbl, n = ndi.label(m)
    if n > 0:
        sizes = ndi.sum(np.ones_like(lbl), lbl, range(1, n + 1))
        m = lbl == (1 + int(np.argmax(sizes)))     # 최대 연결성분만
    m = ndi.binary_closing(m, iterations=2)
    m = ndi.binary_fill_holes(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--bg", choices=["auto", "black", "gray"], default="auto")
    ap.add_argument("--th", type=float, default=None, help="임계값(기본: black=15, gray=35)")
    ap.add_argument("--frames", default=None, help="유효 프레임 슬라이스, 예: 0:214 (Step 2 결과)")
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.frames_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    if args.frames:
        a, b = args.frames.split(":")
        files = files[int(a or 0):int(b) if b else None]
    os.makedirs(args.out_dir, exist_ok=True)

    # 모드/임계 결정 (첫 프레임 기준)
    img0 = np.asarray(Image.open(os.path.join(args.frames_dir, files[0])).convert("RGB"), np.float32)
    bg, bg_lum = detect_bg(img0)
    mode = args.bg if args.bg != "auto" else ("black" if bg_lum < 25 else "gray")
    th = args.th if args.th is not None else (15.0 if mode == "black" else 35.0)
    print(f"배경 모드: {mode} (모서리 평균색 {np.round(bg,1).tolist()}, 휘도 {bg_lum:.1f}), 임계 {th}")

    fracs = []
    for f in files:
        img = np.asarray(Image.open(os.path.join(args.frames_dir, f)).convert("RGB"), np.float32)
        m = mask_of(img, mode, bg, th)
        fracs.append(float(m.mean()))
        rgba = np.dstack([img, (m * 255)]).astype(np.uint8)
        Image.fromarray(rgba, "RGBA").save(os.path.join(args.out_dir, f))
    print(f"RGBA {len(files)}장 저장 → {args.out_dir}  (평균 전경 비율 {np.mean(fracs):.3f})")
    print("⚠️ 마스크 품질을 몇 장 눈으로 확인할 것(머리카락 경계/구멍 누락 등).")


if __name__ == "__main__":
    main()
