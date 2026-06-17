"""[Step 3] 전경 마스크 → RGBA. 배경을 지우고 인물만 알파(투명도)로 남긴다.

배경 종류에 맞춰 키잉 방식 지원:
  - black: 검정 배경 → 휘도 임계 (mean(RGB) > th,  기본 th=15)
  - gray : 단색/회색 배경 → 배경색과의 거리 (dist > th, 기본 th=35; 배경색은 네 모서리 평균)
  - green: 녹색 크로마키 → 녹색 우세도 (G - max(R,B) < th 면 전경, 기본 th=40)
           + 디스필: 전경의 G를 max(R,B) 이하로 클램프 → 머리카락/엣지 초록빛 헤일로 제거
  - auto : 모서리 색으로 자동 판별(기본). 녹색 우세→green / 어두움→black / 그 외→gray.
공통 후처리: 구멍 메우기 → 최대 연결성분만 유지 → closing. 결과는 RGBA PNG.

사용법:
  python src/make_masks.py <frames_dir> <out_rgba_dir>
                           [--bg auto|black|gray|green] [--th N] [--frames a:b] [--no-despill]
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


def green_excess(img):
    """녹색 우세도 = G - max(R,B). 양수 클수록 녹색 배경(크로마키)."""
    return img[:, :, 1] - np.maximum(img[:, :, 0], img[:, :, 2])


def despill_green(img):
    """녹색 스필 제거: 각 픽셀의 G를 max(R,B) 이하로 클램프(엣지 초록빛 제거)."""
    out = img.copy()
    out[:, :, 1] = np.minimum(out[:, :, 1], np.maximum(out[:, :, 0], out[:, :, 2]))
    return out


def mask_of(img, mode, bg, th):
    if mode == "black":
        m = img.mean(2) > th
    elif mode == "green":
        m = green_excess(img) < th                 # 전경 = 녹색 우세가 아닌 픽셀
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
    ap.add_argument("--bg", choices=["auto", "black", "gray", "green"], default="auto")
    ap.add_argument("--th", type=float, default=None, help="임계값(기본: black=15, gray=35, green=40)")
    ap.add_argument("--frames", default=None, help="유효 프레임 슬라이스, 예: 0:214 (Step 2 결과)")
    ap.add_argument("--no-despill", action="store_true", help="녹색 모드에서 디스필 비활성화")
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.frames_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    if args.frames:
        a, b = args.frames.split(":")
        files = files[int(a or 0):int(b) if b else None]
    os.makedirs(args.out_dir, exist_ok=True)

    # 모드/임계 결정 (첫 프레임 기준)
    img0 = np.asarray(Image.open(os.path.join(args.frames_dir, files[0])).convert("RGB"), np.float32)
    bg, bg_lum = detect_bg(img0)
    bg_gex = float(bg[1] - max(bg[0], bg[2]))
    if args.bg != "auto":
        mode = args.bg
    else:
        mode = "green" if bg_gex > 20 else ("black" if bg_lum < 25 else "gray")
    defaults = {"black": 15.0, "gray": 35.0, "green": 40.0}
    th = args.th if args.th is not None else defaults[mode]
    despill = (mode == "green") and not args.no_despill
    print(f"배경 모드: {mode} (모서리 평균색 {np.round(bg,1).tolist()}, 휘도 {bg_lum:.1f}, "
          f"녹색우세 {bg_gex:.1f}), 임계 {th}" + (", 디스필 ON" if despill else ""))

    fracs = []
    for f in files:
        img = np.asarray(Image.open(os.path.join(args.frames_dir, f)).convert("RGB"), np.float32)
        m = mask_of(img, mode, bg, th)
        rgb = despill_green(img) if despill else img
        fracs.append(float(m.mean()))
        rgba = np.dstack([rgb, (m * 255)]).astype(np.uint8)
        Image.fromarray(rgba, "RGBA").save(os.path.join(args.out_dir, f))
    print(f"RGBA {len(files)}장 저장 → {args.out_dir}  (평균 전경 비율 {np.mean(fracs):.3f})")
    print("⚠️ 마스크 품질을 몇 장 눈으로 확인할 것(머리카락 경계/구멍 누락/초록빛 잔여 등).")


if __name__ == "__main__":
    main()
