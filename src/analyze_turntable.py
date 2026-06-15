"""[Step 2] 턴테이블 회전 구조 분석 + 정지/중복 꼬리 탐지.

추출된 프레임 폴더를 받아:
  1) 인접 프레임 상관(ZNCC)으로 끝의 '정지 꼬리'를 자동 탐지 → 유효 1회전 범위 출력
  2) 프레임0 대비 상관(c0)으로 후면(≈180°) 위치를 출력(균일 회전 sanity check)
  3) 확인용 몽타주 PNG 저장(반드시 눈으로 정면→후면→정면 확인)

사용법:
  python experiments/analyze_turntable.py <frames_dir> [--montage out.png] [--stat-th 0.9999]
"""
import argparse, os
import numpy as np
from PIL import Image, ImageDraw


def load_gray(frames_dir, size=(160, 90)):
    files = sorted(f for f in os.listdir(frames_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    g = np.stack([np.asarray(Image.open(os.path.join(frames_dir, f)).convert("L").resize(size), np.float32)
                  for f in files])
    return files, g


def zncc_matrix(g):
    """평탄화 → 평균제거 → 정규화. corr(i,j)=gf[i]·gf[j] (피어슨=ZNCC)."""
    n = len(g)
    gf = g.reshape(n, -1)
    gf = gf - gf.mean(1, keepdims=True)
    gf = gf / (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-8)
    return gf


def save_montage(frames_dir, files, last, out, cols=5, cell=(200, 200)):
    idx = list(range(0, last + 1, max(1, (last + 1) // 14))) + [last]
    idx = sorted(set(i for i in idx if i <= last))
    W, H = cell
    rows = (len(idx) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * W, rows * H), (30, 30, 30))
    d = ImageDraw.Draw(sheet)
    for k, i in enumerate(idx):
        im = Image.open(os.path.join(frames_dir, files[i])).convert("RGB").resize((W, H))
        r, c = divmod(k, cols)
        sheet.paste(im, (c * W, r * H))
        d.text((c * W + 4, r * H + 4), f"f{i}", fill=(255, 255, 0))
    sheet.save(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames_dir")
    ap.add_argument("--montage", default=None, help="확인용 몽타주 저장 경로(기본: <frames_dir>/../_contact.png)")
    ap.add_argument("--stat-th", type=float, default=0.9999, help="정지로 간주할 인접프레임 상관 임계값")
    args = ap.parse_args()

    files, g = load_gray(args.frames_dir)
    n = len(files)
    gf = zncc_matrix(g)
    adj = (gf[:-1] * gf[1:]).sum(1)                      # 인접 프레임 상관
    c0 = gf @ gf[0]                                      # 프레임0 대비 상관

    last = n - 1
    while last > 0 and adj[last - 1] > args.stat_th:     # 뒤에서부터 정지 구간 제거
        last -= 1

    print(f"총 프레임: {n}")
    print(f"유효 1회전: frame 0..{last}  (정지/중복 {last+1}..{n-1} 제거 → {last+1}장 사용)")
    print(f"[참고] 최저상관 위치: frame {int(c0.argmin())}/{n} — 뒤통수/후면 자기유사로 부정확할 수 있음.")
    print(f"       균일 회전 여부는 반드시 아래 몽타주로 직접 확인할 것(정면→후면→정면).")

    out = args.montage or os.path.join(os.path.dirname(args.frames_dir.rstrip('/')), "_contact.png")
    save_montage(args.frames_dir, files, last, out)
    print(f"확인용 몽타주 저장: {out}  → 눈으로 정면→후면→정면 확인할 것")


if __name__ == "__main__":
    main()
