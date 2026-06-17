"""[Step 2] 턴테이블 회전 구조 분석 + 정지/중복 꼬리 탐지.

추출된 프레임 폴더를 받아:
  1) 인접 프레임 상관(ZNCC)으로 끝의 '정지 꼬리'를 자동 탐지
  2) 프레임0(정면) 대비 상관(c0)으로 '한 바퀴 닫힘'(정면 재일치) 지점을 탐지
     → analytic pose는 [0°,360°) 균등분배 규약이므로, 정면으로 되돌아온
       프레임(360°≈0°) 및 이후는 '정면 중복'이라 반드시 제거해야 한다.
       (남기면 같은 정면이 0°와 ~357°+ 두 각도로 학습되어 정면이 두 겹/ghost.)
  3) 확인용 몽타주 PNG 저장(반드시 눈으로 정면→후면→정면 + 끝-시작 비중복 확인)

사용법:
  python src/analyze_turntable.py <frames_dir> [--montage out.png]
                                  [--stat-th 0.9999] [--wrap-th 0.98]
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


def detect_still_tail(adj, stat_th):
    """뒤에서부터 인접상관>stat_th(정지) 구간을 제거한 마지막 이동 프레임 인덱스."""
    last = len(adj)                                      # = n-1
    while last > 0 and adj[last - 1] > stat_th:
        last -= 1
    return last


def detect_wrap_closure(c0, wrap_th):
    """프레임0(정면)으로 되돌아온 '한 바퀴 닫힘' 지점을 찾는다.

    회전 후반부(back half)에서 c0(=프레임0과의 상관)가 다시 올라가 wrap_th를
    처음 넘는 프레임 = 정면 재일치(≈360°). 그런 프레임이 없으면(끝이 정면으로
    안 돌아옴) 후반부 최대상관 위치를 참고로만 반환한다.

    Returns: (idx, corr, strong)  strong=True 면 정면 중복으로 판단해 잘라야 함.
    """
    n = len(c0)
    lo = max(2, n // 2)                                  # 시작 근방은 제외하고 후반부만
    back = np.arange(lo, n)
    cb = c0[back]
    strong = back[cb >= wrap_th]
    if len(strong):
        idx = int(strong[0])                            # 가장 이른 정면 재일치
        return idx, float(c0[idx]), True
    idx = int(back[int(np.argmax(cb))])
    return idx, float(c0[idx]), False


def save_montage(frames_dir, files, last, out, dup_idx=None, cols=5, cell=(200, 200)):
    idx = list(range(0, last + 1, max(1, (last + 1) // 14))) + [last]
    idx = sorted(set(i for i in idx if i <= last))
    labels = {i: (f"f{i}", (255, 255, 0)) for i in idx}
    # 잘라낼 '정면 중복' 프레임을 함께 보여줘 첫 프레임(f0)과 눈으로 비교하게 한다.
    if dup_idx is not None and 0 <= dup_idx < len(files) and dup_idx not in labels:
        idx.append(dup_idx)
        labels[dup_idx] = (f"DUP f{dup_idx} (drop)", (255, 80, 80))
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
    ap.add_argument("--montage", default=None, help="확인용 몽타주 저장 경로(기본: <frames_dir>/../_contact.png)")
    ap.add_argument("--stat-th", type=float, default=0.9999, help="정지로 간주할 인접프레임 상관 임계값")
    ap.add_argument("--wrap-th", type=float, default=0.98,
                    help="정면 재일치(한 바퀴 닫힘)로 간주할 프레임0 대비 상관 임계값")
    args = ap.parse_args()

    files, g = load_gray(args.frames_dir)
    n = len(files)
    gf = zncc_matrix(g)
    adj = (gf[:-1] * gf[1:]).sum(1)                      # 인접 프레임 상관
    c0 = gf @ gf[0]                                      # 프레임0 대비 상관

    last_still = detect_still_tail(adj, args.stat_th)    # 정지꼬리 제거 후 마지막 이동 프레임
    wrap_idx, wrap_corr, wrap_strong = detect_wrap_closure(c0, args.wrap_th)

    if wrap_strong:
        last_use = min(last_still, wrap_idx - 1)         # 정면 재일치 직전까지만 = [0°,360°)
    else:
        last_use = last_still

    print(f"총 프레임: {n}")
    print(f"[정지꼬리] 인접상관>{args.stat_th}: frame 0..{last_still} 이동 "
          f"(이후 {last_still + 1}..{n - 1} 정지)")
    if wrap_strong:
        print(f"[한바퀴닫힘] ⚠️ frame {wrap_idx} 이 시작(정면)과 재일치 "
              f"(corr={wrap_corr:.4f} ≥ {args.wrap_th}).")
        print(f"            analytic pose는 [0°,360°) 규약 → frame {wrap_idx} 및 이후는 "
              f"'정면 중복'이라 제거(안 그러면 정면 두 겹/ghost).")
    else:
        print(f"[한바퀴닫힘] 강한 정면 재일치 없음 "
              f"(후반부 최대 corr={wrap_corr:.4f} @frame {wrap_idx} < {args.wrap_th}) "
              f"→ 끝이 정면으로 안 돌아옴, 중복 가능성 낮음.")
    print(f"  ✅ 권장 사용 범위: frame 0..{last_use}  ({last_use + 1}장)")
    print(f"[참고] 최저상관(후면≈180°) 위치: frame {int(c0.argmin())}/{n} "
          f"— 후면 자기유사로 부정확할 수 있음.")
    print(f"       ※ 최종 판단은 반드시 몽타주로: 정면→후면→정면 균일 회전 + "
          f"끝 프레임이 첫 프레임(f0)과 겹치지 않는지 확인.")

    out = args.montage or os.path.join(os.path.dirname(args.frames_dir.rstrip('/')), "_contact.png")
    save_montage(args.frames_dir, files, last_use, out,
                 dup_idx=(wrap_idx if wrap_strong else None))
    print(f"확인용 몽타주 저장: {out}")
    if wrap_strong:
        print(f"  → 빨간 'DUP f{wrap_idx}' 칸이 f0(정면)과 같아 보이면 제대로 잡힌 것(그 프레임부터 제거).")


if __name__ == "__main__":
    main()
