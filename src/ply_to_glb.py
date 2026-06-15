"""3DGS .ply -> .glb 변환 (point cloud / mesh).

3DGS(가우시안 splat)는 glTF에 네이티브 개념이 없어 둘 중 하나로 *손실 변환*한다:
  - points: 가우시안 중심점 + SH-DC 색 → GLB POINTS 프리미티브(데이터에 충실, 점으로 보임)
  - mesh  : 밀도 그리드 + marching cubes 표면 재구성 + 최근접 색 → GLB 메시(솔리드, 더 뭉툭/lossy)
좌표: 3DGS는 Z-up, glTF는 Y-up → (x,y,z)→(x,z,-y) 회전 적용. 중심을 원점으로 이동.

사용법:
  python src/ply_to_glb.py IN.ply OUT_PREFIX [--mode both|points|mesh] [--opacity 0.3] [--grid 192]
출력: OUT_PREFIX_points.glb / OUT_PREFIX_mesh.glb
"""
import argparse, numpy as np


def read_3dgs_ply(path):
    f = open(path, "rb"); assert f.readline().strip() == b"ply"; f.readline()
    props = []; n = 0
    while True:
        l = f.readline().strip()
        if l.startswith(b"element vertex"): n = int(l.split()[-1])
        elif l.startswith(b"property"): props.append(l.split()[2].decode())
        elif l == b"end_header": break
    dt = np.dtype([(p, "f4") for p in props])
    d = np.frombuffer(f.read(n * dt.itemsize), dtype=dt, count=n)
    xyz = np.stack([d["x"], d["y"], d["z"]], 1).astype(np.float64)
    op = 1.0 / (1.0 + np.exp(-d["opacity"].astype(np.float64)))
    C0 = 0.28209479177387814
    rgb = np.clip(0.5 + C0 * np.stack([d["f_dc_0"], d["f_dc_1"], d["f_dc_2"]], 1).astype(np.float64), 0, 1)
    return xyz, op, rgb


def zup_to_yup(v):
    return np.stack([v[:, 0], v[:, 2], -v[:, 1]], 1)


def to_points_glb(xyz, rgb, out):
    import trimesh
    pc = trimesh.PointCloud(vertices=xyz, colors=(np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    pc.export(out)
    print(f"[points] {len(xyz):,} pts -> {out}")


def to_mesh_glb(xyz, rgb, out, grid=192, level=0.5):
    from scipy import ndimage as ndi
    from skimage import measure
    from scipy.spatial import cKDTree
    import trimesh
    lo, hi = xyz.min(0), xyz.max(0)
    ext = hi - lo
    res = ext.max() / grid
    dims = np.maximum((ext / res).astype(int) + 4, 8)
    vol = np.zeros(dims, np.float32)
    idx = ((xyz - lo) / res + 2).astype(int)
    idx = np.clip(idx, 0, np.array(dims) - 1)
    np.add.at(vol, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)   # 점 밀도 누적
    vol = ndi.gaussian_filter(vol, sigma=1.2)
    vol /= vol.max() + 1e-9
    verts, faces, _, _ = measure.marching_cubes(vol, level=level * vol[vol > 0].mean())
    verts_world = verts * res + lo - 2 * res                  # 그리드→월드 좌표
    # 최근접 가우시안 색 이식
    tree = cKDTree(xyz)
    _, nn = tree.query(verts_world, k=1)
    vcol = (np.clip(rgb[nn], 0, 1) * 255).astype(np.uint8)
    vcol = np.concatenate([vcol, np.full((len(vcol), 1), 255, np.uint8)], 1)
    m = trimesh.Trimesh(vertices=verts_world, faces=faces, vertex_colors=vcol, process=False)
    m.export(out)
    print(f"[mesh] {len(verts):,} verts / {len(faces):,} faces -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_ply"); ap.add_argument("out_prefix")
    ap.add_argument("--mode", choices=["both", "points", "mesh"], default="both")
    ap.add_argument("--opacity", type=float, default=0.3, help="이 불투명도 이상만 사용")
    ap.add_argument("--grid", type=int, default=192, help="mesh 밀도 그리드 해상도")
    a = ap.parse_args()

    xyz, op, rgb = read_3dgs_ply(a.in_ply)
    keep = op > a.opacity
    xyz, rgb = xyz[keep], rgb[keep]
    print(f"loaded {keep.sum():,} / {len(keep):,} gaussians (opacity>{a.opacity})")
    xyz = zup_to_yup(xyz)
    xyz = xyz - xyz.mean(0)                                   # 원점으로 이동

    if a.mode in ("both", "points"):
        to_points_glb(xyz, rgb, a.out_prefix + "_points.glb")
    if a.mode in ("both", "mesh"):
        to_mesh_glb(xyz, rgb, a.out_prefix + "_mesh.glb", grid=a.grid)


if __name__ == "__main__":
    main()
