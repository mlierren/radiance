"""Render a turntable orbit of a standard 3DGS ply using the r3dg CUDA rasterizer.
Usage: python render_turntable.py IN.ply OUT_DIR N_FRAMES ELEV_DEG [CANVAS]
Outputs PNG frames f_%04d.png (black bg) to OUT_DIR."""
import sys, os, math
import numpy as np, torch
from types import SimpleNamespace
from PIL import Image
REPO='/workspace/radiance/third_party/Relightable3DGaussian'
sys.path.insert(0, REPO)
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from gaussian_renderer.render import render_view

PLY=sys.argv[1]; OUT=sys.argv[2]
N=int(sys.argv[3]) if len(sys.argv)>3 else 300
ELEV=float(sys.argv[4]) if len(sys.argv)>4 else 0.0
S=int(sys.argv[5]) if len(sys.argv)>5 else 800
FOV=40.0
os.makedirs(OUT, exist_ok=True)

g=GaussianModel(3, render_type='render')
g.load_ply(PLY)
g.active_sh_degree=g.max_sh_degree

xyz=g.get_xyz.detach().cpu().numpy()
op=g.get_opacity.detach().cpu().numpy().reshape(-1)
vis=op>0.5; pv=xyz[vis] if vis.sum()>1000 else xyz
lo,hi=np.percentile(pv,1,0),np.percentile(pv,99,0)
center=(lo+hi)/2; ext=hi-lo
half=float(max(ext))/2*1.25
radius=half/math.tan(math.radians(FOV)/2)
print('center',np.round(center,3),'ext',np.round(ext,3),'radius',round(radius,3))

fovx=fovy=math.radians(FOV)
fx=S/(2*math.tan(fovx/2)); fy=S/(2*math.tan(fovy/2)); cx=cy=S/2
pipe=SimpleNamespace(debug=False, compute_cov3D_python=False, compute_SHs_python=False)
bg=torch.tensor([0,0,0],dtype=torch.float32,device='cuda')

def look_at_c2w(eye,target,up=(0,0,1)):
    eye=np.array(eye,float);target=np.array(target,float);up=np.array(up,float)
    f=target-eye;f/=np.linalg.norm(f);r=np.cross(f,up);r/=np.linalg.norm(r);u=np.cross(r,f)
    c2w=np.eye(4);c2w[:3,0],c2w[:3,1],c2w[:3,2],c2w[:3,3]=r,u,-f,eye;return c2w

el=math.radians(ELEV)
for i in range(N):
    th=2*math.pi*i/N
    eye=center+np.array([radius*math.cos(th)*math.cos(el),radius*math.sin(th)*math.cos(el),radius*math.sin(el)])
    c2w=look_at_c2w(eye,center); c2w[:3,1:3]*=-1
    w2c=np.linalg.inv(c2w); R=np.transpose(w2c[:3,:3]); T=w2c[:3,3]
    cam=Camera(0,R,T,fovx,fovy,fx,fy,cx,cy,None,f'v{i}',i,width=S,height=S)
    with torch.no_grad():
        img=render_view(cam,g,pipe,bg,1.0,None)['render']
    arr=(img.clamp(0,1).permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8)
    Image.fromarray(arr).save(os.path.join(OUT,'f_%04d.png'%i))
print('rendered',N,'frames ->',OUT)
