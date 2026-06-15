"""Anisotropic vertical (Z) rescale of a 3DGS ply to correct single-elevation depth-stretch.
Proper: scales positions' z by ZF AND transforms each gaussian covariance Sigma' = M Sigma M^T
(M=diag(1,1,ZF)), re-decomposing into new scale+rotation. SH/opacity/f_dc untouched.
Usage: python zrescale_ply.py IN.ply OUT.ply ZF"""
import sys, numpy as np

IN, OUT, ZF = sys.argv[1], sys.argv[2], float(sys.argv[3])

def read_ply(path):
    f=open(path,'rb'); assert f.readline().strip()==b'ply'; f.readline()
    props=[]; n=0
    while True:
        l=f.readline().strip()
        if l.startswith(b'element vertex'): n=int(l.split()[-1])
        elif l.startswith(b'property'): props.append(l.split()[2].decode())
        elif l==b'end_header': break
    dt=np.dtype([(p,'f4') for p in props])
    d=np.frombuffer(f.read(n*dt.itemsize),dtype=dt,count=n).copy()
    return d, props

d, props = read_ply(IN)
n=len(d)
print("loaded", n, "gaussians, ZF=", ZF)

# quaternion (w,x,y,z) -> R  (batched)
q=np.stack([d['rot_0'],d['rot_1'],d['rot_2'],d['rot_3']],1).astype(np.float64)
q/=np.linalg.norm(q,axis=1,keepdims=True)+1e-12
w,x,y,z=q[:,0],q[:,1],q[:,2],q[:,3]
R=np.empty((n,3,3))
R[:,0,0]=1-2*(y*y+z*z); R[:,0,1]=2*(x*y-w*z); R[:,0,2]=2*(x*z+w*y)
R[:,1,0]=2*(x*y+w*z); R[:,1,1]=1-2*(x*x+z*z); R[:,1,2]=2*(y*z-w*x)
R[:,2,0]=2*(x*z-w*y); R[:,2,1]=2*(y*z+w*x); R[:,2,2]=1-2*(x*x+y*y)

s=np.exp(np.stack([d['scale_0'],d['scale_1'],d['scale_2']],1).astype(np.float64))
# Sigma = R diag(s^2) R^T
Sig = R @ (s[:,:,None]**2 * np.transpose(R,(0,2,1)))
M=np.diag([1.0,1.0,ZF])
Sig = M[None] @ Sig @ M[None]
Sig = 0.5*(Sig+np.transpose(Sig,(0,2,1)))  # symmetrize
lam, V = np.linalg.eigh(Sig)              # ascending eigenvalues, V orthonormal
lam=np.clip(lam,1e-12,None)
s_new=np.sqrt(lam)                         # (n,3)
R_new=V
# ensure proper rotation (det=+1)
det=np.linalg.det(R_new)
R_new[det<0,:,0]*=-1
# R -> quaternion (w,x,y,z)
tr=R_new[:,0,0]+R_new[:,1,1]+R_new[:,2,2]
qw=np.empty(n); qx=np.empty(n); qy=np.empty(n); qz=np.empty(n)
t0=tr>0
S=np.sqrt(np.clip(tr[t0],0,None)+1.0)*2
qw[t0]=0.25*S; qx[t0]=(R_new[t0,2,1]-R_new[t0,1,2])/S; qy[t0]=(R_new[t0,0,2]-R_new[t0,2,0])/S; qz[t0]=(R_new[t0,1,0]-R_new[t0,0,1])/S
# fallback for tr<=0: per-element largest diagonal
idx=np.where(~t0)[0]
for i in idx:
    m=R_new[i]
    if m[0,0]>m[1,1] and m[0,0]>m[2,2]:
        S=np.sqrt(1+m[0,0]-m[1,1]-m[2,2])*2; qw[i]=(m[2,1]-m[1,2])/S; qx[i]=0.25*S; qy[i]=(m[0,1]+m[1,0])/S; qz[i]=(m[0,2]+m[2,0])/S
    elif m[1,1]>m[2,2]:
        S=np.sqrt(1+m[1,1]-m[0,0]-m[2,2])*2; qw[i]=(m[0,2]-m[2,0])/S; qx[i]=(m[0,1]+m[1,0])/S; qy[i]=0.25*S; qz[i]=(m[1,2]+m[2,1])/S
    else:
        S=np.sqrt(1+m[2,2]-m[0,0]-m[1,1])*2; qw[i]=(m[1,0]-m[0,1])/S; qx[i]=(m[0,2]+m[2,0])/S; qy[i]=(m[1,2]+m[2,1])/S; qz[i]=0.25*S

# write back
d['z']=(d['z'].astype(np.float64)*ZF).astype(np.float32)
d['scale_0']=np.log(s_new[:,0]).astype(np.float32)
d['scale_1']=np.log(s_new[:,1]).astype(np.float32)
d['scale_2']=np.log(s_new[:,2]).astype(np.float32)
qn=np.stack([qw,qx,qy,qz],1); qn/=np.linalg.norm(qn,axis=1,keepdims=True)+1e-12
d['rot_0']=qn[:,0].astype(np.float32); d['rot_1']=qn[:,1].astype(np.float32)
d['rot_2']=qn[:,2].astype(np.float32); d['rot_3']=qn[:,3].astype(np.float32)

with open(OUT,'wb') as f:
    f.write(b'ply\nformat binary_little_endian 1.0\n')
    f.write(('element vertex %d\n'%n).encode())
    for p in props: f.write(('property float %s\n'%p).encode())
    f.write(b'end_header\n')
    f.write(d.tobytes())
print("wrote", OUT)
# report new extent
op=1/(1+np.exp(-d['opacity'].astype(np.float64))); vis=op>0.5
pv=np.stack([d['x'],d['y'],d['z']],1)[vis]
lo,hi=np.percentile(pv,1,0),np.percentile(pv,99,0); e=hi-lo
print("new extent x,y,z:",np.round(e,3).tolist(),"aspect z/median(xy):",round(e[2]/np.median(e[:2]),2))
