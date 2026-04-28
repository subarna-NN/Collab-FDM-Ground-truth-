# ================================================================
#  FDM Solution: 2D Advection-Diffusion Transport Equation
#  x_vec=(x,z) in R x (0,1),  t in (0,T)
#
#  PDE:  dC/dt + v.grad(C) = div(D.grad(C))
#  v = (Pe*u(z), -omega),  u(z)=(1/alpha^2)*[1-cosh(alpha*(z-1))/cosh(alpha)]
#  D = diag(R, 1)   [R=1 => identity tensor D=I]
#
#  BCs:  C=0 at x=+-Lx  (Dirichlet)
#        dC/dz + (omega-beta)*C = 0  at z=0   (Robin)
#        dC/dz + omega*C = 0          at z=1   (Robin)
#  IC:   C(x,z,0) = delta(x)*delta(z-z0)  [Gaussian approx]
#
#  Scheme: Forward Euler, upwind advection, central diffusion
#          dt auto-chosen from CFL+diffusion stability (safety 0.4)
# ================================================================

import numpy as np
import matplotlib.pyplot as plt

# ---- Parameters ------------------------------------------------
Pe    = 10.0    # Peclet number              [1, 100]
omega = 1.0     # vertical advection / BC    [0, 10]
beta  = 5.0     # desorption coefficient     [0, 100]
alpha = 2.0     # flow-profile shape param   [0.5, 5]
R     = 1.0     # diffusivity scaling (Dxx = R)
z0    = 0.5     # source depth in (0,1)
Lx    = 5.0     # half-domain in x
T     = 2.0     # final time
Nx    = 200     # grid points in x
Nz    = 80      # grid points in z

# ---- Spatial grid ----------------------------------------------
x  = np.linspace(-Lx, Lx, Nx)
z  = np.linspace( 0.0, 1.0, Nz)
dx = x[1] - x[0]
dz = z[1] - z[0]
X, Z = np.meshgrid(x, z, indexing="ij")   # shape (Nx, Nz)

# ---- Velocity and diffusion -----------------------------------
def u_profile(z_arr, a):
    return (1.0/a**2)*(1.0 - np.cosh(a*(z_arr-1.0))/np.cosh(a))

Vx_1d = Pe * u_profile(z, alpha)          # (Nz,)  z-dependent only
Vz    = -omega                             # scalar

Vx2D  = np.broadcast_to(Vx_1d[np.newaxis,:], (Nx,Nz)).copy()
Vz2D  = np.full((Nx,Nz), Vz)

Dxx = R        # D = diag(R, 1); R=1 => D = identity
Dzz = 1.0

# ---- Auto-stable dt  (safety factor 0.4) ----------------------
mvx = float(np.max(np.abs(Vx2D)));  mvz = abs(Vz)
dt_d = 0.5/(Dxx/dx**2 + Dzz/dz**2)
dt_a = min(dx/mvx if mvx>1e-14 else np.inf,
           dz/mvz if mvz>1e-14 else np.inf)
dt_lim = min(dt_d, dt_a)
dt = 0.4*dt_lim
Nt = int(np.ceil(T/dt)) + 1
dt = T/(Nt-1)           # exact
t  = np.linspace(0.0, T, Nt)

print(f"Grid : Nx={Nx}, Nz={Nz}, Nt={Nt}")
print(f"Step : dx={dx:.4f}, dz={dz:.4f}, dt={dt:.2e}  [limit={dt_lim:.2e}] OK")

# ---- Initial condition (Gaussian ~ delta*delta) ----------------
eps = 3.0*max(dx, dz)
C   = np.exp(-(X**2 + (Z-z0)**2)/(2*eps**2)) / (2*np.pi*eps**2)

# ---- Boundary conditions ---------------------------------------
denom_z0 = 3.0 - 2.0*(omega-beta)*dz
denom_z1 = 3.0 + 2.0*omega*dz

def apply_bcs(C):
    C[0,:]  = 0.0;  C[-1,:] = 0.0
    if abs(denom_z0) > 1e-14:
        C[:,0] = (4.0*C[:,1] - C[:,2]) / denom_z0
    else:
        C[:,0] = 0.0
    if abs(denom_z1) > 1e-14:
        C[:,-1] = (4.0*C[:,-2] - C[:,-3]) / denom_z1
    else:
        C[:,-1] = 0.0
    return C

C = apply_bcs(C)

# ---- FDM time loop --------------------------------------------
save_every = max(1,(Nt-1)//50)
snap_C = [C.copy()];  snap_t = [0.0]

print(f"Running {Nt} steps ...")

for n in range(Nt-1):
    # central diffusion
    d2x = np.zeros_like(C);  d2z = np.zeros_like(C)
    d2x[1:-1,:] = (C[2:,:] - 2*C[1:-1,:] + C[:-2,:]) / dx**2
    d2z[:,1:-1] = (C[:,2:] - 2*C[:,1:-1] + C[:,:-2]) / dz**2

    # upwind advection x
    adv_x = np.zeros_like(C)
    pos = Vx2D[1:-1,:] >= 0
    adv_x[1:-1,:] = np.where(pos,
        (C[1:-1,:]-C[:-2,:]) /dx,
        (C[2:,  :]-C[1:-1,:])/dx)

    # upwind advection z (Vz constant)
    adv_z = np.zeros_like(C)
    if Vz >= 0:
        adv_z[:,1:-1] = (C[:,1:-1]-C[:,:-2])/dz
    else:
        adv_z[:,1:-1] = (C[:,2:]-C[:,1:-1])/dz

    C = C + dt*(-Vx2D*adv_x - Vz2D*adv_z + Dxx*d2x + Dzz*d2z)
    C = apply_bcs(C)

    if (n+1) % save_every == 0:
        snap_C.append(C.copy());  snap_t.append(t[n+1])

print("FDM done.")
snap_C = np.array(snap_C);  snap_t = np.array(snap_t)

# ---- Plots  ---------------------------------------------------
import os
OUT = "/content/fdm_outputss/"
os.makedirs(OUT, exist_ok=True)   # auto-create on Colab (or any system)

# 1. Velocity profile
fig,ax = plt.subplots(1,2,figsize=(9,4))
ax[0].plot(u_profile(z,alpha),z,"b-",lw=2)
ax[0].set(xlabel="u(z)",ylabel="z",title=f"u(z), alpha={alpha}")
ax[0].axvline(0,c="k",lw=0.5);ax[0].grid(alpha=0.3)
ax[1].plot(Pe*u_profile(z,alpha),z,"r-",lw=2)
ax[1].set(xlabel="Pe*u(z)",ylabel="z",title=f"Horiz. velocity, Pe={Pe}")
ax[1].axvline(0,c="k",lw=0.5);ax[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT+"velocity_profile.png",dpi=150,bbox_inches="tight");plt.close()

# 2. Concentration snapshots
nsn = min(6,len(snap_C))
idx = np.linspace(0,len(snap_C)-1,nsn,dtype=int)
vmax= float(snap_C.max())
fig,axs = plt.subplots(2,3,figsize=(14,8))
for k,i in enumerate(idx):
    a = axs.ravel()[k]
    im= a.pcolormesh(x,z,snap_C[i].T,cmap="hot_r",vmin=0,vmax=vmax,shading="auto")
    plt.colorbar(im,ax=a,shrink=0.85)
    a.set(title=f"t={snap_t[i]:.3f}",xlabel="x",ylabel="z")
    a.set_xlim([x[0],x[-1]]);a.set_ylim([0,1])
plt.suptitle("C(x,z,t) - 2D Advection-Diffusion FDM",fontsize=13,y=1.01)
plt.tight_layout()
plt.savefig(OUT+"concentration_snapshots.png",dpi=150,bbox_inches="tight");plt.close()

# 3. Mass conservation
mass = [snap_C[i].sum()*dx*dz for i in range(len(snap_C))]
fig,ax = plt.subplots(figsize=(7,3.5))
ax.plot(snap_t,mass,"b-",lw=1.5,label="Total mass")
ax.axhline(mass[0],c="r",ls="--",lw=1,label="Initial mass")
ax.set(xlabel="t",ylabel="integral C dxdz",title="Mass conservation")
ax.legend();ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT+"mass_conservation.png",dpi=150,bbox_inches="tight");plt.close()

# 4. Centreline
iz0 = int(np.argmin(np.abs(z-z0)))
cml = plt.cm.plasma
fig,ax = plt.subplots(figsize=(9,4))
for k,i in enumerate(idx):
    cc = cml(k/max(1,nsn-1))
    ax.plot(x,snap_C[i,:,iz0],color=cc,lw=1.5,label=f"t={snap_t[i]:.2f}")
ax.set(xlabel="x",ylabel=f"C(x,z~{z[iz0]:.2f},t)",title=f"Centreline at z~z0={z0}")
ax.legend(fontsize=8);ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT+"centerline_concentration.png",dpi=150,bbox_inches="tight");plt.close()

# ---- Summary --------------------------------------------------
print()
print("="*60)
print("  FDM Summary")
print("="*60)
print(f"  Domain  : x in [{-Lx},{Lx}], z in [0,1], t in [0,{T}]")
print(f"  Grid    : Nx={Nx}, Nz={Nz}, Nt={Nt}")
print(f"  Spacing : dx={dx:.4f}, dz={dz:.4f}, dt={dt:.2e}")
print(f"  Params  : Pe={Pe}, omega={omega}, beta={beta}, alpha={alpha}, R={R}, z0={z0}")
print(f"  D       : Dxx=R={Dxx:.3f}, Dzz=1.0  => D=diag({Dxx:.3f},1.0)")
print(f"  max|Vx| : {mvx:.4f},  Vz={Vz:.4f}")
print(f"  Mass(0) : {mass[0]:.6f}")
print(f"  Mass(T) : {mass[-1]:.6f}")
print(f"  maxC(0) : {snap_C[0].max():.4f}")
print(f"  maxC(T) : {snap_C[-1].max():.6f}")
print("="*60)
print("Plots saved.")
