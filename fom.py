"""
written by Shady, https://github.com/Shady-Ahmed
edited by Mehrdad, https://github.com/zomorodiyan
"""
#%% Import libraries, functions, inputs
import numpy as np
from tools import jacobian, laplacian, initial, RK3, tbc, \
     poisson_fst, BoussRHS, velocity, export_data, export_data_test
from inputs import lx, ly, nx, ny, Re, Ri, Pr, dt, nt, ns, freq

#%% grid
dx = lx/nx; dy = ly/ny
x = np.linspace(0.0,lx,nx+1); y = np.linspace(0.0,ly,ny+1)
X, Y = np.meshgrid(x, y, indexing='ij')

# initialize
n=0; time=0
w,s,t = initial(nx,ny)
export_data(nx,ny,n,w,s,t)

#%% time integration
for n in range(1,nt+1):
    time = time+dt
    w,s,t = RK3(BoussRHS,nx,ny,dx,dy,Re,Pr,Ri,w,s,t,dt)
    u,v = velocity(nx,ny,dx,dy,s)
    umax = np.max(np.abs(u)); vmax = np.max(np.abs(v))
    cfl = np.max([umax*dt/dx, vmax*dt/dy])
    if cfl >= 0.8:
        print('CFL exceeds maximum value')
        break

    if n%freq==0:
        export_data(nx,ny,n,w,s,t)
        print('FOM ',"{:.0f}".format((n)/(nt)*100), '%   ', end='\r')
print('')
