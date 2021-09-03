from tools import import_data,import_data2,import_data3
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

# Load inputs
from inputs import lx, ly, nx, ny, nt, ns, nr, Re
x = np.linspace(0.0,lx,nx+1)
y = np.linspace(0.0,ly,ny+1)
X, Y = np.meshgrid(x, y, indexing='ij')

#%% Load FOM results for t=0,2,4,8
n=0 #t=0
w0,s0,t0 = import_data(nx,ny,n)
n=int(2*nt/8) #t=2
w2,s2,t2 = import_data(nx,ny,n)
n=int(4*nt/8) #t=4
w4,s4,t4 = import_data(nx,ny,n)
n=int(8*nt/8) #t=8
w8,s8,t8 = import_data(nx,ny,n)


if not os.path.exists('./results/nx_'+str(nx)+'_Re_'+str(int(Re))):
    os.makedirs('./results/nx_'+str(nx)+'_Re_'+str(int(Re)))

s1 = import_data3(nx,ny,'rom')
s2 = import_data3(nx,ny,'fom')
t = np.arange(0, ns+1, 1)
n=0
name = 'alpha'
for j in range(2):
    fig, axs = plt.subplots(5,2,figsize=(20,10))
    for ax in axs.flat:
        ax.plot(t, s1[:ns+1,n])
        ax.plot(t, s2[:ns+1,n])
        #ax.plot(t, s3[:ns+1,n])
        n=n+1
        if(n==1 or n==nr+1):
            ax.legend(["fom", "rom"], loc='upper left')
        ax.grid()
        #plt.ylim(-1,1)
        if(n==nr-1 or n==nr):
            ax.set(xlabel='timeStep')
        if(n==2*nr-1 or n==2*nr):
            ax.set(xlabel='timeStep')
    fig.savefig('./results/nx_'+str(nx)+'_Re_'+str(int(Re))+'/'+name+'.png')
    name = 'beta'
    fig.clear(True)
'''
if(n==0):
    plt.show()
'''
#plt.clf()
