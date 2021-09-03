import numpy as np
#----------FOM--------------
ns = 800 #number of snapshots
nx = 512 #number of meshes in x direction
ny = int(nx/8) #number of meshes in y direction
lx = 8 #length in x direction
ly = 1 #length in y direction
Re = 1e3 #Reynolds Number: inertial/viscous
Ri = 4 #Richardson Number: Buoyancy/flow_shear
Pr = 1 #Prandtl Number: momentum_diffusivity/thermal_diffusivity
Tm = 8 #maximum time
dt = 1e-3 #time_step_size
nt = np.int(np.round(Tm/dt)) #number of time_steps
freq = np.int(nt/ns) #every freq time_step we export data
#----------POD--------------
nr = 10 #number of pod modes
#----------LSTM-------------
ws = 5 #window size
n_each = 1 #one every n_each time snapshots is used for lstm's data-windows
