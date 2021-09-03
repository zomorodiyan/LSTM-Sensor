import numpy as np
import matplotlib.pyplot as plt
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
from tensorflow.keras.models import load_model
from tools import initial, podproj_svd, RK3, BoussRHS, podrec_svd, RK3t,\
        BoussRHS_t, export_data_test, export_data2, export_data3
from sklearn.preprocessing import MinMaxScaler

# run a coarse fom, get psi omega theta for each step and each mesh
# (run fom.py get fom_nx'x'ny)

# run svd on psi phi theta, get phi_psi phi_omega phi_theta for each mesh
#   and alpha beta for number of modes and each time step
# (run pod.py get pod_nx'x'ny'.npz')

# run lstm on alpha beta, get model
# (run lstm.py get lstm_nx'x'ny'.h5')

# then run this

# Load inputs
from inputs import lx, ly, nx, ny, Re, Ri, Pr, dt, nt, ns, freq, nr, n_each, ws

#%% grid
dx = lx/nx
dy = ly/ny
x = np.linspace(0.0,lx,nx+1)
y = np.linspace(0.0,ly,ny+1)
X, Y = np.meshgrid(x, y, indexing='ij')

a_window = np.empty([ws*n_each,nr])
b_window = np.empty([ws*n_each,nr])

#%% the FOM, and ROM, part of the diagrams
# Load Data
filename = './results/pod_'+ str(nx) + 'x' + str(ny) + '.npz'
data = np.load(filename)
aTrue = data['aTrue']; bTrue = data['bTrue']

# Scale Data
data = np.concatenate((aTrue, bTrue), axis=1) # axes 0:snapshots 1:states
scaler2 = MinMaxScaler(feature_range=(-1,1))
scaled = scaler2.fit_transform(data)
ablstm = np.empty([ns+1,scaled.shape[1]])
xtest = np.empty([1,ws,scaled.shape[1]])
ablstm[0:ws*n_each,:] = scaled[0:ws*n_each,:]

# Recreate the lstm model, including its weights and the optimizer
model = load_model('./results/lstm_'+str(nx)+'x'+str(ny)+'.h5')
xtest = np.copy(np.expand_dims(ablstm[0:ws*n_each:n_each,:], axis=0))
for i in range(ws*n_each, ns+1):
    print('ROM ',"{:.0f}".format((i-ws*n_each)/(ns+1-ws*n_each)*100), '%   ', end='\r')
    ablstm[i,:] = model.predict(xtest)
    xtest = np.copy(np.expand_dims(ablstm[i-ws*n_each+1:i-n_each+2:n_each,:], axis=0))

#%% export alpha and beta for rom, fom, romfom
export_data3(nx,ny,ablstm,'rom')
export_data3(nx,ny,scaled,'fom')
