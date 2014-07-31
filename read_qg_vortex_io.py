import numpy as np
import matplotlib.pyplot as plt

# Script to read in data that was made already with qg_vortex_pc_timing.py

# Read from necessary files

## Contains: Lr,Lz,f,g,N,Ro,Lv,Lh,Bu,Nr,Nz,nEV
parms = np.fromfile('InputData') 
## Contains: [<size of a vector>, kt0, nconv0, kt1,nconv1, ...] 
nconvData = np.fromfile('nconvData') 
## Eigenvalues and eigenvectors
eigVals = np.fromfile('eigVals',dtype=np.complex128)
eigVecs = np.fromfile('eigVecs',dtype=np.complex128)

size = nconvData[0]
Nz = parms[10]
Nr = parms[9]
N2 = (Nr-1)/2
Lr = parms[0]
Lz = parms[1]
r = np.linspace(-Lr, Lr, Nr+1)
z = np.linspace(-Lz, 0, Nz)
r = r[::-1]
z = z[::-1]

# Just 1 iteration unless for loop is changed in main code
for eSolve in range((len(nconvData)-1)/2):

    kt = nconvData[eSolve*2+1]
    nconv = nconvData[(eSolve+1)*2]

    eigVecs = eigVecs.reshape([nconv,size])

    # Sort eigenvalues and eigenvectors
    ind = (-np.imag(eigVals)).argsort()
    eigVecs = eigVecs[ind,:]
    eigVals = eigVals[ind]

    # Count how many eigenvalues are valid
    countEigVals = 0
    while np.imag(eigVals[countEigVals]) > 1e-11:
        countEigVals+=1
        
    if countEigVals == 0:
        Print("No valid eigenvalues have converged")
    
    # Print eigenvalues and plot eigenvector
    for i in range(countEigVals):
        print eigVals[i]
        omega  = eigVals[i]*kt
        eigVec1 = eigVecs[i]
        psi = eigVec1.reshape([Nz-2,N2],order='F')
        lvl = np.linspace(psi.real.min(),psi.real.max(),20)
        plt.contourf(r[1:N2+1]/1e3, z[1:Nz-1]/1e3, psi.real, levels=lvl)
        plt.colorbar()
        plt.xlabel('r')
        plt.ylabel('z')
        plt.title(['   m = ', (kt),
                   '   gr (direct) = ', omega.imag])
        #plt.savefig('QG_Vortex.eps', format='eps', dpi=1000)
        plt.show()

