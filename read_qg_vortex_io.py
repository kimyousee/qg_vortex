import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=3)

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
    while (countEigVals < eigVals.shape[0] and np.imag(eigVals[countEigVals]) > 1e-11):
        countEigVals+=1
        
    if countEigVals == 0:
        print("No valid eigenvalues have converged")
    print("\nEigenvalues: \n")
    # Print eigenvalues and plot eigenvector

    for i in range(countEigVals):
        print eigVals[i]
        omega  = eigVals[i]*kt
        eigVec1 = eigVecs[i]
        psi = eigVec1.reshape([Nz-2,N2],order='F')
        lvlr = np.linspace(psi.real.min(),psi.real.max(),20)
        lvli = np.linspace(psi.imag.min(),psi.imag.max(),20)
        fig = plt.figure()

        plt.rcParams["axes.titlesize"] = 8
        rmode = fig.add_subplot(1,2,1)
        rmode.tick_params(axis='both', labelsize=8)
        plt.contourf(r[1:N2+1]/1e3, z[1:Nz-1]/1e3, psi.real, levels=lvlr)
        rcbar = plt.colorbar()
        cl = plt.getp(rcbar.ax,'ymajorticklabels')
        plt.setp(cl,fontsize=8)

        plt.xlabel('r')
        plt.ylabel('z')
        gr = np.array([omega.imag])
        plt.title(['   psi.real',
                   '   m = ', kt,
                   '   gr = ', gr ])

        imode = fig.add_subplot(1,2,2)
        imode.tick_params(axis='both', labelsize=8)
        plt.contourf(r[1:N2+1]/1e3, z[1:Nz-1]/1e3, psi.imag, levels=lvli)
        icbar = plt.colorbar()
        icl = plt.getp(icbar.ax,'ymajorticklabels')
        plt.setp(icl,fontsize=8)

        plt.xlabel('r')
        plt.ylabel('z')
        plt.title(['   psi.imag',
                   '   m = ', kt,
                   '   gr = ', gr ])

        fig = "QG_Vortex_m%d" % i
        plt.savefig(fig, format='eps', dpi=1000)
        plt.show()

print ""
