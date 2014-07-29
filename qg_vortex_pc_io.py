import sys, slepc4py
slepc4py.init(sys.argv)

from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np
import scipy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt
import scipy.linalg as spalg
from scipy.sparse.linalg import eigs

Print = PETSc.Sys.Print
rank = PETSc.COMM_WORLD.Get_rank()

# This version of the code uses petsc and works in parallel
# Run this using -eps_target 4.87e-06,4.3e-07

def petscKron(A,B):
    dim = A.shape[0]*B.shape[0] # length of resulting matrix
    
    # Used to get indexes where values are non-zero
    Br,Bc = np.nonzero(B)
    Ar,Ac = np.nonzero(A)

    # Need to have values on first axis
    Ar = np.asarray(Ar).ravel(); Ac = np.asarray(Ac).ravel()
    Br = np.asarray(Br).ravel(); Bc = np.asarray(Bc).ravel()

    # Distance between each 'block'
    n = B.shape[1]
    
    # create petsc resulting matrix
    K = PETSc.Mat().createAIJ([dim,dim])
    K.setFromOptions(); K.setUp()
    start,end = K.getOwnershipRange()

    for i in xrange(len(Ar)): # Go through each non-zero value in A
        # br,bc are used to track which 'block' we're in (in result matrix)
        br,bc = n*Ar[i], n*Ac[i]

        for j in xrange(len(Br)): # Go through non-zero values in B
            # kr,kc used to see where to put the number in K (the indexs)
            kr = (Br[j]+br).astype(np.int32)
            kc = (Bc[j]+bc).astype(np.int32)

            if start <= kr < end: # Make sure we're in the correct processor
                K[kr, kc] = A[Ar[i],Ac[i]] * B[Br[j],Bc[j]]

    K.assemble()
    return K

def geometry(Nr,Nz,parms):

    r = np.linspace(-parms.Lr, parms.Lr, Nr+1)
    hr= r[1]-r[0]
    r = r[::-1]
    e = np.ones(Nr)

    Dr = (np.diag(e,-1) - np.diag(e,1))/(2*hr)
    Dr[0,0:2] = [1,-1]/hr
    Dr[Nr,Nr-1:Nr+1] = [1,-1]/hr

    Dr2 = (np.diag(e,-1) - 2*np.diag(np.ones(Nr+1),0) + np.diag(e,1))/hr**2
    Dr2[0,0:3] = [1,-2,1]/hr**2
    Dr2[Nr,Nr-2:Nr+1] = [1,-2,1]/hr**2

    z = np.linspace(-parms.Lz, 0, Nz)
    hz=z[1]-z[0]
    z = z[::-1]
    e = np.ones(Nz-1)

    Dz = (np.diag(e,-1) - np.diag(e,1))/(2*hz)
    Dz[0,0:3] = [-3,4,-1]/(2*hz)
    Dz[Nz-1,Nz-3:Nz] = [1,-4,3]/(2*hz)

    Dz2 = (np.diag(e,-1) - 2*np.diag(np.ones(Nz),0) + np.diag(e,1))/hz**2
    Dz2[0,0:3] = [1,-2,1]/hz**2
    Dz2[Nz-1,Nz-3:Nz] = [1,-2,1]/hz**2

    sp.dia_matrix(Dr); sp.dia_matrix(Dr2)
    sp.dia_matrix(Dz); sp.dia_matrix(Dz2)

    return [Dr,Dr2,r,Dz,Dz2,z]

def build_Lap(r,z,Dr,Dr2,Dz,Dz2,parms):

    Bu = parms.Bu
    N  = len(r)
    N2 = N/2-1
    M  = len(z)

    # Dirichlet BCs:
    D1d = sp.dia_matrix(Dr2[1:N2+1, 1:N2+1])
    D2d = sp.dia_matrix(Dr2[1:N2+1, N-2:N2:-1])
    E1d = sp.dia_matrix(Dr[ 1:N2+1, 1:N2+1])
    E2d = sp.dia_matrix(Dr[ 1:N2+1, N-2:N2:-1])
    
    # Neumann BCs in z
    temp = -np.linalg.inv(np.array([ [Dz[0,0],Dz[0,M-1]],[Dz[M-1,0],Dz[M-1,M-1]] ]))
    BCz1 = np.dot(temp,Dz[ [0,M-1], 1:M-1 ])
    Dzz  = sp.dia_matrix(Dz2[1:M-1,1:M-1] + np.dot(Dz2[1:M-1,[0,M-1]],BCz1))

    # Laplacian in polar coordinates:  d_rr + 1/r*d_r + Bu*d_zz
    # Note that this does not include the azimumthal derivative
    R = sp.dia_matrix(np.diag(1/r[1:N2+1]))
    Lap = petscKron( (D1d+D2d+R*(E1d+E2d)).todense(), np.eye(M-2) ) + Bu*petscKron(np.eye(N2),Dzz.todense())
    
    return Lap

def build_AB(m,r,z,Lap,Pr_bar,Qr_bar,parms):
    N = len(r)
    N2 = N/2-1
    Nz = len(z)
    M2 = Nz/2

    Pr_bar = Pr_bar.ravel(order='F')
    Qr_bar = Qr_bar.ravel(order='F')

    B = PETSc.Mat().createAIJ(Lap.getSize())
    B = Lap - petscKron(np.diag(m**2/(r[1:N2+1]**2)), np.eye(Nz-2))

    Pr_bP = B.getVecLeft()
    sp,ep = Pr_bP.getOwnershipRange()
    Pr_bP[sp:ep] = Pr_bar[sp:ep]

    Qr_bP = B.getVecLeft()
    sq,eq = Qr_bP.getOwnershipRange()
    Qr_bP[sq:eq] = Qr_bar[sq:eq]
    Qr_bP.assemble()

    A = PETSc.Mat().createAIJ(B.getSize())
    A = B.copy()
    
    A.diagonalScale(Pr_bP)
    A.setDiagonal(-Qr_bP,addv=True)

    # # Old code
    # for i in xrange(start,end):
    #     A2[i,:] = Pr_bar[i]*B[i,:]
    #     A2[i,i] = Pr_bar[i]*B[i,i] - Qr_bar[i]

    A.assemble()

    return A, B

class parms(object):
    # Note 'data' file is in this order, and evals is last in the file (12)
    def __init__(self,Lr,Lz,f,g,N,Ro,Lv,Lh,Bu,pr,Nr,Nz):
        self.Lr = Lr # Horizontal scale
        self.Lz = Lz # Vertical scale
        self.f  = f  # Coriols Parameters
        self.g  = g  # Gravity
        self.N  = N  # buoyancy frequency
        self.Ro = Ro
        self.Lv = Lv # normalized depth
        self.Lh = Lh # normalized width
        self.Bu = Bu # Burger number
        self.pr = pr # 'BT' or 'BC'
        self.Nr = Nr
        self.Nz = Nz

def solve_eigensystem(A,B,nEV,cnt,Nz,N2,guess,kt,nconvAr,ncCnt,problem_type=SLEPc.EPS.ProblemType.GNHEP):
    rank = PETSc.COMM_WORLD.getRank()
    size = PETSc.COMM_WORLD.getSize()
    eigVals = open('eigVals','wb')
    eigVecs = open('eigVecs','wb')
    E = SLEPc.EPS(); E.create(comm=SLEPc.COMM_WORLD)

    # for when we want to use the guessVec file as a guess
    if setSpace == False:
        guessVecFile = PETSc.Viewer().createBinary('guessVec.bin','r')
        if guessVecFile != None:
            guessVec = PETSc.Vec().load(guessVecFile)
            E.setInitialSpace(guessVec)

    E.setOperators(A,B)
    E.setDimensions(nEV, PETSc.DECIDE)
    E.setProblemType(problem_type)
    # E.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_IMAGINARY)
    # E.setTarget(guess)
    # E.setTolerances(1e-5, max_it=100)
    
    E.setFromOptions()
    #E.view()

    E.solve()

    vr, wr = A.getVecs()
    vi, wi = A.getVecs()

    nconv = E.getConverged()
    if nconv <= nEV: evals = nconv
    else: evals = nEV
    eigValsnp = np.zeros(nconv,dtype=complex)
    eigVecsnp = np.zeros([nconv,vr.getSize()],dtype=complex)

    #store how many eigenvalues converged
    nconvAr[ncCnt+1] = kt
    nconvAr[ncCnt+2] = nconv
    
    if nconv == 0:
        eigVals.close();eigVecs.close();data.close()
        Print("\nNo eigenvalues have converged\n")
        return

    Print("\nEigenvalues: ")

    for i in range(nconv):
        eigVal = E.getEigenvalue(i)
        omega  = eigVal*kt
        
        eigValsnp[i] = eigVal # store eigenvalue

        E.getEigenvector(i,vr,vi)

        scatter, vrSeq = PETSc.Scatter.toZero(vr)
        im = PETSc.InsertMode.INSERT_VALUES
        sm = PETSc.ScatterMode.FORWARD
        scatter.scatter(vr,vrSeq,im,sm)
        if rank == 0:
            eigVecsnp[i,:] = vrSeq[:] # only for complex petsc

        # when we want to make the guessVec file
        if setSpace == True:
            guessVec = PETSc.Viewer().createBinary('guessVec.bin','w') 
            guessVec(vr)

    # Put eigenvalues and eigenvectors into their files
    if rank == 0:
        eigValsnp.tofile(eigVals)
        eigVecsnp.tofile(eigVecs)

    eigVals.close();eigVecs.close();data.close()

    # Sort and Plot. 
    #You don't have to do this block if you just want to use the read_qg_vortex file
    if rank == 0:
        eigVals = np.fromfile('eigVals',dtype=np.complex128)
        eigVecs = np.fromfile('eigVecs',dtype=np.complex128)
        eigVecs = eigVecs.reshape([nconv,vr.getSize()])

        ind = (-np.imag(eigVals)).argsort()
        eigVecs = eigVecs[ind,:]
        eigVals = kt*eigVals[ind]

        countEigVals = 0
        while np.imag(eigVals[countEigVals]) > 1e-11:
            countEigVals+=1

        if countEigVals == 0:
            Print("No valid eigenvalues have converged")

        for i in range(countEigVals):
            print eigVals[i]/kt
            eigVec1 = eigVecs[i]
            psi = eigVec1.reshape([Nz-2,N2],order='F')
            lvl = np.linspace(psi.real.min(),psi.real.max(),20)
            plt.contourf(r[1:N2+1]/1e3, z[1:Nz-1]/1e3, psi.real, levels=lvl)
            plt.colorbar()
            plt.xlabel('r')
            plt.ylabel('z')
            plt.title(['   m = ', (kt),
            '   gr (direct) = ', omega.imag])
            if i == 0:
                plt.savefig('QG_Vortex.eps', format='eps', dpi=1000)
            #plt.show()

if __name__ == '__main__':
    opts = PETSc.Options()
    nEV = opts.getInt('nev', 5)
    setSpace = opts.getBool('setSpace', True) # true = write eigenvec to binary file
    
    Nr  = opts.getInt('Nr', 41)
    Nz  = opts.getInt('Nz', 20)
    
    ## Specify Physical Parameters
    Lr,Lz,f,g,N,Ro,Lv,Lh,Bu,pr = 6, 6, 8e-5,9.81,np.sqrt(5)*1e-3, 1, 1, 1, 0.1**2,'BC'
    parms = parms(Lr,Lz,f,g,N,Ro,Lv,Lh,Bu,pr,Nr,Nz)
    parmsArr = np.array([Lr,Lz,f,g,N,Ro,Lv,Lh,Bu,Nr,Nz,nEV]) #note pr is not included in the array/file
    
    # Store data in files
    data = open("InputData", "wb")
    nconvData = open("nconvData","wb")

    if rank == 0:
        parmsArr.tofile(data)

    M  = Nz/2
    N2 = (Nr-1)/2

    ktv = 2

    # Define Geometry and Differentiation Matrices
    Dr,Dr2,r,Dz,Dz2,z = geometry(Nr,Nz,parms)

    # Build Laplacian
    Lap = build_Lap(r,z,Dr,Dr2,Dz,Dz2,parms)

    # Build Profile and set up reduced matrix
    rr, zz = np.meshgrid(r,z)
    rin = rr[1:Nz-1,1:N2+1]
    zin = zz[1:Nz-1,1:N2+1]

    # Defining the Basic State
    ## Note: Pr is 1/r*partial_r*Psi
    gaus = Ro*np.exp((-rin**2)/(Lh**2)-((zin+Lz/2)**2)/(Lv**2))
    Psib = -f*Lh**2/4*gaus
    Pr_b =  f/2*gaus
    Q_b  = -f*((rin**2 - Lh**2)/Lh**2+Bu*(Lh/Lv)**2*((zin+Lz/2)**2-0.5*Lv**2)/Lv**2)*gaus
    Qr_b =2*f/Lh**2*((rin**2-2*Lh**2)/Lh**2+Bu*(Lh/Lv)**2*((zin+Lz/2)**2-0.5*Lz**2)/Lz**2)*gaus

    guess = 4.87e-06+4.3e-07*1j
    cnt = 0
    ncCnt = 0
    nconvAr = np.zeros(3) # change this depending on length of for loop * 2 (also storing the kt) + 1 (storing length of eigenvector)
    nconvAr[0] = Lap.getSize()[0]
    for kt in range(2,3): #ktv=1
        # Build A and B for eigen-analysis

        A, B = build_AB(kt,r,z,Lap,Pr_b,Qr_b,parms)

        solve_eigensystem(A,B,nEV,cnt,Nz,N2,guess,kt,nconvAr,ncCnt)

        cnt += 1
        ncCnt += 2

    nconvAr.tofile(nconvData)
    nconvData.close()
