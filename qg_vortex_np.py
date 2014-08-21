
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
#np.set_printoptions(precision=3)

# This version of the code uses numpy/scipy (eig/eigs)

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

    sp.csr_matrix(Dr); sp.csr_matrix(Dr2)
    sp.csr_matrix(Dz); sp.csr_matrix(Dz2)

    return [Dr,Dr2,r,Dz,Dz2,z]

def build_Lap(r,z,Dr,Dr2,Dz,Dz2,parms):
    Bu = parms.Bu
    N  = len(r)
    N2 = N/2-1
    M  = len(z)

    # Dirichlet BCs:
    D1d = sp.csr_matrix(Dr2[1:N2+1, 1:N2+1])
    D2d = sp.csr_matrix(Dr2[1:N2+1, N-2:N2:-1])
    E1d = sp.csr_matrix(Dr[ 1:N2+1, 1:N2+1])
    E2d = sp.csr_matrix(Dr[ 1:N2+1, N-2:N2:-1])
    #print D1d.todense()

    # Neumann BCs in z
    BCz1 = -np.linalg.solve(
        np.array([ [Dz[0,0],Dz[0,M-1]],[Dz[M-1,0],Dz[M-1,M-1]] ]),
                    Dz[ [0,M-1], 1:M-1 ])
    Dzz  = sp.csr_matrix(Dz2[1:M-1,1:M-1] + np.dot(Dz2[1:M-1,[0,M-1]],BCz1))
    
    # Laplacian in polar coordinates:  d_rr + 1/r*d_r + Bu*d_zz
    # Note that this does not include the azimumthal derivative
    R = sp.csr_matrix(np.diag(1/r[1:N2+1]))
    Lap = sp.kron( (D1d+D2d+R*(E1d+E2d)), np.eye(M-2) )+Bu*sp.kron(np.eye(N2),Dzz)
    return Lap

def build_AB(m,r,z,Lap,Pr_bar,Qr_bar,parms):
    N = len(r)
    N2 = N/2-1
    Nz = len(z)
    M2 = Nz/2

    Pr_bar = Pr_bar.ravel(order='F')
    Qr_bar = Qr_bar.ravel(order='F')

    B = Lap - sp.kron(np.diag(m**2/(r[1:N2+1]**2)), np.eye(Nz-2))
    A = np.diag(Pr_bar)*B - np.diag(Qr_bar)*np.eye((Nz-2)*N2)
    B = sp.csr_matrix(B)
    A = sp.csr_matrix(A)
    return A, B

class parms(object):
    def __init__(self,Lr,Lz,f,g,N,Ro,Lv,Lh,Bu,pr):
        self.Lr =  Lr # Horizontal scale
        self.Lz =  Lz # Vertical scale
        self.f  =  f  # Coriols Parameters
        self.g  =  g  # Gravity
        self.N  =  N  # buoyancy frequency
        self.Ro =  Ro
        self.Lv =  Lv # normalized depth
        self.Lh =  Lh # normalized width
        self.Bu =  Bu # Burger number
        self.pr =  pr # 'BT' or 'BC'
"""
def solve_eigensystem(A,nEV,cnt,problem_type=SLEPc.EPS.ProblemType.NHEP):
    E = SLEPc.EPS(); E.create(comm=SLEPc.COMM_WORLD)

    E.setOperators(A)
    E.setDimensions(nEV, PETSc.DECIDE)
    E.setProblemType(problem_type)
    #E.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_IMAGINARY)
    #E.setTolerances(1e-5, max_it=100)
    
    E.setFromOptions()
    E.view()
    
    E.solve()
"""
if __name__ == '__main__':
    opts = PETSc.Options()
    nEV = opts.getInt('nev', 5)

    ## Specify Physical Parameters
                 #Lr,Lz,f,   g,   N,              Ro,Lv,Lh, Bu,    pr
    parms = parms(6, 6, 8e-5,9.81,np.sqrt(5)*1e-3, 1, 1, 1, 0.1**2,'BC')

    Nr  = 61;         N2 = (Nr-1)/2
    Nz  = 30;          M  = Nz/2

    ktv = 2

    # Define Geometry and Differentiation Matrices
    Dr,Dr2,r,Dz,Dz2,z = geometry(Nr,Nz,parms)

    # Build Laplacian
    Lap = build_Lap(r,z,Dr,Dr2,Dz,Dz2,parms)

    # Build Profile and set up reduced matrix
    rr, zz = np.meshgrid(r,z)
    rin = rr[1:Nz-1,1:N2+1]
    zin = zz[1:Nz-1,1:N2+1]

    # Build basic state
    Lz = parms.Lz
    f  = parms.f
    Lh = parms.Lh
    Lv = parms.Lv
    Bu = parms.Bu
    Ro = parms.Ro

    # Defining the Basic State
    ## Note: Pr is 1/r*partial_r*Psi
    gaus = Ro*np.exp((-rin**2)/(Lh**2)-((zin+Lz/2)**2)/(Lv**2))
    Psib = -f*Lh**2/4*gaus
    Pr_b =  f/2*gaus
    Q_b  = -f*((rin**2 - Lh**2)/Lh**2+Bu*(Lh/Lv)**2*((zin+Lz/2)**2-0.5*Lv**2)/Lv**2)*gaus
    Qr_b =2*f/Lh**2*((rin**2-2*Lh**2)/Lh**2+Bu*(Lh/Lv)**2*((zin+Lz/2)**2-0.5*Lz**2)/Lz**2)*gaus
    #print ((zin+Lz/2)**2)/(Lv**2)

    cnt = 0
    for kt in range(2,3): #ktv=1
        # Build A and B for eigen-analysis

        A, B = build_AB(kt,r,z,Lap,Pr_b,Qr_b,parms)

        # Using eig
        eigVals, eigVecs = spalg.eig(A.todense(),B.todense())
        ind = (-np.imag(eigVals)).argsort() #get indices in descending order
        eigVecs = eigVecs[:,ind]
        eigVals = kt*eigVals[ind]
        
        ii = 0
        omega1 = kt*eigVals[:]
        growth1 = omega1[ii].imag
        freq1   = omega1[ii].real

        # Plotting for eig
        psi1 = eigVecs[:,ii].reshape([Nz-2,N2],order='F')

        lvlr = np.linspace(psi1.real.min(),psi1.real.max(),20)

        plt.rcParams["axes.titlesize"] = 8
        plt.xlabel('r')
        plt.ylabel('z')
        plt.contourf(r[1:N2+1]/1e3, z[1:Nz-1]/1e3, psi1.real, levels=lvlr)
        plt.colorbar()
        plt.title(['   m = ', (kt),
        '   ii = ', (ii),
        '   gr (direct) = ', (omega1[ii].imag )])
        plt.show()

        # Using eigs
        sig0 = eigVals[ii]
        evals_all, evecs_all = eigs(A,10,B,ncv=21,which='LI',maxiter=500,sigma=sig0)

        growth2 = (evals_all[ii]*kt).imag
        freq2   = (evals_all[ii]*kt).real

        # Plotting for eigs
        psi2 = evecs_all[:,ii].reshape([Nz-2,N2],order='F')

        lvlr2 = np.linspace(psi2.real.min(),psi2.real.max(),20)
        plt.xlabel('r')
        plt.ylabel('z')
        plt.contourf(r[1:N2+1]/1e3, z[1:Nz-1]/1e3, psi2.real, levels=lvlr2)
        plt.colorbar()
        plt.title(['   m = ', (kt),
        '   ii = ', (ii),
        '   gr (indirect) = ', (evals_all[ii]*kt ).imag])

        plt.show()

        print " "
        print "kt = ", kt, "and ii = ", ii
        print "  Direct Method: growth = ", growth1;
        print "Indirect Method: growth = ", growth2

        cnt = cnt + 1

