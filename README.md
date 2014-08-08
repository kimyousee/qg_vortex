qg_vortex
=========
Based off of the qg_vortex_stab_eigs.m code by Francis J. Poulin in collaboration with Claire Menesguen

qg_vortex_np.py: uses numpy only (can choose to use either eig or eigs)

qg_vortex_pc.py: uses petsc. Can now plot in parallel

qg_vortex_pc_io.py: similar to qg_vortex_pc.py except it outputs files: eigVecs, eigVals, InputData, nconvData; which can be used when you run read_qg_vortex_io.py.

read_qg_vortex_io.py: To use this, run qg_vortex_pc_io.py first (to make the files, then run this) It plots both real and imag parts of the eigenvalues
