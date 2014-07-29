qg_vortex
=========
qg_vortex_np.py: uses numpy only (can choose to use either eig or eigs)

qg_vortex_pc.py: uses petsc. Can now plot in parallel

qg_vortex_pc_io.py: similar to qg_vortex_pc.py except it outputs files: eigVecs, eigVals, InputData, nconvData; which can be used when you run read_qg_vortex_io.py.

read_qg_vortex_io.py: To use this, run qg_vortex_pc_io.py first (to make the files, then run this. If you're using this to view the plots, you can get rid of the if statement in the solve_eigensystem in qg_vortex_pc_io.py that has to do with plotting.
