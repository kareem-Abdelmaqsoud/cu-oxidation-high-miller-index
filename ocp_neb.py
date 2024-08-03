# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
from ase.io import read, Trajectory
from ase.optimize import BFGS
from ase.neb import NEB, interpolate, idpp_interpolate
from ase.calculators.singlepoint import SinglePointCalculator
from ocpmodels.common.relaxation.ase_utils import OCPCalculator
from vasp_interactive import VaspInteractive
from ocdata.utils.vasp import calculate_surface_k_points

# -----------------------------------------------------------------------------
# NEB
# -----------------------------------------------------------------------------

# Calculation parameters.
phase = "surface" # gas | surface
fmax = 0.05 # [eV/A]
steps = 1000 # [-]
n_images = 10 # [-]
k_neb = 0.10 # [eV/A]
run_neb = True
restart = False

# Vasp parameters.
vasp_flags = {
    "ibrion": -1,
    "ediffg": -0.03,
    "encut": 350.0,
    "gga": "Bf",
    "isif": 0,
    "ispin": 1,
    "isym": 0,
    "laechg": True,
    "lbeefens": True,
    "lmaxmix": 2,
    "lreal": "Auto",
    "luse_vdw": True,
    "lwave": False,
    "lcharg": False,
    "ncore": 4,
    "nsw": 0,
    "symprec": 1e-10,
    "zab_vdw": -1.8867,
    'xc': "PBE",
}

# Read atoms objects.
if restart is True:
    images = read("neb.traj", f"-{n_images}:")
else:
    atoms_first = read('first/relax.traj')
    atoms_last = read('last/relax.traj')
    images = [atoms_first]
    images += [atoms_first.copy() for ii in range(n_images-2)]
    images += [atoms_last]
    interpolate(images=images, mic=False, apply_constraint=False)
    idpp_interpolate(
        images=images,
        traj=None,
        log=None,
        mic=False,
        steps=1000,
        fmax=0.01,
        optimizer=BFGS,
    )
    # Write trajectory.
    with Trajectory(filename="neb_initial.traj", mode='w') as traj:
        for atoms in images:
            traj.write(atoms)

# Set k-points.
if phase == "surface":
    kpts = calculate_surface_k_points(atoms_first)
else:
    kpts = None

# Setup NEB calculation.
neb = NEB(
    images=images,
    k=k_neb,
    climb=False,
    parallel=False,
    method='aseneb',
    allow_shared_calculator=True,
)

# Setup Vasp interactive calculator.
with OCPCalculator(checkpoint="/home/jovyan/shared-scratch/kabdelma/high_miller_idx/gnoc_oc22_oc20_all_s2ef.pt") as calc:
    for atoms in images:
        atoms.calc = calc
    for ii in (0, -1):
        images[ii].get_potential_energy()
        images[ii].calc = SinglePointCalculator(
            atoms=images[ii],
            energy=images[ii].calc.results["energy"],
            forces=images[ii].calc.results["forces"],
        )
    # Setup relax calculation.
    opt = BFGS(
        atoms=atoms,
        logfile="neb.log",
        trajectory="neb.traj",
    )
    if run_neb is True:
        # Run the NEB calculation.
        opt.run(fmax=fmax, steps=steps)
        converged = opt.converged()
        print(f"converged = {converged}")

        # Write trajectory.
        with Trajectory(filename="neb_final.traj", mode='w') as traj:
            for atoms in images:
                traj.write(atoms, **atoms.calc.results)

# -----------------------------------------------------------------------------
# END
# -----------------------------------------------------------------------------
