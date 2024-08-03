from ase.neighborlist import NeighborList, natural_cutoffs
from ase.atoms import Atoms
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
import numpy as np

def get_atoms(atomslike, copy=False):
    if isinstance(atomslike, Atoms):
        if copy:
            return atomslike.copy()
        return atomslike
    elif isinstance(atomslike, Structure):
        return AseAtomsAdaptor().get_atoms(atomslike)
    else:
        raise ValueError("atomslike must be an ase.Atoms or pymatgen.Structure object.")
    
def calculate_gcn(atoms_in, cm=None, cutoffs=None):
    """Calculate the generalized coordination number of an Atoms object.

    Args:
        atoms_in (ase.Atoms): A slab tagged along the z axis.
        cm (np.ndarray, optional): Connectivity matrix. Defaults to None.
        cutoffs (float or List[float], optional): Cutoffs for neighbor list calculation. Defaults to None.

    Returns:
        np.array: Generalized coordination number of each atom in the slab.
    """
    new_atoms = get_atoms(atoms_in, copy=True)
    atoms = new_atoms.repeat(
        (5, 5, 1),
    )
    if cutoffs == None:
        cutoffs = natural_cutoffs(atoms, mult=1.1)

    if cm is None:
        nl = NeighborList(cutoffs=cutoffs, self_interaction=False, bothways=True)
        nl.update(atoms)
        cm = nl.get_connectivity_matrix()

    cn = cm.sum(axis=1)
    cn_max = cn.max()
    gcn = np.array(cm * cn / cn_max).squeeze()
    return gcn[: len(atoms_in)]