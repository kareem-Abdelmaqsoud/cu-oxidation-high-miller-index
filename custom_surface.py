# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
from ase.build import bulk, surface
from ase.constraints import FixAtoms

# -------------------------------------------------------------------------------------
# CUT SURFACE
# -------------------------------------------------------------------------------------

def cut_surface(
    atoms,
    vectors=[[1, 0], [0, 1]],
    origin=(0., 0.),
    tol=1e-5,
):
    """Cut a surface structure with surface vectors.

    Args:
        atoms ase.Atoms: Structure to cut.
        vectors (list, optional): Surface vectors used to cut the surface.
        Defaults to [[1, 0], [0, 1]].
        origin (tuple, optional): Origin of the cut. Defaults to (0., 0.).
        tol (float, optional): Tolerance use to avoid errors due to float precision. 
        Defaults to 1e-5.

    Returns:
        ase.Atoms: Structure cut.
    """

    # Wrap atoms into cell.
    atoms.wrap(eps=tol)
    
    # Transform vectors into arrays.
    vectors = np.array(vectors)
    if vectors[0,0] < 0 or vectors[1,1] < 0:
        raise RuntimeError("vectors[0,0] and vectors[1,1] must be > 0.")

    # Calculate the number of repetitions and the translation necessary.
    repeat = [vectors[0,0]+2, vectors[1,1]+2, 1]
    translate = [-1, -1]
    if vectors[0,1] >= 0:
        repeat[1] += vectors[0,1]
    else:
        repeat[1] -= vectors[0,1]
        translate[1] += vectors[0,1]
    if vectors[1,0] >= 0:
        repeat[0] += vectors[1,0]
    else:
        repeat[0] -= vectors[1,0]
        translate[0] += vectors[1,0]

    # Calculate the new cell.
    cell_2d = atoms.cell[:2,:2]
    cell_new = atoms.cell.copy()
    cell_new[:2,:2] = np.dot(vectors, cell_2d)
    
    # Translate and repeat the atoms.
    atoms.translate([*np.dot(translate, cell_2d), 0.])
    atoms *= [int(ii) for ii in np.ceil(repeat)]
    atoms.set_cell(cell_new)
    
    # Delete atoms outside the cell.
    scaled = atoms.get_scaled_positions(wrap=False) + tol
    del atoms[
        [ii for ii, pos in enumerate(scaled) if (pos < 0).any() or (pos > 1).any()]
    ]
    
    return atoms

# -------------------------------------------------------------------------------------
# REORDER STRUCTURE
# -------------------------------------------------------------------------------------

def reorder_surface(atoms, tol=1e-5):
    """Reorder the atoms of a surface according to the positions."""
    
    # Wrap atoms into cell.
    atoms.wrap(eps=tol)
    
    # Calculate layers and sort the atoms.
    order = np.zeros([len(atoms), 3], dtype=int)
    for axis in [0, 1, 2]:
        scaled = atoms.get_scaled_positions(wrap=False)[:, axis]
        values = [scaled[0]]
        for pos in scaled[1:]:
            if not np.isclose(pos, values, atol=tol, rtol=tol).any():
                values += [pos]
        values = np.sort(values)
        for ii, pos in enumerate(scaled):
            close = np.isclose(pos, values[::-1], atol=tol, rtol=tol)
            order[ii, axis] = np.where(close)[0][0]
    dummy = -(order[:,2]*1e12+order[:,1]*1e6+order[:,0])
    atoms = atoms[np.argsort(dummy)]
    
    return atoms

# -------------------------------------------------------------------------------------
# ENLARGE STRUCTURE
# -------------------------------------------------------------------------------------

def enlarge_structure(
    atoms,
    atoms_1x1,
    repetitions,
    repetitions_new,
    atoms_template = None,
):
    """Function to build a large supercell from a smaller one and
    reduce lateral interactions between periodic replica of adsorbates.

    Args:
        atoms (ase.Atoms): [description]
        atoms_1x1 (ase.Atoms): [description]
        repetitions (tuple): [description]
        repetitions_new (tuple): [description]

    Returns:
        ase.Atoms: [description]
    """
    
    atoms_x = atoms_1x1.copy()
    atoms_x *= (repetitions_new[0]-repetitions[0], repetitions[1], 1)
    atoms_x.translate([atoms.cell[0,0], atoms.cell[0,1], 0.])
    
    atoms_y = atoms_1x1.copy()
    atoms_y *= (repetitions_new[0], repetitions_new[1]-repetitions[1], 1)
    atoms_y.translate([atoms.cell[1,0], atoms.cell[1,1], 0.])
    
    ind = atoms.constraints[0].get_indices()
    ind_x = atoms_x.constraints[0].get_indices()+len(atoms)
    ind_y = atoms_y.constraints[0].get_indices()+len(atoms)+len(atoms_x)
    index = np.hstack((ind, ind_x, ind_y))
    
    atoms_large = atoms.copy()+atoms_x+atoms_y
    atoms_large.constraints[0].index = index
    
    atoms_large.set_cell([
        atoms_large.cell[0]*repetitions_new[0]/repetitions[0],
        atoms_large.cell[1]*repetitions_new[1]/repetitions[1],
        atoms_large.cell[2]
    ])
    
    return atoms_large

# -------------------------------------------------------------------------------------
# GET SURFACE
# -------------------------------------------------------------------------------------

def get_1x1_surface(
    lattice,
    miller_indices,
    layers,
    vacuum,
    vectors,
    del_indices,
    index_zero,
):
    
    atoms = surface(
        lattice=lattice,
        indices=miller_indices,
        layers=layers,
        vacuum=vacuum,
        periodic=True,
    )
    atoms = cut_surface(
        atoms=atoms,
        vectors=vectors,
    )
    del atoms[del_indices]
    atoms.rotate(atoms.cell[0], "x", rotate_cell=True)
    if index_zero is not None:
        atoms.translate(-atoms[index_zero].position)
    atoms.center(vacuum, 2)
    atoms.wrap()
    atoms = reorder_surface(atoms)
    indices = [
        aa.index for aa in atoms if aa.position[2] < atoms.cell[2,2]/2.-1e-3
    ]
    atoms.constraints.append(FixAtoms(indices=indices))

    return atoms

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():
    
    vacuum = 6.
    
    # Bulk.
    lattice = bulk(
        name="Cu",
        crystalstructure="fcc",
        a=3.677,
        cubic=True,
    )
    
    # # 100 surface.
    # atoms_100_1x1 = get_1x1_surface(
    #     lattice=lattice,
    #     miller_indices=(1,0,0),
    #     layers=3,
    #     vacuum=vacuum,
    #     vectors=[[1/2, 1/2], [-1/2, 1/2]],
    #     del_indices=[0],
    #     index_zero=4,
    # )
    # atoms_100 = atoms_100_1x1.copy()
    # atoms_100 *= (3,3,1)
    # atoms_100 = reorder_surface(atoms_100)
    # atoms_100.write("atoms_100.traj")
    
    # # 110 surface.
    # atoms_110_1x1 = get_1x1_surface(
    #     lattice=lattice,
    #     miller_indices=(1,1,0),
    #     layers=4,
    #     vacuum=vacuum,
    #     vectors=[[1/2, 0], [0, 1]],
    #     del_indices=[6, 7],
    #     index_zero=5,
    # )
    # atoms_110 = atoms_110_1x1.copy()
    # atoms_110 *= (3,2,1)
    # atoms_110 = reorder_surface(atoms_110)
    # atoms_110.write("atoms_110.traj")
    
    # 111 surface.
    atoms_111_1x1 = get_1x1_surface(
        lattice=lattice,
        miller_indices=(1,1,1),
        layers=4,
        vacuum=vacuum,
        vectors=[[1/2, 0], [0, 1/2]],
        del_indices=[],
        index_zero=3,
    )
    atoms_111 = atoms_111_1x1.copy()
    atoms_111 *= (3,3,1)
    atoms_111 = reorder_surface(atoms_111)
    atoms_111.write("atoms_111.traj")
    
    # 211 surface.
    atoms_211_1x1 = get_1x1_surface(
        lattice=lattice,
        miller_indices=(2,1,1),
        layers=6,
        vacuum=vacuum,
        vectors=[[1, 0], [0, 1/2]],
        del_indices=[],
        index_zero=10,
    )
    atoms_211 = atoms_211_1x1.copy()
    atoms_211 *= (1,4,1)
    atoms_211 = reorder_surface(atoms_211)
    atoms_211.write("atoms_211.traj")

    # 221 surface.
    atoms_221_1x1 = get_1x1_surface(
        lattice=lattice,
        miller_indices=(2,2,1),
        layers=8,
        vacuum=vacuum,
        vectors=[[1, 0], [-1/2, 1/2]],
        del_indices=[],
        index_zero=10,
    )
    atoms_221 = atoms_221_1x1.copy()
    atoms_221 *= (1,4,1)
    atoms_221 = reorder_surface(atoms_221)
    atoms_221.write("atoms_221.traj")
    
    # 321 surface.
    atoms_321_1x1 = get_1x1_surface(
        lattice=lattice,
        miller_indices=(3,2,1),
        layers=10,
        vacuum=vacuum,
        vectors=[[1/2, 0], [-1/2, 1]],
        del_indices=[],
        index_zero=17,
    )
    atoms_321 = atoms_321_1x1.copy()
    atoms_321 *= (1,2,1)
    atoms_321 = reorder_surface(atoms_321)
    atoms_321.write("atoms_321.traj")

    # from ase.gui.gui import GUI
    
    # gui = GUI([atoms_100, atoms_110, atoms_111, atoms_211, atoms_221, atoms_321])
    # gui.run()

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
