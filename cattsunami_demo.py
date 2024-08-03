from fairchem.applications.cattsunami.core import Reaction
from fairchem.data.oc.core import Slab, Adsorbate, Bulk, AdsorbateSlabConfig
from fairchem.core.common.relaxation.ase_utils import OCPCalculator
from ase.optimize import BFGS
from x3dase.visualize import view_x3d_n
from ase.io import read
from x3dase.x3d import X3D
from fairchem.applications.cattsunami.databases import DISSOCIATION_REACTION_DB_PATH
from fairchem.data.oc.databases.pkls import ADSORBATE_PKL_PATH, BULK_PKL_PATH
from fairchem.core.models.model_registry import model_name_to_local_file
import matplotlib.pyplot as plt
from fairchem.applications.cattsunami.core.autoframe import AutoFrameDissociation
from fairchem.applications.cattsunami.core import OCPNEB
from ase.io import read

#Optional
from IPython.display import Image
from x3dase.x3d import X3D

#Set random seed
import numpy as np
np.random.seed(22)



ADSORBATE_PKL_PATH = "/home/jovyan/shared-scratch/kabdelma/high_miller_idx/updated_adsorbates.pkl"
DISSOCIATION_REACTION_DB_PATH =  "/home/jovyan/shared-scratch/kabdelma/high_miller_idx/updated_reactions.pkl"

# Instantiate the reaction class for the reaction of interest
reaction = Reaction(reaction_str_from_db="*O2 -> *O + *O",
                    reaction_db_path=DISSOCIATION_REACTION_DB_PATH,
                    adsorbate_db_path = ADSORBATE_PKL_PATH)


# Instantiate our adsorbate class for the reactant and product
reactant = Adsorbate(adsorbate_id_from_db=reaction.reactant1_idx, adsorbate_db_path=ADSORBATE_PKL_PATH)
product1 = Adsorbate(adsorbate_id_from_db=reaction.product1_idx, adsorbate_db_path=ADSORBATE_PKL_PATH)
product2 = Adsorbate(adsorbate_id_from_db=reaction.product2_idx, adsorbate_db_path=ADSORBATE_PKL_PATH)

# Grab the bulk and cut the slab we are interested in
bulk = Bulk(bulk_src_id_from_db="mp-30", bulk_db_path=BULK_PKL_PATH)
slab = Slab.from_bulk_get_specific_millers(bulk = bulk, specific_millers=(2,1,1))


# Perform site enumeration
# For AdsorbML num_sites = 100, but we use 3 here for brevity. This should be increased for practical use.
reactant_configs = AdsorbateSlabConfig(slab = slab[0], adsorbate = reactant,
                                       mode="random_site_heuristic_placement",
                                       num_sites = 100).atoms_list
product1_configs = AdsorbateSlabConfig(slab = slab[0], adsorbate = product1,
                                      mode="random_site_heuristic_placement",
                                      num_sites = 100).atoms_list
product2_configs = AdsorbateSlabConfig(slab = slab[0], adsorbate = product2,
                                      mode="random_site_heuristic_placement",
                                      num_sites = 100).atoms_list

# Instantiate the calculator
# NOTE: If you have a GPU, use cpu = False
# NOTE: Change the checkpoint path to locally downloaded files as needed
checkpoint_path = model_name_to_local_file('EquiformerV2-31M-S2EF-OC20-All+MD', local_cache='/tmp/ocp_checkpoints/')
cpu = False
calc = OCPCalculator(checkpoint_path = checkpoint_path, cpu = cpu)


# Relax the reactant systems
reactant_energies = []
for config in reactant_configs:
    config.calc = calc
    opt = BFGS(config)
    opt.run(fmax = 0.05, steps=200)
    reactant_energies.append(config.get_potential_energy())


# Relax the product systems
product1_energies = []
for config in product1_configs:
    config.calc = calc
    opt = BFGS(config)
    opt.run(fmax = 0.05, steps=200)
    product1_energies.append(config.get_potential_energy())

product2_energies = []
for config in product2_configs:
    config.calc = calc
    opt = BFGS(config)
    opt.run(fmax = 0.05, steps=200)
    product2_energies.append(config.get_potential_energy())

print(np.sort(reactant_energies)[:16])

af = AutoFrameDissociation(
            reaction = reaction,
#             reactant_system = reactant_configs[reactant_energies.index(min(reactant_energies))],
            reactant_system = reactant_configs[np.argsort(reactant_energies)[15]],
            product1_systems = product1_configs,
            product1_energies = product1_energies,
            product2_systems = product2_configs,
            product2_energies = product2_energies,
            r_product1_max=2, #r1 in the above fig
            r_product2_max=3, #r3 in the above fig
            r_product2_min=1, #r2 in the above fig
)


nframes = 10
frame_sets, mapping_idxs = af.get_neb_frames(calc,
                               n_frames = nframes,
                               n_pdt1_sites=4, # = 5 in the above fig (step 1)
                               n_pdt2_sites = 4, # = 5 in the above fig (step 2)
                              )


# This will run all NEBs enumerated - to just run one, run the code cell below.
# On GPU, each NEB takes an average of ~1 minute so this could take around a half hour on GPU
# But much longer on CPU
# Remember that not all NEBs will converge -- the k, nframes would be adjusted to achieve convergence

fmax = 0.05 # [eV / ang**2]
delta_fmax_climb = 0.4
converged_idxs = []

for idx, frame_set in enumerate(frame_sets):
    neb = OCPNEB(
        frame_set,
        checkpoint_path=checkpoint_path,
        k = 1,
        batch_size=8,
        cpu = False,
    )
    optimizer = BFGS(
        neb,
        trajectory=f"cattsunami_trajs/211/O2_dissoc_on_Cu_{idx}.traj",
    )
    conv = optimizer.run(fmax=fmax + delta_fmax_climb, steps=200)
    if conv:
        neb.climb = True
        conv = optimizer.run(fmax=fmax, steps=300)
        if conv:
            converged_idxs.append(idx)
            
print(converged_idxs)