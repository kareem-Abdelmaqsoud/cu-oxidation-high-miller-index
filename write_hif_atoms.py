# from adsorbate_slab_config_fixed_seed import AdsorbateSlabConfig
from ocdata.core import Adsorbate, Bulk, Slab 
from ocdata.utils.vasp import write_vasp_input_files
from ocdata.core import Adsorbate, AdsorbateSlabConfig, Bulk, Slab
import pickle
from ase.io import read
from glob import glob
import numpy as np
import pandas as pd
from dscribe.descriptors import SOAP
from tqdm import tqdm
from multiprocessing import Pool



df = pd.read_excel('experimental_data/cu_oxidation_per_dosage_fractions.xlsx')
df_low_miller = df.query("(6 < miller_1 <= 20) or (6 < miller_2 <= 20) or (6 < miller_3 <= 20)")
miller_indices = df_low_miller[["miller_1", "miller_2", "miller_3"]].values


all_adslabs = []
all_placement_nums = []
all_miller_indices = []
def write_atom_object(mp_args):
    _, miller_idx_split = mp_args
    for miller_idx in tqdm(miller_idx_split):
        bulk = Bulk(bulk_src_id_from_db = "mp-30",
        bulk_db_path = "/home/jovyan/shared-scratch/kabdelma/Open-Catalyst-Dataset/ocdata/databases/pkls/bulks.pkl")
        slabs = Slab.from_bulk_get_specific_millers(tuple(miller_idx), bulk = bulk,)
        slab = slabs[0]
        # oxygen adsorbate
        adsorbate = Adsorbate(adsorbate_id_from_db=0, adsorbate_db_path = "/home/jovyan/shared-scratch/kabdelma/Open-Catalyst-Dataset/ocdata/databases/pkls/adsorbates.pkl")
        adslab_heuristic  = AdsorbateSlabConfig(slab, adsorbate, mode= "random_site_heuristic_placement")
        adslabs = [*adslab_heuristic.atoms_list]
        all_adslabs.append(adslabs)
        all_placement_nums.append(list(np.arange(0,len(adslabs))))
        all_miller_indices.append([str(miller_idx)]*100)
        print(miller_idx)
    return all_adslabs,all_placement_nums,all_miller_indices


# Create a multiprocessing Pool
NUM_WORKERS = 8
# split up the dictionary that has all the atoms objects
splits = np.array_split(miller_indices, NUM_WORKERS)
# pool over these splits 
pool = Pool(NUM_WORKERS)     
mp_args = [(lmdb_idx, subsplit) for lmdb_idx, subsplit in enumerate(splits)]

all_outputs = list(pool.imap(write_atom_object, mp_args))

# combine the outputs from the pool
for output in all_outputs:
    all_adslabs += output[0]
    all_placement_nums += output[1]
    all_miller_indices += output[2]

all_miller_indices = np.array(all_miller_indices).reshape(-1,)
all_placement_nums = np.array(all_placement_nums).reshape(-1,)
all_adslabs = np.array(all_adslabs).reshape(-1,)

atoms_dict = {
    "all_miller_indices": all_miller_indices,
    "all_placement_nums": all_placement_nums,
    "atoms": all_adslabs,
}
df = pd.DataFrame(atoms_dict)
df.to_pickle('hif_adslabs_atoms_objects.pickle',protocol=pickle.HIGHEST_PROTOCOL)