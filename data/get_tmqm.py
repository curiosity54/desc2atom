from atomic_datasets.datasets import tmQM
from invertatomrep.utils import extract_descriptor, generate_nu2_degen_structs, convert_coords_to_frames
from ase.visualize import view
import torch
import ase 

frames = []
dataset = tmQM(
    root_dir="data/",
    split="val",
)

for graph in dataset[::15]:
    positions = graph["nodes"]["positions"]   # (N, 3) array of 3D coordinates in Angstroms
    species = graph["nodes"]["species"]       # (N,) array of atomic species indices
    atom_types = graph["nodes"]["atom_types"] # 
    frames.append( ase.Atoms(''.join(atom_types), positions = positions, pbc = False))

NSKIP = 20
BOX_A = 30.0
from tqdm import tqdm
from pymatgen.core import Lattice, Structure
from invertatomrep.descriptor import DescriptorCalculator
import hickle
calculator = DescriptorCalculator(
    cutoff_type="crystalnn_cutoff",
    lmax=2,
    rcut=5.0,
    correlation_order=2,
    central_atom=True,
    dtype=torch.float64,
)

def frame_to_structure(frame, box=BOX_A):
    # Center in the box so the molecule is not sitting on a face/corner.
    pos = frame.get_positions().copy()
    pos -= pos.mean(axis=0)
    pos += box / 2
    return Structure(
        Lattice.cubic(box),
        frame.get_chemical_symbols(),
        pos,
        coords_are_cartesian=True,
    )


def filter_local_env(env, weights, max_dist):
    """Drop PBC image neighbors; keep central site (r=0) and real first-shell sites."""
    norms = torch.norm(env, dim=1)
    keep = norms <= max_dist + 1e-6
    return env[keep], weights[keep]



ATOM_INDEX = 0
print(
    f"{len(frames)} structures (NSKIP={NSKIP}, atom_index={ATOM_INDEX})",
    flush=True,
)

envs = []

for frame in tqdm(frames, desc="Environments"):
    struct = frame_to_structure(frame)
    env0, w0 = calculator.get_local_geometry(struct, atom_index=ATOM_INDEX)
    env0, _ = filter_local_env(env0, w0, calculator.rcut)
    envs.append(env0)
hickle.dump(envs, 'tmqm_envs_nskip15val.hickle')
