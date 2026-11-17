from torch_geometric.datasets import QM9
import hickle 
dataset = QM9(root="data/QM9")
# save XYZ 
from ase.io import read, write
from io import StringIO
frames = []
with open("data/QM9/raw/gdb9.sdf") as f:
    block = []
    for line in f:
        if line.strip() == "$$$$":
            if block:
                mol_str = "".join(block)
                atoms = read(StringIO(mol_str), format="sdf")
                print(len(atoms))
                block = []
                frames.append(atoms)
        else:
            block.append(line)

write('data/QM9/qm9.xyz', frames)


# get local environments 
from invertatomrep.descriptor import DescriptorCalculator
calculator = DescriptorCalculator(
        cutoff_type="radial_cutoff",
        lmax=2,
        rcut=5.0,
        correlation_order=[3],
        # central_atom=,
        dtype=torch.float64,
        # cg = cg
    )

NSKIP = 50
DATA_PATH = 'data/QM9/qm9.xyz'
frames = read(str(DATA_PATH), ":")[::NSKIP]
print(f"{len(frames)} frames (NSKIP={NSKIP})", flush=True)

envs = []
for frame in tqdm(frames, desc="Environments"):
    struct = frame_to_structure(frame)
    env0, w0 = calculator.get_local_geometry(struct, atom_index=ATOM_INDEX)
    envs.append(env0)

hickle.dump(envs, 'qm9_envs.hickle')
