"""Invert descriptors for pairs of nu=3-degenerate Boron environments."""

from pathlib import Path

import hickle
import torch
from ase.io import read, write
from tqdm import tqdm

from invertatomrep.descriptor import DescriptorCalculator, optimize_to_N_points
from invertatomrep.utils import convert_coords_to_frames, extract_descriptor

torch.set_default_dtype(torch.float64)

DEVICE = "cpu"
ORIGIN_INDEX = 0
PARITY = 0

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BORON_XYZ = PROJECT_ROOT / "data" / "boron" / "boron_10_pbeccpvdz.xyz"
OUTPUT_DIR = PROJECT_ROOT / "results" / "nu3_degenerate"

PAIR_START = 0
PAIR_STOP = None  # None => all pairs

NTRIALS = 100 # number of inversion trials for each pair
SEED_BASE = 2733247 # starting seed 

# threshold to assign closest structure, ambiguous if |d0 - d1| / (d0 + d1) is below this relative L2 margin
AMBIG_REL = 0.05

# nu=3 calculator to generate the inversion target (Both structures in the pair A and B share this descriptor)
calculator = DescriptorCalculator(cutoff_type="radial_cutoff",
                                lmax=5,
                                rcut=5.0,
                                correlation_order=3,
                                dtype=torch.float64,
                                device=DEVICE,
                                )
# nu=4 calculator to decide which structure of the pair, A or B, is the recovered environment closest to
nu4_calculator = DescriptorCalculator(
    cutoff_type="radial_cutoff",
    lmax=4,
    rcut=5.0,
    correlation_order=4,
    dtype=torch.float64,
    device=DEVICE,
)

USE_MSE = True

_feat_suffix = f"cutoff{calculator.rcut}lmax{calculator.lmax}nu3"
ENVS_PATH = OUTPUT_DIR / "boron_envs.hickle"
FEAT_PATH = OUTPUT_DIR / f"boron_feats_{_feat_suffix}.hickle"


def compute_descriptor_tensor(calc, env, parity=PARITY):
    desc = calc.get_descriptor(
                                env,
                                parity=parity,
                                use_structure=False,
                                coordinates="relative",
                            )
    return list(extract_descriptor(desc, stack_parities=True))[0]


def build_envs_from_xyz():
    """Build local envs ordered A0, B0, A1, B1, ... from consecutive xyz frames."""
    frames = read(str(BORON_XYZ), ":")
    n_pairs = len(frames) // 2
    if len(frames) % 2 != 0:
        print(f"Warning: XYZ has {len(frames)} frames; using the first {n_pairs} complete A/B pairs.", flush=True)

    envs = []
    print(f"Building envs from {BORON_XYZ} ({n_pairs} pairs)", flush=True)
    for pair_id in tqdm(range(n_pairs), desc="envs"):
        for ab in (0, 1):
            coords = torch.from_numpy(frames[2 * pair_id + ab].positions)
            env0, _ = calculator._local_geometry_from_tensor(coords, atom_index=ORIGIN_INDEX)
            envs.append(env0)

    hickle.dump(envs, ENVS_PATH)
    print(f"Saved {len(envs)} envs to {ENVS_PATH}", flush=True)
    return envs


def build_feats_from_envs(envs):
    """Just one nu=3 feature per pair (descriptor from A, since it is the same for both A and B)."""
    if len(envs) % 2 != 0:
        raise ValueError(f"Expected even number of envs, got {len(envs)}")

    desc_all = []
    n_pairs = len(envs) // 2
    print(f"Computing descriptors for {n_pairs} pairs", flush=True)
    for pair_id in tqdm(range(n_pairs), desc="descriptors"):
        i_a = 2 * pair_id  # A; B at i_a+1 is degenerate with same target descriptor
        desc_all.append(compute_descriptor_tensor(calculator, envs[i_a]))

    hickle.dump(desc_all, FEAT_PATH)
    print(f"Saved {len(desc_all)} pair descriptors to {FEAT_PATH}", flush=True)
    return desc_all


def load_inputs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if ENVS_PATH.exists():
        environments = hickle.load(ENVS_PATH)
        print(f"Loaded {len(environments)} environments from {ENVS_PATH}", flush=True)
    else:
        environments = build_envs_from_xyz()

    if FEAT_PATH.exists():
        descriptors = hickle.load(FEAT_PATH)
        print(f"Loaded {len(descriptors)} descriptors from {FEAT_PATH}", flush=True)
    else:
        descriptors = build_feats_from_envs(environments)

    if len(environments) != 2 * len(descriptors):
        raise ValueError(f"Expected two environments for each descriptor; got {len(environments)} environments and {len(descriptors)} descriptors.")
    return environments, descriptors


envs_all = []
X_all = []
out_dir = OUTPUT_DIR

OPT_KW = { "device": DEVICE,
    "angle_tolerance": 0.1,
    "origin_tol": 1e-6,
    "max_outer_iter": 10,
    "max_inner_iter": 1000,
    "lr": 1e-2,
    "min_loss": 1e-14,
    "patience": 2,
    "collapse_every": 500,
    "log_every": 50,
    "verbose": True,
    "parity": PARITY,
    "use_structure": False,
    "atom_index": None,
    "mse": USE_MSE,
}


def to_device(x):
    if isinstance(x, torch.Tensor):
        return x.to(device=DEVICE, dtype=calculator.dtype)
    return torch.as_tensor(x, device=DEVICE, dtype=calculator.dtype)


def pair_env_indices(pair_id):
    """Frame indices in envs_all for A and B of one degenerate pair."""
    return 2 * pair_id, 2 * pair_id + 1


def invert_one_pair(pair_id, seed):
    target = to_device(X_all[pair_id]).reshape(-1)
    i_a, _ = pair_env_indices(pair_id)
    ref = to_device(envs_all[i_a])
    n_init = len(envs_all[i_a]) * 3 # num initial atoms
    return optimize_to_N_points(
                        target_descriptor=target,
                        calculator=calculator,
                        reference_structure=ref,
                        n_points=n_init,
                        seed=seed,
                        **OPT_KW,
                    )


def save_one_pair(pair_id, trial, seed, recovered_pts, best_loss, f_log):
    recovered_pts = recovered_pts.detach().cpu()
    xyz_path = out_dir / f"recovered_pair{pair_id:04d}_trial{trial:04d}.xyz"
    recovered_xyz = convert_coords_to_frames([torch.vstack((torch.zeros(1, 3, dtype=recovered_pts.dtype), recovered_pts))])
    frame = recovered_xyz[0]
    loss_val = float(best_loss.detach().cpu().item()) if torch.is_tensor(best_loss)  else float(best_loss)
    i_a, i_b = pair_env_indices(pair_id)
    n_recovered = len(recovered_pts)
    n_atoms_xyz = len(frame)
    frame.info["pair_id"] = pair_id
    frame.info["trial"] = trial
    frame.info["seed"] = seed
    frame.info["best_loss"] = loss_val
    frame.info["env_index_A"] = i_a
    frame.info["env_index_B"] = i_b
    frame.info["n_recovered"] = n_recovered
    frame.info["n_atoms_xyz"] = n_atoms_xyz
    write(xyz_path, frame)
    line = f"{pair_id}\t{trial}\t{seed}\t{i_a}\t{i_b}\t{n_recovered}\t{n_atoms_xyz}\t{loss_val:.8e}\t{xyz_path.name}\n"
    f_log.write(line)
    f_log.flush()
    print(line.rstrip(), flush=True)


def run_inversion(pair_start, pair_end):
    loss_log = out_dir / f"recovered_losses_pairs_{pair_start}_{pair_end}.txt"
    print(f"Inverting pairs [{pair_start}, {pair_end}), {NTRIALS} trials each", flush=True)

    with loss_log.open("w", encoding="utf-8", buffering=1) as f_log:
        f_log.write(  "# pair_id\ttrial\tseed\tenv_A\tenv_B\tn_recovered\tn_atoms_xyz\tbest_loss\txyz_file\n")
        for pair_id in tqdm(range(pair_start, pair_end)):
            for trial in range(NTRIALS):
                seed = SEED_BASE + trial
                result = invert_one_pair(pair_id, seed)
                save_one_pair(pair_id, trial, seed, *result, f_log)

    print(f"Results written to {out_dir}", flush=True)


def nu4_descriptor_from_env(env):
    """nu=4 descriptor used to decide which pair member is closer."""
    return compute_descriptor_tensor(nu4_calculator, env)


def nu4_descriptor_from_structure(positions):
    positions = torch.as_tensor(positions, dtype=nu4_calculator.dtype)
    environment, neighbor_weights = nu4_calculator._local_geometry_from_tensor(positions, atom_index=ORIGIN_INDEX)
    descriptor = nu4_calculator.get_descriptor(environment, parity=PARITY, use_structure=False, coordinates="relative", local_weights=neighbor_weights)
    return list(extract_descriptor(descriptor, stack_parities=True))[0]


def assign_closest_structure(phi_r, phi_0, phi_1, ambig_rel=AMBIG_REL):
    """Assign recovered structure to pair member 0 or 1 by L2 distance in nu=4 space."""
    d0 = torch.norm(phi_r - phi_0).item()
    d1 = torch.norm(phi_r - phi_1).item()
    rel_diff = abs(d0 - d1) / (d0 + d1 + 1e-30)
    if rel_diff < ambig_rel:
        return "ambiguous", d0, d1
    return (0 if d0 < d1 else 1), d0, d1


def find_closest_structure(pair_start, pair_end):
    recovered_paths = sorted(out_dir.glob("recovered_pair*_trial*.xyz"))
    if not recovered_paths:
        print("No recovered frames found; skipping closest-structure assignment.", flush=True)
        return

    print("Closest structure: pair member 0 or 1 (based on nu=4 L2)", flush=True)

    for pair_id in range(pair_start, pair_end):
        i_0, i_1 = pair_env_indices(pair_id)
        phi_0 = nu4_descriptor_from_env(envs_all[i_0])
        phi_1 = nu4_descriptor_from_env(envs_all[i_1])

        for path in recovered_paths:
            if not path.name.startswith(f"recovered_pair{pair_id:04d}_"):
                continue
            rframe = read(path)
            if rframe.info.get("pair_id", pair_id) != pair_id:
                continue
            best_loss = rframe.info.get("best_loss", float("inf"))
            loss_value = float(best_loss.detach().cpu()) if torch.is_tensor(best_loss) else float(best_loss)
            trial = rframe.info.get("trial", path.stem)
            phi_r = nu4_descriptor_from_structure(rframe.positions)
            closest, d0, d1 = assign_closest_structure(phi_r, phi_0, phi_1)
            print(f"pair={pair_id} trial={trial} closest_structure={closest} d0={d0:.6e} d1={d1:.6e} loss={loss_value:.3e}", flush=True)


def main():
    print(f"Using MSE: {USE_MSE}", flush=True)
    envs_all, descriptors_all = load_inputs()
    n_pairs = len(descriptors_all)
    pair_end = n_pairs if PAIR_STOP is None else PAIR_STOP
    if not 0 <= PAIR_START < pair_end <= n_pairs:
        raise ValueError(f"Invalid pair range [{PAIR_START}, {pair_end}) for {n_pairs} pairs.")

    run_inversion(PAIR_START, pair_end)
    find_closest_structure(PAIR_START, pair_end)


if __name__ == "__main__":
    main()
