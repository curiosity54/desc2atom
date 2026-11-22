"""Reconstruct QM9 environments from precomputed descriptors."""

import argparse
from pathlib import Path

import hickle
import torch
from ase.io import write
from tqdm import tqdm

from invertatomrep.descriptor import DescriptorCalculator, optimize_to_N_points
from invertatomrep.utils import convert_coords_to_frames

torch.set_default_dtype(torch.float64)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CORRELATION_ORDER = 3
LMAX = 6
RCUT = 5.0
USE_MSE = True
MAX_ENVIRONMENTS = 1000
NSKIP = 50

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# modify this to the desired output directory
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "qm9"
FEATURE_PATH = PROJECT_ROOT / (
    f"qm9_feats_cutoff{RCUT}lmax{LMAX}nu[{CORRELATION_ORDER}]"
    f"_every{NSKIP}.hickle"
)
ENVIRONMENTS_PATH = PROJECT_ROOT / "data" / "qm9" / f"qm9_envs_crystalnn_every{NSKIP}.hickle"


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--start", type=int, default=0, help="First environment index"
    )
    parser.add_argument(
        "--stop",
        type=int,
        default=None,
        help="Last environment index (exclusive). Default: all loaded environments",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    calculator = DescriptorCalculator(
        cutoff_type="radial_cutoff",
        lmax=LMAX,
        rcut=RCUT,
        correlation_order=[CORRELATION_ORDER],
        dtype=torch.float64,
        device=DEVICE,
    )
    optimization_options = {
        "parity": 0,
        "max_outer_iter": 10,
        "max_inner_iter": 1000,
        "lr": 1e-2,
        "min_loss": 1e-12,
        "patience": 3,
        "collapse_every": 500,
        "verbose": True,
        "use_structure": False,
        "atom_index": None,
        "mse": USE_MSE,
        "seed": 579790,
    }

    descriptors = hickle.load(FEATURE_PATH)[:MAX_ENVIRONMENTS]
    environments = hickle.load(ENVIRONMENTS_PATH)[:MAX_ENVIRONMENTS]
    if len(descriptors) != len(environments):
        raise ValueError(f"Feature and environment files must contain the same number of entries; got {len(descriptors)} and {len(environments)}.")

    stop = len(descriptors) if args.stop is None else args.stop
    if not 0 <= args.start < stop <= len(descriptors):
        raise ValueError(f"Invalid range [{args.start}, {stop}) for {len(descriptors)} environments.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    loss_log = args.output_dir / f"losses_{args.start}_{stop}.txt"
    print(f"Inverting environments [{args.start}, {stop}) on {DEVICE}", flush=True)

    with loss_log.open("w", encoding="utf-8", buffering=1) as log:
        log.write("# index\toriginal_num_atoms\trecovered_num_atoms\tbest_loss\txyz_file\n")
        for index in tqdm(range(args.start, stop), desc="QM9 environments"):
            target = torch.as_tensor(descriptors[index], device=DEVICE, dtype=calculator.dtype).reshape(-1)
            reference = torch.as_tensor(environments[index], device=DEVICE, dtype=calculator.dtype)
            recovered, loss = optimize_to_N_points(
                                            target_descriptor=target,
                                            calculator=calculator,
                                            reference_structure=reference,
                                            n_points=len(reference) * 2,
                                            device=DEVICE,
                                            **optimization_options,
                                        )

            recovered = recovered.detach().cpu()
            loss_value = float(loss.detach().cpu()) if torch.is_tensor(loss) else float(loss)
            xyz_path = args.output_dir / f"recovered_{index:05d}.xyz"
            frame = convert_coords_to_frames([recovered])[0]
            frame.info["original_num_atoms"] = len(reference)
            frame.info["best_loss"] = loss_value
            write(xyz_path, frame)

            line = f"{index}\t{len(reference)}\t{len(recovered)}\t{loss_value:.8e}\t{xyz_path.name}\n"
            log.write(line)
            print(line.rstrip(), flush=True)

    print(f"Results written to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

