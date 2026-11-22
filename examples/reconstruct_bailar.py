"""Reconstruct a descriptor interpolation along the Bailar-twist endpoints."""

from pathlib import Path

import hickle
import numpy as np
import torch
from ase.io import read, write

from invertatomrep.descriptor import DescriptorCalculator, optimize_invert_stack_invert
from invertatomrep.utils import convert_coords_to_frames, extract_descriptor

torch.set_default_dtype(torch.float64)

NU = 4
LMAX_BY_NU = {2: 8, 3: 6, 4: 4}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PARITY = 0
USE_MSE = False
SEEDS = (2244, 4908421, 41517, 183570, 24151)
ANGLE_TOLERANCE = {2: np.pi / 10, 3: np.pi / 8, 4: np.pi / 6}[NU]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BAILAR_TWIST_PATH = PROJECT_ROOT / "data" / "bailar_twist" / "actual_bailar_twist.xyz"
OUTPUT_DIR = PROJECT_ROOT / "results" / "bailar_twist"


def load_bailar_twist():
    """Load the bundled reference path as coordinate tensors."""
    frames = read(BAILAR_TWIST_PATH, ":")
    atom_counts = {len(frame) for frame in frames}
    assert len(atom_counts) == 1, f"All Bailar-twist frames must contain the same number of atoms: {sorted(atom_counts)}"
    return [torch.as_tensor(frame.positions, dtype=torch.float64) for frame in frames]


def compute_descriptor(calculator, coordinates):
    descriptor = calculator.compute_optimization_descriptor(
        coordinates,
        parity=PARITY,
        use_structure=False,
        atom_index=None,
    )
    return extract_descriptor(descriptor)


def move_tensors_to_cpu(values):
    return {
        key: value.detach().cpu() if torch.is_tensor(value) else value
        for key, value in values.items()
    }


def main():
    reference_coordinates = load_bailar_twist()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    calculator = DescriptorCalculator(
        cutoff_type="radial_cutoff",
        lmax=LMAX_BY_NU[NU],
        rcut=4.0,
        correlation_order=NU,
        dtype=torch.float64,
        device=DEVICE,
    )
    optimization_options = {
        "parity": PARITY,
        "max_outer_iter": 7,
        "max_inner_iter": 1000,
        "lr": 1e-2,
        "refine_lr": 5e-3,
        "min_loss": 1e-7,
        "patience": 3,
        "verbose": True,
        "use_structure": False,
        "atom_index": None,
        "mse": USE_MSE,
        "jitter_scale": 2e-1,
        "refine_max_inner_iter": 7,
        "angle_tolerance": ANGLE_TOLERANCE,
    }


    start_descriptor = compute_descriptor(calculator, reference_coordinates[0])
    end_descriptor = compute_descriptor(calculator, reference_coordinates[-1])
    interpolation_values = torch.linspace(0.0, 1.0, len(reference_coordinates))
    target_descriptors = [(1 - value) * start_descriptor + value * end_descriptor for value in interpolation_values]
    descriptor_path = OUTPUT_DIR / f"target_descriptors_nu{NU}_lmax{LMAX_BY_NU[NU]}.hickle"
    hickle.dump(target_descriptors, descriptor_path)

    number_of_initial_points = 2 * len(reference_coordinates[0])
    for seed in SEEDS:
        tag = f"nu{NU}_lmax{LMAX_BY_NU[NU]}_interpolation_seed{seed}"
        run_directory = OUTPUT_DIR / tag
        details_directory = run_directory / "details"
        details_directory.mkdir(parents=True, exist_ok=True)

        recovered_coordinates = []
        best_losses = {}
        for step, (value, target_descriptor) in enumerate(
            zip(interpolation_values, target_descriptors)
        ):
            points, loss, details = optimize_invert_stack_invert(
                                                        target_descriptor=target_descriptor.reshape(-1),
                                                        calculator=calculator,
                                                        n_points=number_of_initial_points,
                                                        device=DEVICE,
                                                        return_details=True,
                                                        seed=seed,
                                                        **optimization_options,
                                                    )
            points = points.detach().cpu()
            loss_value = float(loss.detach().cpu()) if torch.is_tensor(loss) else float(loss)

            details.update(step=step, seed=seed, t=float(value))
            hickle.dump(move_tensors_to_cpu(details), details_directory / f"recovered_{step:02d}.hickle",)
            recovered_coordinates.append(points)
            best_losses[float(value)] = loss_value
            print(f"step={step:02d}, t={float(value):.3f}, loss={loss_value:.6e}, points={len(points)}, stacked={len(details['stacked_points'])}", flush=True)
        hickle.dump(best_losses, run_directory / "best_losses.hickle")
        write(run_directory / "recovered_frames.xyz", convert_coords_to_frames(recovered_coordinates))
        print(f"Finished seed {seed}; results written to {run_directory}", flush=True)


if __name__ == "__main__":
    main()

