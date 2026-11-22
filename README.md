# desc2atom

Reconstruct atomic environments from atom-density based descriptors. Many machine-learning potentials encode local geometry as a descriptor
(power spectrum, bispectrum, ...). Here we probe the inverse question, namely,

> given a descriptor, can we recover the neighbor positions that produced it?

The core workflow is as follows 
1. **Target** — compute or load a descriptor for an environment.
2. **Invert** — optimize a local environment (positions of neighboring atoms) so that the descriptor centered on the origin matches the target.
3. **Inspect** — write recovered coordinates (XYZ) and the final loss.
---

## Environment

Python 3.12 is recommended.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install 'cmake<4'
uv pip install e3nn==0.5 metatensor==0.2.0 metatensor-core==0.1.12 \
  metatensor-learn==0.3.1 metatensor-operations==0.3.2 metatensor-torch==0.7.3 \
  torch==2.3.1 featomic==0.6.1 featomic-torch==0.6.1 \
  opt-einsum==3.3.0 scikit-learn==1.5.0 hickle ase pymatgen
```
---

## Tutorial: Reconstruct atomic environments across a descriptor path

Consider, for example, `examples/interpolate_coordination_envs.py`. This script considers the interpolation of descriptors between an environment with 5 neighbors (trigonal bipyramid) and 6 neighbors (octahedron), as defined in `invertatomrep/simple_geometries.py`.

The script first builds a descriptor trajectory

```text
trigonal bipyramid  --> d0
octahedron          --> d1

for t in [0, 1]:
    target = (1 - t) * d0 + t * d1     # interpolate in descriptor space
    local_environment <-- invert(target)             # recover neighbor coordinates relative to the origin
```
The target descriptors is obtained as a linear interpolation between the descriptors corresponding to the two endpoints. 

### Main pieces in the code

| Piece | Role |
|---|---|
| `DescriptorCalculator` | Builds atom-density descriptors for given `(lmax, rcut, correlation_order)`. |
| `compute_descriptor(...)` | Evaluates the descriptor on a tensor specifying the atomic environment. |
| `optimize_invert_stack_invert(...)` | Starts from an overcomplete set of atoms, collapses atoms that are along the same radial direction (i.e. those whose angular separation falls below a threshold), and refines until the descriptor loss is small. |
| `results/coordination_interpolation/` | Output directory for XYZ frames, losses, and optimization details. |

### Run it

```bash
python -m examples/interpolate_coordination_envs.py
```

Hyperparameters are defined at the top of the file:

- `NU` - correlation order (2 / 3 / 4) of the descriptor
- `LMAX_BY_NU` - maximum angular channels per correlation order
- `NSTEPS` — number of interpolation steps along the linear path between the descriptors
- `SEEDS` — random seeds for independent inversions
- `DEVICE` — `"cuda"` when available, otherwise `"cpu"`
- `USE_MSE` - whether to use mean square loss for optimization or MAE

Outputs for each seed are saved under
```text
results/coordination_interpolation/nu{NU}_lmax{L}_interpolation_seed{SEED}/
  recovered_frames.xyz
  best_losses.hickle
  details/
```

The recovered frames `recovered_frames.xyz` can be visualized in any molecular viewer. 

---

## Other examples

Building on this example, other scripts in this repository more or less reuse the same idea
with different targets:

| Script | Target | Data |
|---|---|---|
| `examples/reconstruct_bailar.py` | Descriptor interpolation along the Bailar twist | `data/bailar_twist/actual_bailar_twist.xyz` |
| `examples/reconstruct_qm9.py` | Recover descriptors corresponding to precomputed QM9 environment environments | `data/qm9/qm9_envs_crystalnn_every50.hickle` (+ feature hickle) |
| `examples/invert_nu3_degen.py` | Recover environments that share accidentally bispectrum-degenerate | `data/boron/boron_10_pbeccpvdz.xyz` |

```bash
python -m examples/reconstruct_bailar.py
```

QM9:
```bash
python -m examples/reconstruct_qm9.py --start 0 --stop 10
```
To repeat the same reconstruction for environments from MP20 or tmQM, replace `ENVIRONMENTS_PATH` and `FEATURE_PATH` in `examples/reconstruct_qm9.py` with the corresponding files under `data/mp20/` and `data/tmqm/`.

Accidental descriptor degeneracy (builds environments from the XYZ on first run):

```bash
python -m examples/invert_nu3_degen.py
```
