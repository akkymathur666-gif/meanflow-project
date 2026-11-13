## Results

All numbers below come from the completed local run (CPU, PyTorch 2.15.0, Python 3.12.4). Only the identity test and the toy experiments were run. **MNIST and CIFAR-10 have not been run yet** (no CUDA GPU was available), so there are no image results or FID scores.

### 1. MeanFlow identity check (`test_identity.py`)

All four checks pass: *MeanFlow identity and its JVP form hold numerically.*

| Check | Max abs. error |
|---|---|
| (1) \|u(z, t−h, t) − v(z, t)\| (finite-h limit) | 3.35e-07 |
| (2) \|u − (v − (t−r) du/dt)\| (total derivative) | 1.15e-10 |
| (3) \|v ∂ᵤu + ∂ₜu − du/dt\| | 9.98e-10 |
| (4) \|u − (v − (t−r) · JVP)\| | 3.67e-10 |

### 2. Toy data: 1-step MeanFlow vs multi-step Flow Matching

Sliced Wasserstein-2 (SW₂, lower is better), 10,000 training iterations (`python run_toy.py --iters 10000 --out results_toy`).
FM-*n* = Flow Matching with *n* Euler steps; MF-*n* = MeanFlow with *n* steps.

| dataset | noise floor | FM-1 | FM-8 | FM-128 | MF-1 | MF-2 | MF-8 |
|---|---|---|---|---|---|---|---|
| moons | 0.048 | 1.391 | 0.185 | 0.034 | 0.393 | 0.358 | 0.245 |
| 8gaussians | 0.070 | 1.390 | 0.081 | 0.069 | 0.350 | 0.241 | 0.260 |
| spiral | 0.045 | 0.944 | 0.095 | 0.031 | 0.266 | 0.136 | 0.097 |

**Findings**
- With one step, MeanFlow is 3.5× (moons), 4.0× (8gaussians) and 3.5× (spiral) better than Flow Matching.
- A second MeanFlow step helps on every dataset.
- Multi-step Flow Matching is still better at 8 and 128 steps in this small setup (FM-128 is close to the noise floor), so MeanFlow does **not** close the gap here.

Samples (columns: data, FM-1, FM-8, FM-128, MF-1, MF-2):

![moons](results_toy/samples_moons.png)
![8gaussians](results_toy/samples_8gaussians.png)
![spiral](results_toy/samples_spiral.png)

### 3. Ablations on 8gaussians (mean ± std, 3 seeds)

| config | SW₂ 1-step | SW₂ 2-step |
|---|---|---|
| default (p_neq=0.25, JVP, adaptive) | 0.297 ± 0.081 | 0.280 ± 0.042 |
| p_neq=0.0 (pure FM targets) | 0.286 ± 0.036 | 0.301 ± 0.025 |
| p_neq=0.5 | 97.986 ± 132.607 | 258.023 ± 354.766 |
| p_neq=1.0 | 4.700 ± 1.441 | 4.819 ± 2.497 |
| finite-difference instead of JVP | 0.285 ± 0.048 | 0.202 ± 0.005 |
| no adaptive loss weighting | 2.949 ± 0.804 | 1.960 ± 0.164 |

**Findings**
- Adaptive loss weighting is essential (about 10× worse without it).
- A finite-difference derivative is statistically indistinguishable from the JVP at 1 step.
- p_neq=0.5 diverged in at least one seed. p_neq=0 did not fail, which does **not** match Table 1a of the MeanFlow paper; the cause is not yet investigated.

### Limitations
- Toy 2-D data only, 10,000 iterations, CPU; ablations use 3 seeds.
- MNIST and CIFAR-10 (`train_images.py`) are implemented but not yet run.

### Reproduce
```bash
python test_identity.py
python run_toy.py --iters 10000 --out results_toy
# GPU required in practice:
python train_images.py --dataset mnist --iters 20000
python train_images.py --dataset cifar10 --iters 60000
```
