# MeanFlow — One-step Generative Modeling (Deep Learning 2026, DL-086)

From-scratch PyTorch implementation of **Mean Flows for One-step Generative Modeling**
(Geng, Deng, Bai, Kolter, He; arXiv:2505.13447, 2025).

Core contribution implemented here (`meanflow.py`): the **average-velocity network**
`u_θ(z, r, t)`, the **MeanFlow identity target** computed with one **JVP**
(`torch.func.jvp`), and the **1-step / few-step sampler**, compared against a
**flow-matching** baseline that uses the same network with `r = t`.

> **Status:** code written to spec; the MeanFlow identity (and its JVP form) is verified
> numerically by `test_identity.py` (numpy, passes). Training scripts have **not yet been
> run end-to-end** — run them on Colab and fill the results tables below.

## Files

| file | purpose |
|---|---|
| `meanflow.py` | networks (MLP, small U-Net), MeanFlow loss (JVP and finite-diff variants), FM loss, samplers, EMA, sliced-Wasserstein metric |
| `datasets.py` | 2-D toys: two moons, 8 Gaussians, spiral |
| `test_identity.py` | numpy check of the MeanFlow identity on an analytic Gaussian case (no torch needed) |
| `run_toy.py` | FM vs MeanFlow on toys + ablations → `results_toy/` |
| `train_images.py` | FM vs MeanFlow on MNIST / CIFAR-10 (+ FID) → `results_images/` |
| `MeanFlow_Colab.ipynb` | runnable Colab notebook |
| `report_draft.md` | report draft (convert to IEEE two-column) |
| `VIVA_NOTES.md` | derivation + likely viva questions |

## Setup and run

```bash
pip install -r requirements.txt
python test_identity.py                      # sanity check of the maths (no GPU)
python run_toy.py --quick                    # ~1 min smoke test
python run_toy.py --iters 10000              # main toy results + ablations
python train_images.py --dataset mnist   --iters 20000
python train_images.py --dataset cifar10 --iters 60000   # scale down if Colab T4 time is short
```

Seeds: main toy run uses seed 0; ablations use seeds 100, 101, 102. Fixed in code.
If `torch.func.jvp` raises "forward AD not supported" for some op, use `--dt_mode fd`
and report it under failure cases.

## Results (fill after running)

Toy (SW2 ↓), from `results_toy/results.md`:

| dataset | noise floor | FM 1-step | FM 8-step | FM 128-step | MF 1-step | MF 2-step |
|---|---|---|---|---|---|---|
| moons | TODO | TODO | TODO | TODO | TODO | TODO |
| 8gaussians | TODO | TODO | TODO | TODO | TODO | TODO |
| spiral | TODO | TODO | TODO | TODO | TODO | TODO |

Images (FID ↓), from `results_images/*/results.md`:

| dataset | FM 1-step | FM 50-step | MF 1-step | MF 2-step | paper (MF 1-step) |
|---|---|---|---|---|---|
| MNIST | TODO | TODO | TODO | TODO | n/a |
| CIFAR-10 | TODO | TODO | TODO | TODO | TODO (take from paper) |

## What differs from the paper

- Small U-Net / MLP instead of the paper's architectures; far fewer iterations.
- t/r sampling distribution, fraction of `r ≠ t`, loss-weighting constants are my
  defaults — compare with the paper's appendix and list the differences here.
- FID on 10k samples (not 50k), so numbers are not directly comparable to the paper.
- Sliced-Wasserstein distance is used on the 2-D toys (no FID there).

## References

- Z. Geng, M. Deng, X. Bai, J. Z. Kolter, K. He, "Mean Flows for One-step Generative Modeling", arXiv:2505.13447, 2025.
- If you consult any official / third-party implementation, cite it here (course requirement).
