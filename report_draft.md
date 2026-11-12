# Reproducing Mean Flows for One-step Generative Modeling at Small Scale
*(Report draft — convert to IEEE two-column, 4–6 pages. Items marked **TODO** need your experimental results.)*

Name, roll no. (24CS3033 / DL-086), Deep Learning 2026

## Abstract
We re-implement MeanFlow (Geng et al., 2025) from scratch in PyTorch, verify its identity numerically, and compare one-step MeanFlow with multi-step flow matching on 2-D toy data, MNIST and CIFAR-10 under a free-Colab-T4 budget. **TODO:** one sentence with the headline numbers and the main ablation finding.

## 1. Problem and Motivation
Flow-matching and diffusion models generate samples by integrating an ODE over many steps. Few-step generation is hard because a single Euler step uses the *instantaneous* velocity, which is the tangent of a curved trajectory. MeanFlow instead models the *average* velocity over an interval, so a single network evaluation can jump from noise to data.

## 2. Method
Notation: `z_t = (1−t)x + tε`, `v = ε − x`, `t=1` noise, `t=0` data.

**Average velocity.** `u(z_t, r, t) = (1/(t−r)) ∫_r^t v(z_τ, τ) dτ`.

**MeanFlow identity.** Differentiating `(t−r)u = ∫_r^t v dτ` in `t` along the flow gives
`u(z_t, r, t) = v(z_t, t) − (t−r) d/dt u(z_t, r, t)`, with `d/dt u = v·∂_z u + ∂_t u`, a single Jacobian–vector product with tangent `(v, 0, 1)`.

**Loss.** `L(θ) = E ‖u_θ(z_t, r, t) − sg(v − (t−r)(v·∂_z u_θ + ∂_t u_θ))‖²`, where `sg` is stop-gradient; a fraction of samples uses `r = t` (reducing to flow matching). Optional adaptive weighting `w = 1/(‖Δ‖² + c)^p`.

**Sampling.** One step: `x = ε − u_θ(ε, 0, 1)`; `n` steps: `z_r = z_t − (t−r)u_θ(z_t, r, t)` on a uniform grid.

## 3. Implementation
- `meanflow.py`: residual MLP (toys) and small U-Net (images) taking `(z, r, t)`, conditioned on `t` and `t−r` through sinusoidal embeddings; `torch.func.jvp` for the total derivative; EMA of weights; flow-matching baseline uses the same network with `r = t`.
- Only forward-AD-compatible layers are used (conv, conv-transpose, GroupNorm, SiLU, linear).
- Hyper-parameters (batch size, LR, iterations, t/r distribution, fraction `r≠t`, adaptive weighting): **TODO** — copy from the commands you actually ran, and list differences from the paper.
- Validation of the maths: `test_identity.py` checks the identity and its JVP form on an analytic Gaussian case; max error ≈ 1e-10 (total derivative form) and ≈ 4e-10 (JVP form) in float64.
- Compute: Colab T4, fp32. **TODO:** wall-clock per run.

## 4. Results versus the paper
**TODO** — insert `results_toy/results.md` and `results_images/*/results.md`, and the sample figures `samples_*.png`, `mf_1step.png`.
State what changed from the paper (model size, iterations, FID on 10k samples instead of 50k, etc.), and put the paper's reported CIFAR-10 one-step number next to yours. Do not claim parity unless your numbers show it.

## 5. Ablations
Run in `run_toy.py` on 8 Gaussians (mean ± std over 3 seeds): fraction of `r≠t` (0, 0.25, 0.5, 1.0), JVP vs finite-difference derivative, with vs without adaptive loss weighting. **TODO:** table + 3–4 sentences of interpretation. Expected to check: `p_neq=0` should give poor 1-step samples because the model never sees `r≠t`.

## 6. Failure Cases / What Did Not Work
**TODO** — fill honestly from your runs, e.g. ops without forward-mode AD, instabilities, sensitivity to the `(r, t)` sampling distribution, FID variance at small sample size.

## 7. Conclusion
**TODO** (2–3 sentences).

## References
[1] Z. Geng, M. Deng, X. Bai, J. Z. Kolter, K. He, "Mean Flows for One-step Generative Modeling," arXiv:2505.13447, 2025.
[2] Add flow-matching / rectified-flow references and any repository you consulted.
