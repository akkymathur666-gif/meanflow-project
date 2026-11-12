# Viva notes — MeanFlow

## Board derivation (be able to do this cold)

Setup: `z_t = (1−t)x + tε`, `v = ε − x`, flow ODE `dz/dt = v(z_t, t)` (marginal velocity).

1. **Average velocity** over `[r, t]`:
   `u(z_t, r, t) := (1/(t−r)) ∫_r^t v(z_τ, τ) dτ  =  (z_t − z_r)/(t − r)`.
2. **Multiply out:** `(t−r) u(z_t, r, t) = ∫_r^t v(z_τ, τ) dτ`.
3. **Differentiate w.r.t. t** (r fixed, z_t moves along the flow, so `dz_t/dt = v(z_t, t)`):
   `u + (t−r) d/dt u = v(z_t, t)`.
4. **MeanFlow identity:** `u(z_t, r, t) = v(z_t, t) − (t−r) · d/dt u(z_t, r, t)`.
5. **Total derivative = JVP:** `d/dt u = v · ∂_z u + ∂_t u` = JVP of `u_θ` at `(z, r, t)` with tangent `(v, 0, 1)`.
6. **Training:** replace the marginal `v` by the conditional `v = ε − x` (unbiased, same argument as flow matching), use stop-gradient on the target:
   `L = E ‖ u_θ(z_t, r, t) − sg( v − (t−r)(v·∂_z u_θ + ∂_t u_θ) ) ‖²`.
7. **Sampling:** `z_r = z_t − (t−r) u_θ(z_t, r, t)`; one step: `x = ε − u_θ(ε, 0, 1)`.
8. **Consistency check:** `r → t` gives `u → v` (flow matching). That is why samples with `r = t` are just FM.

## Likely questions

- *Why does one step work now but not for flow matching?* FM learns the instantaneous velocity; a single Euler step with it follows the tangent, not the curved trajectory. MeanFlow learns the interval-averaged velocity, which is exactly the straight-line jump `z_t → z_r`.
- *Why stop-gradient on the target?* Otherwise gradients flow through the JVP (second-order terms) and the target chases the prediction; it is also costlier. (Ablation candidate if you have time.)
- *What does the JVP cost?* About one extra forward-mode pass, roughly 2–3× a plain forward/backward step.
- *Why condition on (t, t−r)?* Design choice: it makes the `r = t` (h = 0) case explicit. Check the paper for what it uses.
- *What if r = t for all samples?* Loss reduces to flow matching; 1-step quality collapses (my ablation `p_neq=0.0`).
- *Is the loss tractable without knowing the true v?* Yes: conditional velocity `ε − x`, and `u_θ` itself supplies the `d/dt u` term.
- *Finite difference vs JVP?* FD is approximate and needs extra forward passes under no_grad; JVP is exact (ablation).
- *Failure modes you saw?* Fill from your runs (instability, forward-AD unsupported ops, FID noise at small N).
