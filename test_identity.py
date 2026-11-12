"""
Numerical sanity check of the MeanFlow identity (numpy only, no PyTorch needed).

Setting where everything is analytic: 1-D data x ~ N(0, s^2), noise eps ~ N(0, 1),
z_t = (1 - t) x + t eps.  Then z_t ~ N(0, sigma(t)^2) with
sigma(t)^2 = (1 - t)^2 s^2 + t^2, the marginal velocity is linear,

    v(z, t) = sigma'(t)/sigma(t) * z,

the flow is z_tau = z_t * sigma(tau)/sigma(t), and the exact average velocity is

    u(z, r, t) = (z_t - z_r)/(t - r) = z * (1 - sigma(r)/sigma(t)) / (t - r).

We check
  (1) u(z, t - h, t) -> v(z, t) as h -> 0,
  (2) identity:  u = v - (t - r) * d/dt u(z_t, r, t)   (total derivative along the flow),
  (3) the JVP form of the total derivative:  d/dt u = v * d_z u + d_t u.
"""
import numpy as np

s = 0.5


def sig(t):
    return np.sqrt((1 - t) ** 2 * s**2 + t**2)


def v(z, t):
    cov = t - (1 - t) * s**2  # = sigma * sigma'
    return cov / sig(t) ** 2 * z


def u(z, r, t):
    return z * (1 - sig(r) / sig(t)) / (t - r)


rng = np.random.default_rng(0)
N = 1000
t = rng.uniform(0.2, 0.95, N)
r = t * rng.uniform(0.0, 0.9, N)
z = rng.normal(size=N) * sig(t)
h = 1e-5

# (1) limit r -> t
e1 = np.max(np.abs(u(z, t - 1e-7, t) - v(z, t)))

# (2) total derivative along the true trajectory z(t') = z * sigma(t') / sigma(t)
zp = z * sig(t + h) / sig(t)
zm = z * sig(t - h) / sig(t)
dudt_total = (u(zp, r, t + h) - u(zm, r, t - h)) / (2 * h)
e2 = np.max(np.abs(u(z, r, t) - (v(z, t) - (t - r) * dudt_total)))

# (3) JVP form: v * d_z u + d_t u  (partials by finite differences)
dz = (u(z + h, r, t) - u(z - h, r, t)) / (2 * h)
dt = (u(z, r, t + h) - u(z, r, t - h)) / (2 * h)
jvp_form = v(z, t) * dz + dt
e3 = np.max(np.abs(jvp_form - dudt_total))
e4 = np.max(np.abs(u(z, r, t) - (v(z, t) - (t - r) * jvp_form)))

print(f"(1) |u(z,t-h,t) - v(z,t)|                    max err = {e1:.2e}")
print(f"(2) |u - (v - (t-r) d/dt u)|  (total deriv)  max err = {e2:.2e}")
print(f"(3) |v*d_z u + d_t u - total derivative|     max err = {e3:.2e}")
print(f"(4) |u - (v - (t-r) * JVP form)|             max err = {e4:.2e}")
assert max(e1, e2, e3, e4) < 1e-4, "MeanFlow identity check FAILED"
print("OK: MeanFlow identity and its JVP form hold numerically.")
