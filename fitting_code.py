import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def dPdt(P, t, gamma, k):
    D, A, F1, F2, S = P
    dD  = -gamma * D * A
    dA  = -gamma * D * A - gamma * S * A
    dF1 =  2 * gamma * D * A + gamma * S * A - k * F1
    dF2 =  k * F1 - k * F2
    dS  =  k * F2
    return [dD, dA, dF1, dF2, dS]

def S_of_t(t, gamma, k, D0):
    P0 = [D0, 100.0 - D0, 0.0, 0.0, 0.0]
    sol = odeint(dPdt, P0, t, args=(gamma, k))
    return sol[:, 4]

t_data, y_data = np.loadtxt("sarscov2.csv", delimiter=",", unpack=True)

def ssr(params, t, y):
    gamma, k, D0 = params
    y_hat = S_of_t(t, gamma, k, D0)
    return np.sum((y_hat - y)**2)

gamma0 = 1e-3
k0     = 0.2
D0_0   = 50.0
x0     = np.array([gamma0, k0, D0_0])

res = minimize(ssr, x0, args=(t_data, y_data), method='Nelder-Mead')
gamma_fit, k_fit, D0_fit = res.x

yhat  = S_of_t(t_data, gamma_fit, k_fit, D0_fit)
resid = yhat - y_data

print(f"Converged: {res.success} — {res.message}")
print("Best-fit parameters:")
print(f"  gamma = {gamma_fit:.6f}")
print(f"  k     = {k_fit:.6f}")
print(f"  D0    = {D0_fit:.6f}")
print(f"Minimum SSR = {res.fun:.6f}")

B = 1000
rng = np.random.default_rng(123)
boot_params = np.zeros((B, 3))
theta0 = np.array([gamma_fit, k_fit, D0_fit])

for b in range(B):
    r_perm = rng.permutation(resid)
    y_boot = yhat + r_perm
    res_b = minimize(ssr, theta0, args=(t_data, y_boot), method='Nelder-Mead')
    boot_params[b] = res_b.x

sorted_params = np.sort(boot_params, axis=0)
idx_lo = int(0.025 * B)
idx_hi = int(0.975 * B) - 1  #
ci_low  = sorted_params[idx_lo, :]

ci_high = sorted_params[idx_hi, :]

names = ["gamma", "k", "D0"]
best  = np.array([gamma_fit, k_fit, D0_fit])

print(f"\n=95% Confidence Intervals (B={B}, assignment rule)")
for j, name in enumerate(names):
    print(f"{name}: best = {best[j]:.6g} ; 95% CI ≈ [{ci_low[j]:.6g}, {ci_high[j]:.6g}]")

t_fine = np.linspace(t_data.min(), t_data.max(), 400)
S_fine = S_of_t(t_fine, gamma_fit, k_fit, D0_fit)


np.save("FINAL_boot_sars2.npy", boot_params)  
print("files saved")

plt.rcParams.update({'font.size': 18})

plt.figure()
plt.scatter(t_data, y_data, label='Data', c="red")
plt.plot(t_fine, S_fine, '-', label='Best-fit S(t)')
plt.xlabel("Fusion time (min)")
plt.ylabel("Fusion rate (%)")
plt.title("SARS-CoV-2")
plt.legend()
plt.tight_layout()
plt.show()
