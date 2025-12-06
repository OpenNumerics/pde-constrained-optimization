### Compute the (T_in, conversion_ratio) curve by doing simple parameter continuation in T_in
import numpy as np
from chemical_pde import solve_reactor, computeSteadyState
from plotSCurve import plotSCurve

y_CO_in = 0.034
u_g = 0.25
T_min = 450.0
T_max = 900.0
dT = 0.005

# Branch 1: T_in < T_crit. First find the solution for a very small T_in (700K)
T_in = T_min
T_in_vals = [T_in]
U = None

N_alpha = 50
alpha_range = (1 + np.arange(N_alpha)) / N_alpha
for alpha in alpha_range:
    print(f'alpha = {alpha}')
    z, Cco_z, Co2_z, T_z, U, _ = solve_reactor(y_CO_in, T_in, u_g, alpha=alpha, U_init=U)
T_max_values = [T_z.max()]
CR_values = [1.0 - Cco_z[-1]/Cco_z[0]]
print('Imitial Tmax: ', T_z.max())

# Slowly increase T by 0.01 for continuation purposes
print('Starting Lower Branch Continuation')
n_steps = int((T_max - T_min) / dT)
for n in range(1, n_steps+1):
    print(f'\nStep {n}: T = {T_in + n * dT}')
    T = T_in + n * dT
    T_in_vals.append(T)

    z, Cco_z, Co2_z, T_z, U_new, converged = solve_reactor(y_CO_in, T, u_g, alpha=1.0, U_init=U)
    if not converged:
        z, Cco_z, Co2_z, T_z, U_new = computeSteadyState(y_CO_in, T, u_g, verbose=False)
    
    U = U_new.copy(deepcopy=True)
    T_max_values.append(T_z.max())
    CR_values.append(np.clip(1.0 - Cco_z[-1]/Cco_z[0], 0.0, 1.0))

# Slowly increase T by 0.01 for continuation purposes
print('Starting Upper Branch Continuation')
n_steps = int((T_max - T_min) / dT)
T_in_upper = T_max
T_in_vals_upper = [T_in_upper]
z, Cco_z, Co2_z, T_z, U = computeSteadyState(y_CO_in, T_in_upper, u_g, verbose=False)

T_max_values_upper = [T_z.max()]
CR_values_upper = [1.0 - Cco_z[-1]/Cco_z[0]]
for n in range(1, n_steps+1):
    print(f'\nStep {n}: T = {T_in_upper - n * dT}')
    T = T_in_upper - n * dT
    T_in_vals_upper.append(T)

    z, Cco_z, Co2_z, T_z, U_new, converged = solve_reactor(y_CO_in, T, u_g, alpha=1.0, U_init=U)

    # Convergence can fail near the transition area, start with an alpha value close to 1.
    if not converged:
        alpha_range = 0.5 + 0.5 * np.arange(0, N_alpha, N_alpha+1) / N_alpha # from 0.1 to 1.
        U_prev = U.copy(deepcopy=True)
        for alpha in alpha_range:
            z, Cco_z, Co2_z, T_z, U_prev, _ = solve_reactor(y_CO_in, T_in, u_g, alpha=alpha, U_init=U_prev)
        U_new = U_prev.copy(deepcopy=True)

    U = U_new.copy(deepcopy=True)
    T_max_values_upper.append(T_z.max())
    CR_values_upper.append(np.clip(1.0 - Cco_z[-1]/Cco_z[0], 0.0, 1.0))

np.save('./data/lower.npy', np.vstack((T_in_vals, CR_values, T_max_values)))
np.save('./data/upper.npy', np.vstack((T_in_vals_upper, CR_values_upper, T_max_values_upper)))

plotSCurve()