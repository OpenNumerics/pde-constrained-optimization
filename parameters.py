import math

# ----------------------------
# Physical & reactor parameters
# ----------------------------
L = 0.5               # reactor length [m]
p_tot = 2.0e5         # total pressure [Pa] (2 bar)
R_gas = 8.314         # J/(mol*K)

# Gas properties (approximate)
rho_g = 0.5           # kg/m^3
cp_g = 1100.0         # J/(kg*K)

# Axial dispersion & thermal conductivity
D_ax = 3.0e-5         # m^2/s
lambda_eff = 0.4      # W/(m*K)

# Geometry: 2 cm ID tube
D_tube = 0.02         # m
P = math.pi * D_tube
A = math.pi * (D_tube**2) / 4.0
P_over_A = P / A

# Wall heat transfer coefficient
h_w = 40.0            # W/(m^2*K)
T_wall = 500.0        # K (could set = T_in for adiabatic-like)

# Catalyst loading
a_s = 3e4             # m^2 catalyst surface / m^3 reactor

# Reaction parameters (tunable)
k0 = 1e11             # mol/(m^2*s)
E_a = 8.0e4           # J/mol  (80 kJ/mol)
K_CO = 2.0e-5         # m^3/mol
K_O2 = 1.5e-5         # m^(3/2)/mol^(1/2)
q = 2.83e5            # J/mol (heat released per mol CO reacted)