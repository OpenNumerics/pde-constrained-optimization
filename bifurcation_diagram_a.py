from dolfin import *
from dolfin_adjoint import *
import numpy as np

from parameters import *
from PDE_setup import inlet_concentrations, C_CO_inlet, C_O2_inlet, T_inlet, W, bcs, ds, OUTLET_ID
from plotBifurcationDiagramA import plotBifurcationDiagramA

set_log_level(LogLevel.ERROR)

# Temperature is fixed now.
y_CO_in = 0.034
u_g = 0.25
T_in = 740.0
Cco_in, Co2_in = inlet_concentrations(y_CO_in, T_in)

a_const = Constant(3e4)

N_alpha = 50
alpha_range = np.arange(N_alpha) / N_alpha

# ----------------------------
# High-level solve function in terms of a
# ----------------------------
def solve_reactor_with_profile(a_val, alpha=1.0, U_init=None):
    """
    Solve the steady-state reactor PDE for given inlet CO mole fraction,
    inlet temperature, superficial velocity u_g, and a piecewise-constant
    a_s(z) profile defined by a_vals (len = N_ZONES).
    Returns (z_coords, C_CO(z), C_O2(z), T(z), U).
    """

    # Build a_s(z) field from a_vals
    C_CO_inlet.assign(Cco_in)
    C_O2_inlet.assign(Co2_in)
    T_inlet.assign(T_in)
    a_const.assign(a_val)

    # Fresh unknown & test functions
    U = Function(W)
    Cco, Co2, T = split(U)
    v1, v2, v3 = TestFunctions(W)

    # Reaction rate (same regularization as before)
    Co2_eff = conditional(ge(Co2, 1e-12), Co2, 1e-12)
    T_eff   = conditional(ge(T, 200.0),   T,   200.0)
    r_local = k0 * exp(-E_a / (R_gas * T_eff)) \
        * (K_CO * Cco * K_O2 * sqrt(Co2_eff)) \
        / (1.0 + K_CO * Cco + K_O2 * sqrt(Co2_eff))**2

    # Residual with a_s_field instead of scalar a_s
    F_co = ( D_ax * dot(grad(Cco), grad(v1))
             + u_g * Cco.dx(0) * v1
             + alpha * a_const * r_local * v1 ) * dx

    F_o2 = ( D_ax * dot(grad(Co2), grad(v2))
             + u_g * Co2.dx(0) * v2
             + 0.5 * alpha * a_const * r_local * v2 ) * dx

    F_T = ( lambda_eff * dot(grad(T), grad(v3))
            + rho_g * cp_g * u_g * T.dx(0) * v3
            - q * alpha * a_const * r_local * v3
            + h_w * P_over_A * (T - T_wall) * v3 ) * dx

    F = F_co + F_o2 + F_T

    # Initial guess
    if U_init is None:
        U.interpolate(Constant((Cco_in, Co2_in, T_in)))
    else:
        U.vector()[:] = U_init.vector()

    # Jacobian for Newton
    dU = TrialFunction(W)
    J = derivative(F, U, dU)

    problem = NonlinearVariationalProblem(F, U, bcs, J)
    solver = NonlinearVariationalSolver(problem)
    prm = solver.parameters
    prm["newton_solver"]["maximum_iterations"] = 25
    prm["newton_solver"]["absolute_tolerance"] = 1e-12
    prm["newton_solver"]["relative_tolerance"] = 1e-8
    prm["newton_solver"]["error_on_nonconvergence"] = True

    try:
        solver.solve()
        converged = True
    except:
        converged = False

    return U, converged

def forward_objective_and_gradient(a_vec, U_init):
    tape = get_working_tape()
    tape.clear_tape()

    # Aolve at alpha=1.0 WITH annotation
    print('Solving objective with a_vec =', a_vec, ' ...')
    U, _ = solve_reactor_with_profile(a_val=a_vec, alpha=1.0, U_init=U_init)

    # 3. Build J(U, a) as before
    Cco_fun, Co2_fun, T_fun = U.split()
    T_max = np.max(T_fun.vector().get_local())

    # Outlet CO "average" → in 1D this acts like evaluation at z=L
    C_in_val, _ = inlet_concentrations(y_CO_in, T_in)
    J_conv_form = (1.0 / C_in_val) * Cco_fun * ds(OUTLET_ID)
    with stop_annotating():
        CR = 1.0 - assemble(J_conv_form)

    beta = Constant(1e-6)
    grad_penalty_form = beta * inner(grad(T_fun), grad(T_fun)) * dx
    with stop_annotating():
        penalty = assemble(grad_penalty_form)

    # Main penalty: excess temperature
    excess = T_fun - Constant(T_in)
    smooth_relu = 0.5 * (excess + sqrt(excess**2 + 1.e-4))  # ~ max(dT, 0)
    excess_penalty_form = smooth_relu**2 * dx
    with stop_annotating():
        excess_penalty = assemble(excess_penalty_form)

    J_form = 100.0 * J_conv_form + grad_penalty_form
    J = assemble(J_form)

    # 4. Reduced functional + gradient wrt a_const
    m = Control(a_const)
    J_reduced = ReducedFunctional(J, m)
    grad_list = J_reduced.derivative()
    grad_J = np.array(grad_list, dtype=float)

    return float(J), grad_J, T_max, CR, penalty, excess_penalty, U

def computeInitial(a):
    U_init = None
    with stop_annotating():
        for alpha in alpha_range:
            U_init, _ = solve_reactor_with_profile(a_val=a, alpha=alpha,U_init=U_init)
    return U_init

# Track the lower branch by solving for each a.
a_min = 20_000.0
a_max = 50_000.0
da = 10.0

a_values = []
T_max_values = []
CR_values = []
grad_penalties = []
excess_penalties = []
J_values = []
dJ_values = []

# Slowly increase a by 50 for continuation purposes
print('Starting Lower Branch Continuation')
n_steps = int((a_max - a_min) / da)
U = None
for n in range(0, n_steps+1):
    a = a_min + n * da
    print(f'\nStep {n}: a = {a}')
    a_values.append(a)

    if U is not None:
        with stop_annotating():
            U_init, converged = solve_reactor_with_profile(a, alpha=0.98, U_init=U.copy(deepcopy=True))
    if U is None or not converged:
        U_init = computeInitial(a)

    J_val, grad_J_val, T_max, CR, grad_penalty, excess_penalty, U = forward_objective_and_gradient(a, U_init)
    print(f'\t\t CR = {CR}: T_max = {T_max}')

    T_max_values.append(T_max)
    CR_values.append(100.0 * CR)
    grad_penalties.append(grad_penalty)    
    excess_penalties.append(excess_penalty)
    J_values.append(J_val)
    dJ_values.append(grad_J_val)

# Store these arrays for optimal objective design.
np.save('./data/bifurcation_diagram_A.npy', np.vstack((a_values, T_max_values, CR_values, grad_penalties, excess_penalties, J_values, dJ_values)))
plotBifurcationDiagramA()