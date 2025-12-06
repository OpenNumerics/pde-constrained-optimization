from dolfin import *
from dolfin_adjoint import *
import matplotlib.pyplot as plt

# ============ Forward (primal) problem setup ============

# Mesh and function space
mesh = UnitSquareMesh(32, 32)
V = FunctionSpace(mesh, "Lagrange", 1)

# Control (design) variable: m(x)
m = Function(V, name="Control")
m.assign(Constant(0.0))  # initial guess

# State variable: u(x)
u = TrialFunction(V)
v = TestFunction(V)
u_sol = Function(V, name="State")

# PDE: -Δu = m in Ω, u = 0 on ∂Ω
a = inner(grad(u), grad(v)) * dx
L = m * v * dx

bc = DirichletBC(V, Constant(0.0), "on_boundary")

# IMPORTANT: tell dolfin-adjoint we're starting a "tape"
# (this happens automatically on first annotated solve, but explicit is fine)
set_log_level(LogLevel.INFO)  # keep output quiet

# Solve forward problem
solve(a == L, u_sol, bc)

# ============ Objective functional J(u, m) ============

alpha = Constant(1e-4)

# Desired state u_d(x)
u_d = interpolate(Expression("sin(pi*x[0])*sin(pi*x[1])",
                             degree=2), V)

# Misfit + Tikhonov regularization
J_form = 0.5 * inner(u_sol - u_d, u_sol - u_d) * dx + 0.5 * alpha * inner(m, m) * dx

J = assemble(J_form)

# ============ Set up control and reduced functional ============

m_control = Control(m)
Jhat = ReducedFunctional(J, m_control)
m_opt = minimize(
    Jhat,
    method="L-BFGS-B",
    options={"maxiter": 50, "gtol": 1e-12}
)

print("Optimized J(m) =", Jhat(m_opt))

# After m_opt = minimize(...)
plt.figure()
p = plot(m_opt)  # u_sol now corresponds to m_opt
plt.colorbar(p)
plt.title("Optimized state u_opt(x)")
plt.show()