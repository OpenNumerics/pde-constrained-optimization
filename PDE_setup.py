from dolfin import IntervalMesh, FiniteElement, MixedElement, FunctionSpace, Constant, SubDomain, near, DirichletBC, MeshFunction, Measure
from dolfin_adjoint import * # Make all bcs adjoint-aware

from parameters import L, p_tot, R_gas

# Create 1D mesh and function spaces
n_cells = 2000
mesh = IntervalMesh(n_cells, 0.0, L)

P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
mixed_element = MixedElement([P1, P1, P1])  # C_CO, C_O2, T
W = FunctionSpace(mesh, mixed_element)

# Helper: inlet concentrations
def inlet_concentrations(y_CO_in, T_in):
    """
    Given inlet CO mole fraction y_CO_in and T_in,
    compute C_CO,in and C_O2,in [mol/m^3].
    Assume air + CO: y_O2 ~ 0.21, y_N2 ~ 0.79 - y_CO_in.
    """
    y_O2_in = 0.21  # simple assumption: air-like O2
    C_total_in = p_tot / (R_gas * T_in)
    C_CO_in = y_CO_in * C_total_in
    C_O2_in = y_O2_in * C_total_in
    #print('Inlet in: ', y_CO_in, T_in, 'Out: ', C_CO_in, C_O2_in)
    return C_CO_in, C_O2_in

# Boundary: inlet concentrations
class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0)
inlet = Inlet()
class Outlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], L)
outlet = Outlet()

boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
inlet.mark(boundary_markers, 1)
outlet.mark(boundary_markers, 2)

ds = Measure("ds", domain=mesh, subdomain_data=boundary_markers)
OUTLET_ID = 2

# We'll set actual inlet values later via parameters
C_CO_inlet = Constant(0.0)
C_O2_inlet = Constant(0.0)
T_inlet = Constant(500.0)

bc_Cco = DirichletBC(W.sub(0), C_CO_inlet, inlet)
bc_Co2 = DirichletBC(W.sub(1), C_O2_inlet, inlet)
bc_T   = DirichletBC(W.sub(2), T_inlet, inlet)
bcs = [bc_Cco, bc_Co2, bc_T]