from dolfin import *
from dolfin_adjoint import *

mesh = UnitIntervalMesh(50)
V = FunctionSpace(mesh, "CG", 1)

u = Function(V)
v = TestFunction(V)

m = Constant(1.0)
control = Control(m)

F = inner(grad(u), grad(v))*dx + m*u*v*dx - Constant(1.0)*v*dx
solve(F == 0, u, [])

J_form = inner(u, u)*dx
J = assemble(J_form)

Jhat = ReducedFunctional(J, control)
dJdm = Jhat.derivative()

print("dJ/dm =", dJdm)