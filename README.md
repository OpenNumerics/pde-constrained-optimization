# PDE‑Constrained Optimization of a Chemical Reactor

We demonstrate how to optimize the yield of a simple
gas‐phase chemical reactor from first principles using the tools of
PDE‑constrained optimization. This repository contains the code behind Hannes Vandecasteele's 
blog posts on PDE-constrained optimization, which is a trimmed down 
version of one of Open Numerics' recent projects for a client.

If you're interested in the theory behind PDE-constrained optimization read (here)[https://www.hvandecasteele.com/blog/pde-constrained-optimization/]. If you want to read about this specific test case, go to the (blog post)[https://www.hvandecasteele.com/blog/pde-constrained-optimization-part2/].

## Background

Converting toxic carbon monoxide into harmless products requires careful
management of transport, reaction, and heat‑transfer processes.  In the model
considered here 

$$
2 \ \text{CO} + O_2 \to 2 CO_2
$$

a mixture of carbon monoxide, oxygen and inert gases enters a
cylindrical reactor at a prescribed inlet temperature and reacts on catalyst
pellets distributed along the tube.  The reactor’s behaviour is governed by
three coupled convection–diffusion–reaction partial differential equations
tracking:
1. Carbon monoxide concentration (mass balance) $C_{\text{CO}}$ convected downstream,
diffuses with coefficient $D$, and is consumed by a catalytic surface
reaction.

$$
\frac{d}{d z}\left( u_g C_{\mathrm{CO}} -  D_{\mathrm{CO}} \frac{d C_{\mathrm{CO}}}{d z} \right)
= - r_{\text{local}}(T, C_{\mathrm{CO}}, C_{\mathrm{O_2}}),
$$

2. Oxygen concentration $C_{O_2}$ behaves similarly to CO but is consumed at half
the rate because two oxygen molecules react with one carbon monoxide
molecule.

$$
\frac{d}{dz} \left( u_g C_{\mathrm{O_2}} - D_{\mathrm{O_2}} \,\frac{d C_{\mathrm{O_2}}}{d z} \right)
= -\frac{1}{2}\, r_{\text{local}}(T, C_{\mathrm{CO}}, C_{\mathrm{O_2}}),
$$

3. Temperature $T(z)$ is also transported with the gas flow, spreads by heat dispersion
and is influenced by the exothermic oxidation reaction and heat loss to the
reactor wall.

$$
\frac{d}{d z} \left(u_g T(z)-  D_T \frac{d T}{d z} \right)
=
- \frac{\Delta H}{\rho C_p} r_{\text{local}}(T, C_{\mathrm{CO}}, C_{\mathrm{O_2}}) - \frac{h_w (P/A)}{\rho C_p}\, \bigl(T - T_{\mathrm{wall}}\bigr).
$$

In short, these equations include many chemical and physical effects. The Arrhenius law describes the local reaction rate’s strong dependence on
temperature, and the Langmuir–Hinshelwood kinetics account for the probability of reactant molecules meeting on the catalyst
surface. Increasing the amount of catalyst boosts conversion, but it also raises temperature, risking thermal runaway.  Heat
escapes through the reactor wall in proportion to the difference between the
gas and wall temperatures and to the reactor’s perimeter‑to‑area ratio. Finally, diffusion smooths out local temperature spikes.

We do not explain these equations further in this README, go to (the blog)[https://hannesvdc.substack.com/p/pde-constrained-optimization-part].

## Temperature spikes and hysteresis

Numerical experiments show that when the inlet temperature is low (≈ 750 K) the
temperature decreases gradually along the tube, yielding a modest conversion
ratio.  Raising the inlet temperature to 800 K causes
a thermal runaway; the reactor temperature exceeds 1100 K because a higher
temperature accelerates the reaction and releases more heat. The figure below shows both scenarios. Notice that much more CO conversion happens in case of a temperature spike!

![Two temperature profiles](images/temperature_profile.png)

When solving the PDEs for a range of inlet temperatures one finds a sharp
bifurcation: above a critical temperature (~ 780 K) the reactor snaps to a
runaway regime, and returning to the safe regime requires cooling well below
that critical temperature.  This hysteresis makes
operating the reactor challenging. The figure below shows the sharp spike in maximum temperature that occures at a specific input.
However, to come down to the lower branch, we need to decrease the inlet temperature much lower than where the spike occured.

![Temperature Hysteresis](images/temperature_hysteresis.png)


## Optimization with a single catalyst loading

To increase yield while avoiding runaway we cast a PDE‑constrained
optimization problem.  The objective is a weighted sum of the conversion
ratio and a penalty on excessive temperature; the optimizer is allowed to
increase the catalyst loading only enough to improve conversion without
overheating .  

$$
\underset{a}{\min} \ J(a) = \frac{C_{\text{CO}}(\text{outlet})}{C_{\text{CO}}(\text{inlet})} + \gamma \int_{\text{inlet}}^{\text{outlet}} \max\left(T(z) - T_{\text{in}}, 0\right)
$$

The inlet temperature is fixed, so the optimization variable is the (uniform) amount of catalyst pellets.  We implement a simple gradient-based optimizer. For an initial candidate catalyst loading we solve the PDEs, compute the objective and its gradient via the adjoint
method, and update the loading accordingly.

Large penalty weights force the temperature to remain close to the inlet temperature at the cost of low conversion, whereas a moderate penalty allows
temperatures to rise slightly and yields a significantly higher conversion at the risk of being closer to to thermal runaway. The figure below shows the optimized catalyst concentrations for several values of $\gamma$ and where the optimum is located on the hystersis curve (of $a$). Note that if $\gamma$ is too small, we end up with thermal runaway anyways.

![Optimal Solutions per $\gamma$.](images/acurve.png)


## Multiple catalyst zones

The model can be extended by distributing the catalyst unevenly.  Here, we divide the reactor into five equal zones and assign a separate catalyst density to each zone. The optimizer tends to place most catalyst at the inlet and much less downstream; this achieves essentially the same conversion while reducing the total amount of catalyst used . When the catalyst is expensive, this is actually a big win! The resulting catalyst profile reflects the fact that the temperature is highest near the inlet and decreases downstream, so additional pellets further along contribute little to the reaction.

![Multiple catalyst zones.](images/multi_zones.png)


## Repository contents

This repository contains Python scripts that replicate the model and optimization procedures described above. The code heavily relies on FEniCS to discretize and solve the PDEs, and dolfin‑adjoint￼for the adjoint method and optimization.

```
├── parameters.py			   				# Common definition of fixed PDE parameters
├── PDE_setup.py			   				# PDE defintion and setup in FEniCS with many helper functions
├── chemical_pde.py			   				# Contains the main solver routine for calculating the steady state of the PDE at given $(T_{\text{in}}, a)$.
├── SCurve.py				   				# Used to calculate the hysteresis curve as a function of $T_{\text{in}}$.
├── plotSCurve.py			   				# Plot the temperature hysteresis curve
├── bifurcation_diagram_a.py   				# Calculate the hysteresis curve as a function of the catalyst concentration $a$.
├── plotBifurcationDiagramA.py 				# Plot the hysteresis curve of $a$.
├── PDE_constrained_optimization.py 		# Main optimization routine for the single catalyst region setup.
├── Functional_PDE_constrained_optimization # Optimization routine for the multi-zone setup
├── data/									# All data files that were generated with the python scripts above.
├── tests/									# My own simple testing routines
├── images/									# For the website and all blog posts.
├── LICENSE
└── README.md
```

## Installation
1.	Clone this repository:
```
git clone git@github.com:hannesvdc/PDEOPT_chemical.git
cd PDEOPT_chemical
```

2.	Install dependencien, ideally in a separate Python virtual environment.
The key dependencies are fenics, dolfin-adjoint, numpy, scipy and
matplotlib.  FEniCS installation can be done via pip on many
platforms:
```
python3 -m venv venv
source venv/bin/activate
pip install fenics dolfin‑adjoint numpy scipy matplotlib
```

3.	Run any of the scripts, for example
```
python3 SCurve.py
```
The scripts will output the hysteresis curve as a function of the inlet temperature.

## Contributing

The code in this project is one big mess, so I don't expect anyone to contribute.
If you do find a bug or have suggestions for extending the project (e.g. different reactor geometries, additional control
variables, or other PDE‑constrained optimization problems), please open an
issue or submit a pull request. I am always available for a quick call too.

## License

This project is licensed under the Apache 2.0 License - see LICENSE.