from physicsnemo.sym.hydra import to_absolute_path, instantiate_arch
from physicsnemo.models.mlp.fully_connected import FullyConnected
from physicsnemo.sym.key import Key
from physicsnemo.sym.node import Node
from physicsnemo.sym.eq.pdes.diffusion import Diffusion
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.domain.constraint import PointwiseInteriorConstraint
from physicsnemo.sym.domain.constraint import PointwiseBoundaryConstraint
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.geometry.primitives_2d import Rectangle
import torch


# -----------------------
# Define PDE
# -----------------------
# Darcy simplified as diffusion equation:
# -div(k grad u) = 0  -> approximated as diffusion

darcy = Diffusion(T="u", D=1.0, dim=2)

nodes = darcy.make_nodes()

# -----------------------
# Neural network
# -----------------------
input_keys = [Key("x"), Key("y")]
output_keys = [Key("u")]

net = FullyConnected(
    in_features=3,      # x, y, and a
    out_features=1,     # u (pressure)
    num_layers=6,
    layer_size=512,
    activation_fn="silu" # PhysicsNeMo default
)

nodes += [net.make_node(name="pinn")]

# -----------------------
# Geometry
# -----------------------
geo = Rectangle((0, 0), (1, 1))

# -----------------------
# Domain
# -----------------------
domain = Domain()

# Interior (physics)
interior = PointwiseInteriorConstraint(
    nodes=nodes,
    geometry=geo,
    outvar={"diffusion_u": 0},
    batch_size=1024
)
domain.add_constraint(interior, "interior")

# Boundary condition u = 0
boundary = PointwiseBoundaryConstraint(
    nodes=nodes,
    geometry=geo,
    outvar={"u": 0},
    batch_size=256
)
domain.add_constraint(boundary, "boundary")

# -----------------------
# Solver
# -----------------------
slv = Solver(cfg={"max_steps": 5000}, domain=domain)

slv.solve()
