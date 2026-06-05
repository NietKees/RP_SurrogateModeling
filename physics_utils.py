from physicsnemo.sym.eq.pdes.diffusion import Diffusion
from physicsnemo.sym.eq.phy_informer import PhysicsInformer
from sympy import Symbol, Function
from physicsnemo.sym.eq.pde import PDE
# from physicsnemo.sym.eq.pdes.burgers import Burgers
def get_physics_informer(device, equation, method="finite_difference", res=64):
    """
        PhysicsInformer:
    """

    if(equation == 'darcy'):
        forcing_fn = 1.0 # scale:* 4.49996e00 * 3.88433e-03  # after scaling
        # forcing_fn = 4.49996e00 * 3.88433e-03  # after scaling
        equation = Diffusion(T="u", time=False, dim=2, D="k", Q=forcing_fn) 
        # equation = DarcyPDE(forcing_fn=1.0)
    elif(equation == 'burger'):
        # equation = Burgers(u="u", dim=1, v=0.01);
        raise ValueError('Equation not defined');
    else:
        raise ValueError('Equation not defined');

    eq_name = equation.__class__.__name__.lower()
    out_key = f"{eq_name}_u"
    # out_key = f"diffusion_u"
    dx = 1.0/( res + 1 )
    return PhysicsInformer(
        required_outputs=[out_key],
        equations=equation,
        grad_method=method,
        device=device,
        fd_dx=dx if method == "finite_difference" or method == "function_wise" else None,
        bounds=[1.0, 1.0] if method == "spectral" else None
    )

# class DarcyPDE(PDE):
    def __init__(self, forcing_fn=1.0):
        # super().__init__()
        self.equations = {}
        self.dim = 2
        
        # 1. Define spatial symbols
        x = Symbol("x")
        y = Symbol("y")
        
        # 2. Define pressure (u) and permeability (k) as functions of x and y
        u = Function("u")(x, y)
        k = Function("k")(x, y)
        
        # 3. Define the physical fluxes: k * \nabla u
        flux_x = k * u.diff(x)
        flux_y = k * u.diff(y)
        
        # 4. Divergence form: -\nabla \cdot (k \nabla u)
        darcy_operator = -(flux_x.diff(x) + flux_y.diff(y))
        
        # 5. Set up the residual dictionary (LHS - RHS = 0)
        self.equations = {
            "diffusion_u": darcy_operator - forcing_fn
        }