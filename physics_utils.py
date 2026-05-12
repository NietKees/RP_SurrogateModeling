from physicsnemo.sym.eq.pdes.diffusion import Diffusion
from physicsnemo.sym.eq.phy_informer import PhysicsInformer

def get_physics_informer(device, equation, method="finite_difference", res=64):
    """
        PhysicsInformer:
    """

    if(equation == 'darcy'):
        forcing_fn = 1.0 * 4.49996e00 * 3.88433e-03  # after scaling
        equation = Diffusion(T="u", time=False, dim=2, D="k", Q=forcing_fn)
    elif(equation == 'burger'):
        raise ValueError('Equation not defined');
    else:
        raise ValueError('Equation not defined');

    eq_name = equation.__class__.__name__.lower()
    out_key = f"{eq_name}_u"

    return PhysicsInformer(
        required_outputs=[out_key],
        equations=equation,
        grad_method=method,
        device=device,
        fd_dx=1.0/res if method == "finite_difference" else None
    )
