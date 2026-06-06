import numpy as np
from scipy.integrate import solve_ivp
import torch
from torch.utils.data import Dataset, DataLoader

def burgers_generator(
    nx=256,
    nt=100,
    nu=0.05,
    tmax=1.0,
    complexity=5,
    amp_scale=1.0,
    freq_decay=1.0,
    seed=None,
):
    """
    Fully controllable 1D Burgers' equation solver.
    """

    rng = np.random.default_rng(seed)

    # -----------------------------
    # Spatial grid
    # -----------------------------
    x = np.linspace(0, 2*np.pi, nx, endpoint=False)
    dx = x[1] - x[0]

    # -----------------------------
    # Initial condition
    # -----------------------------
    u0 = np.zeros(nx)

    for k in range(1, complexity + 1):
        amp = amp_scale * rng.uniform(-1, 1) / (k ** freq_decay)
        phase = rng.uniform(0, 2*np.pi)
        u0 += amp * np.sin(k * x + phase)

    # -----------------------------
    # PDE RHS (periodic BCs via np.roll)
    # -----------------------------
    # def rhs(t, u):
    #     ux = (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)
    #     uxx = (np.roll(u, -1) - 2*u + np.roll(u, 1)) / (dx**2)
    #     return -u * ux + nu * uxx
    
    k_modes = np.fft.fftfreq(nx, d=1.0/nx) * 1j

    def rhs(t, u):
        # 1. Transform u to Fourier space
        u_hat = np.fft.fft(u)
        
        # 2. Compute exact derivatives in Fourier space
        ux_hat = k_modes * u_hat
        uxx_hat = (k_modes**2) * u_hat
        
        # 3. Transform back to physical space
        ux = np.real(np.fft.ifft(ux_hat))
        uxx = np.real(np.fft.ifft(uxx_hat))
        
        # 4. Return physical RHS
        return -u * ux + nu * uxx

    # -----------------------------
    # Time grid
    # -----------------------------
    t_eval = np.linspace(0, tmax, nt)

    sol = solve_ivp(
        rhs,
        (0, tmax),
        u0,
        t_eval=t_eval,
        method="BDF",
        rtol=1e-6,
        atol=1e-6,
        max_step=1e-2,
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    u = sol.y.T  # (nt, nx)
    if np.max(np.abs(u)) > amp_scale:
        raise RuntimeError(f'out of bounds solution_{nu}')
    return x, t_eval, u



# ========================================================
# 1. ADD A COMPATIBLE PYTORCH DATASET WRAPPER
# ========================================================
class BurgersTorchDataset(Dataset):
    def __init__(self, raw_data_list):
        """Wraps the output of get_burgers_batch to ensure correct tensor shapes 
        and dimensions for the FNO model pipeline.
        """
        self.data = raw_data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # sample["u0"] is shape (Nx,) -> Convert to (1, Nx) to add the Channel dimension
        v_init = torch.tensor(sample["u0"], dtype=torch.float32).unsqueeze(0)
        
        # sample["solution"] is shape (Nt, Nx) -> Transpose to (Nx, Nt), then unsqueeze 
        # to add Channel dimension -> Shape: (1, Nx, Nt)
        v = torch.tensor(sample["solution"], dtype=torch.float32).T.unsqueeze(0)
        
        return {
            "v_init": v_init, 
            "v": v,
            "nu": torch.tensor(sample["nu"], dtype=torch.float32)
        }

def get_burgers_batch(        
        num_samples,
        nx=256,
        nt=100,
        nu=0.05,
        tmax=1.0,
        complexity=5,
        amp_scale=1.0,
        freq_decay=1.5,
        seed=None,
        ):
    dataset = []
    errors = {}
    for i in range(num_samples):
        # nu = 10 ** np.random.uniform(-3, -1)
        success = False
        while success == False:
            try:
                x, t, u = burgers_generator(
                    nx=nx,
                    nu=nu,
                    nt=nt,
                    tmax=tmax,
                    complexity=complexity,
                    amp_scale=amp_scale,
                    freq_decay=freq_decay,
                    # seed=i,
                )
                success = True
            except Exception as e:
                # print(e)
                errors[str(e)] = errors.get(str(e), 0) + 1
                continue
            dataset.append(
                {
                    "u0": u[0],
                    "solution": u,
                    "nu": nu,
                }
            )
    if errors:
        print("\n--- Generation Errors Summary ---")
        total_amount_of_errors = 0
        for error_type, count in errors.items():
            total_amount_of_errors += count
            print(f"Count: {count} | Error: {error_type}")
        print(f"\n--- {total_amount_of_errors}/{num_samples + total_amount_of_errors} failed ({(total_amount_of_errors/(num_samples + total_amount_of_errors))*100}%)---")

        
    dataset = BurgersTorchDataset(dataset)
    return dataset



# =========================================
#   Residual loss
# =========================================
"""
Used like
    def physics_loss(u_pred, nu, dx, dt):
        res = burgers_physics_residual(u_pred, nu, dx, dt)
        return torch.mean(res ** 2)
"""
def dx(u, dx):
    return (torch.roll(u, -1, dims=-1) - torch.roll(u, 1, dims=-1)) / (2 * dx)


def dxx(u, dx):
    return (
        torch.roll(u, -1, dims=-1)
        - 2 * u
        + torch.roll(u, 1, dims=-1)
    ) / (dx ** 2)


def burgers_physics_residual(u_pred, nu, dx_val, dt_val):
    """
    Correctly evaluates the 1D+Time Burgers equation residual:
    Residual = u_t + u * u_x - nu * u_xx
    
    Assumes u_pred shape is: [B, channels, nx, nt]
    """
    u = u_pred
    dt = dt_val
    dx = dx_val
    
    # Define explicitly which dimensions correspond to space and time
    spatial_dim = 2   # nx axis
    temporal_dim = 3  # nt axis (or -1)

    # ========================================================
    # 1. SPATIAL DERIVATIVES (Periodic Central Differences on Axis 2)
    # ========================================================
    u_x = (torch.roll(u, shifts=-1, dims=spatial_dim) - 
           torch.roll(u, shifts=1, dims=spatial_dim)) / (2.0 * dx)

    u_xx = (torch.roll(u, shifts=-1, dims=spatial_dim) - 
            2.0 * u + 
            torch.roll(u, shifts=1, dims=spatial_dim)) / (dx ** 2)

    # ========================================================
    # 2. TEMPORAL DERIVATIVES (Non-Periodic Central & Single-Sided)
    # ========================================================
    u_t = torch.zeros_like(u)

    # Interior points along the time axis (t=1 to t=nt-2)
    u_t[..., 1:-1] = (u[..., 2:] - u[..., :-2]) / (2.0 * dt)

    # Boundary conditions for the temporal domain
    u_t[..., 0] = (u[..., 1] - u[..., 0]) / dt       # Forward finite difference at t=0
    u_t[..., -1] = (u[..., -1] - u[..., -2]) / dt    # Backward finite difference at t=max

    # ========================================================
    # 3. BURGERS' RELATION SHIP EVALUATION
    # ========================================================
    residual = u_t + (u * u_x) - (nu * u_xx)
    return residual