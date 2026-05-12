import hydra
from omegaconf import DictConfig
from train_fno import train_fno
from train_pinn import train_pinn
from train_pino import train_pino

@hydra.main(version_base="1.3", config_path=".", config_name="pipeline_config")
def main(cfg: DictConfig):
    equation = cfg.equation
    norm = cfg.normaliser
    if(equation != 'darcy'):
       raise ValueError(f"Equation '{equation}' not defined in physics_utils.")
      
    # Phase 1: Train FNO to get a baseline
    if cfg.do_fno:
        print("--- Starting FNO Training ---")
        fno_model = train_fno(cfg.fno, norm)
    
    # Phase 2: Train PINN (MLP)
    if cfg.do_pinn:
        print("--- Starting PINN Training ---")
        pinn_model = train_pinn(cfg.pinn, norm)

    # Phase 3: Train PINO (Using FNO weights as starting point)
    if cfg.do_pino:

        print("--- Starting PINO Training ---")
        # You can pass the fno_model here to "warm start" the PINO!
        pino_model = train_pino(cfg.pino, norm)

if __name__ == "__main__":
    main()