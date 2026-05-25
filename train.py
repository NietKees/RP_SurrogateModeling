import hydra
from omegaconf import DictConfig
from FNO.darcy_FNO import fno_trainer
from PINN.P2INN_model import train_p2inn
from PINO.PINO import train_pino
from PINN.PiFNO import train_pifno

@hydra.main(version_base="1.3", config_path=".", config_name="pipeline_config")
def main(cfg: DictConfig):
    equation = cfg.equation
    if(equation != 'darcy'):
       raise ValueError(f"Equation '{equation}' not defined in physics_utils.")
      
    # Phase 1: Train FNO to get a baseline
    if cfg.do_fno:
        print("--- Starting FNO Training ---")
        fno_model = fno_trainer(cfg.fno)
    
    # Phase 2: Train PINN (MLP)
    if cfg.do_pinn:
        print("--- Starting PINN Training ---")
        # pinn_model = train_p2inn(cfg.pinn)
        PiFNO = train_pifno(cfg.pino)
    # Phase 3: Train PINO (Using FNO weights as starting point)
    if cfg.do_pino:

        print("--- Starting PINO Training ---")
        # You can pass the fno_model here to "warm start" the PINO!
        pino_model = train_pino(cfg.pino)

if __name__ == "__main__":
    main()