import hydra
from omegaconf import DictConfig
from FNO.darcy_FNO import fno_trainer
from PINN.ParametricPINN import train_p2inn
from PINO.PINO import train_pino
from PINN.PiFNO import train_pifno

@hydra.main(version_base="1.3", config_path=".", config_name="pipeline_config")
def main(cfg: DictConfig):
    equation = cfg.equation
    if(equation != 'darcy'):
       raise ValueError(f"Equation '{equation}' not defined in physics_utils.")
    
    if cfg.do_pino:
        print("--- Starting PINO Training ---")
        pino_model = train_pino(cfg.pino)  
    
    if cfg.do_fno:
        print("--- Starting FNO Training ---")
        fno_model = fno_trainer(cfg.fno)
    
    if cfg.do_pinn:
        print("--- Starting PINN Training ---")
        # pinn_model = train_p2inn(cfg.pinn)
        PiFNO = train_pifno(cfg.pifno)
   

if __name__ == "__main__":
    main()