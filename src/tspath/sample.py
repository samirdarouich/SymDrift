import logging
import uuid

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch_geometric.loader import DataLoader as GeometricDataLoader
from tqdm import tqdm
from tspath.utils import print_config

logger = logging.getLogger(__name__)
OmegaConf.register_new_resolver("uuid", lambda x: str(uuid.uuid1()))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@hydra.main(
    config_path="../../../configs", version_base="1.2", config_name="flow_matching_inference"
)
def main(cfg):
    
    logger.info("Starting inference...")
    if cfg.get("print_config", True):
        fields = (
            "run",
            "diffusion",
            "dataset",
            "seed",
        )
        print_config(cfg, fields=fields, resolve=False)
        
    ########## Hyperparameters and settings ##########
    pl.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision("medium")

    ########## Dataset ##########
    test_dataset = instantiate(cfg.dataset.test_dataset)
    test_dataloader = GeometricDataLoader(
        dataset=test_dataset, **cfg.dataset.test_dataloader
    )

    diff_process = instantiate(cfg.diffusion)
    logger.info("Loading model checkpoint: <{}>".format(cfg.diffusion.checkpoint_path))
    state_dict = torch.load(
        cfg.diffusion.checkpoint_path, weights_only=False, map_location=device
    )["state_dict"]
    diff_process.load_state_dict(state_dict)
    diff_process.to(device)
    diff_process.eval()

    rmsds = []
    for batch in tqdm(test_dataloader, desc="Evaluating test dataset"):
        batch = batch.to(device)
        _, rmsd = diff_process.sample(
            batch=batch,
            num_steps=cfg.diffusion.n_sample_steps,
            save_folder=f"nfe_{cfg.diffusion.n_sample_steps}",
            save_trajectory=getattr(cfg.diffusion, "save_trajectory", False),
        )
        rmsds.append(rmsd)
    rmsds = torch.cat(rmsds, dim=0)
    logger.info(
        f"Final Test RMSD:\n"
        f"  mean: {rmsds.mean().item():.4f} Å\n"
        f"  median: {rmsds.median().item():.4f} Å"
    )


if __name__ == "__main__":
    main()
