import logging
import hydra
from hydra.utils import instantiate
import pytorch_lightning as pl
import torch
import os
from omegaconf import OmegaConf
import uuid
from tqdm import tqdm
from tspath.utils import print_config

OmegaConf.register_new_resolver("uuid", lambda x: str(uuid.uuid1()))
OmegaConf.register_new_resolver(
    "replace",
    lambda s, old, new: s.replace(old, new)
)

log = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@hydra.main(config_path='configs',version_base='1.2',config_name='train')
def train(cfg):
    
    log.info("Starting training for run: {}".format(cfg.run.id))
    if cfg.get("print_config", True):
        fields = (
            "run",
            "globals",
            "generative_model",
            "drifting_field",
            "prior",
            "embedder",
            "model",
            "dataset",
            "trainer",
            "callbacks",
            "loggers",
            "seed",
        )
        print_config(cfg, fields=fields, resolve=False)

    ########## Hyperparameters and settings ##########
    pl.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision('medium')

    ########## Dataset ##########
    train_dataset = instantiate(cfg.dataset.train_dataset)
    val_dataset = instantiate(cfg.dataset.val_dataset)

    train_dataloader = instantiate(cfg.dataset.train_dataloader, dataset=train_dataset)
    val_dataloader = instantiate(cfg.dataset.val_dataloader, dataset=val_dataset)
    
    ########## Callbacks and Logger ##########
    callbacks,loggers = [],[]
    for callback_name, callback in cfg.callbacks.items():
        callbacks.append(instantiate(callback))
    for logger_name, logger in cfg.loggers.items():
        loggers.append(instantiate(logger))

    ########## Train the generative model #############
    diff_process = instantiate(cfg.generative_model)
    
    # Load pretrained model if specified
    if cfg.generative_model.pretrained is not None:
        log.info(f"\n\nLoading pretrained model from <{cfg.generative_model.pretrained}>\n\n")
        pretrained = torch.load(cfg.generative_model.pretrained, "cpu", weights_only=False)

        if isinstance(pretrained, torch.nn.Module):
            state_dict = pretrained.state_dict()
        elif isinstance(pretrained, dict):
            state_dict = pretrained["state_dict"]
            if "model" in list(state_dict.keys())[0]:
                state_dict = {
                    k.replace("model.", ""): v for k, v in state_dict.items()
                }
        diff_process.model.load_state_dict(state_dict)
            
    diff_trainer = pl.Trainer(
        callbacks=callbacks, 
        logger=loggers,
        default_root_dir=os.path.join(cfg.run.id),
        **cfg.trainer
    ) 
    diff_trainer.fit(diff_process, train_dataloader, val_dataloader, ckpt_path=cfg.run.ckpt_path)
    
    log.info("Training completed.")


@hydra.main(
    config_path="configs", version_base="1.2", config_name="diffusion_inference"
)
def sample(cfg):
    
    log.info("Starting inference for run: {}".format(cfg.run.id))
    if cfg.get("print_config", True):
        fields = (
            "run",
            "generative_model",
            "dataset",
            "seed",
        )
        print_config(cfg, fields=fields, resolve=False)
        
    ########## Hyperparameters and settings ##########
    pl.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision("medium")

    ########## Dataset ##########
    test_dataset = instantiate(cfg.dataset.test_dataset)
    test_dataloader = instantiate(cfg.dataset.val_dataloader, dataset=test_dataset)

    diff_process = instantiate(cfg.generative_model)
    log.info("Loading model checkpoint: <{}>".format(cfg.generative_model.checkpoint_path))
    state_dict = torch.load(
        cfg.generative_model.checkpoint_path, weights_only=False, map_location=device
    )["state_dict"]
    diff_process.load_state_dict(state_dict)
    diff_process.to(device)
    diff_process.eval()

    nfe = cfg.generative_model.n_sample_steps
    guidance_scale = getattr(cfg.generative_model, "guidance_scale", 0.0)
    conditioned = getattr(cfg.generative_model, "conditioned", True)
    log.info(
        f"Sampling with NFE={nfe}, "
        f"guidance_scale={guidance_scale}, "
        f"conditioned={conditioned}"
    )
    metrics = {}
    for batch in tqdm(test_dataloader, desc="Evaluating test dataset"):
        batch = batch.to(device)
        _, batch_metrics = diff_process.sample(
            batch=batch,
            num_steps=nfe,
            save_folder=f"nfe_{nfe}_gs_{guidance_scale}",
            save_trajectory=getattr(cfg.generative_model, "save_trajectory", False),
            conditioned=conditioned,
            guidance_scale=guidance_scale,
        )
        for k, v in batch_metrics.items():
            if k not in metrics:
                metrics[k] = []
            metrics[k].append(v)
    for k, v in metrics.items():
        metrics[k] = torch.tensor(v)
        log.info(
            f"Test {k}: mean: {metrics[k].mean().item():.4f} "
            f"median: {metrics[k].median().item():.4f}"
        )