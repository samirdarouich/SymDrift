import logging
import os
import socket
import uuid
import json

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm
from tspath.analysis import evaluate_covmat, print_covmat_results
from tspath.utils import print_config, RankedLogger

OmegaConf.register_new_resolver("uuid", lambda x: str(uuid.uuid1()))
OmegaConf.register_new_resolver("replace", lambda s, old, new: s.replace(old, new))

log = RankedLogger(__name__, rank_zero_only=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(config_path="configs", version_base="1.2", config_name="base")
def train(cfg):

    log.info("Running on host: " + str(socket.gethostname()))
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
    torch.set_float32_matmul_precision("medium")

    ########## Dataset ##########
    train_dataset = instantiate(cfg.dataset.train_dataset)
    val_dataset = instantiate(cfg.dataset.val_dataset)

    train_dataloader = instantiate(cfg.dataset.train_dataloader, dataset=train_dataset)
    val_dataloader = instantiate(cfg.dataset.val_dataloader, dataset=val_dataset)

    ########## Callbacks and Logger ##########
    callbacks, loggers = [], []
    for callback_name, callback in cfg.callbacks.items():
        callbacks.append(instantiate(callback))
    for logger_name, logger in cfg.loggers.items():
        loggers.append(instantiate(logger))

    ########## Train the generative model #############
    generative_process = instantiate(cfg.generative_model)

    # Load pretrained model if specified
    if cfg.generative_model.pretrained is not None:
        log.info(
            f"\n\nLoading pretrained model from <{cfg.generative_model.pretrained}>\n\n"
        )
        pretrained = torch.load(
            cfg.generative_model.pretrained, "cpu", weights_only=False
        )

        if isinstance(pretrained, torch.nn.Module):
            state_dict = pretrained.state_dict()
        elif isinstance(pretrained, dict):
            state_dict = pretrained["state_dict"]
            if "model" in list(state_dict.keys())[0]:
                state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        generative_process.model.load_state_dict(state_dict)

    generative_trainer = pl.Trainer(
        callbacks=callbacks,
        logger=loggers,
        default_root_dir=os.path.join(cfg.run.id),
        **cfg.trainer,
    )
    generative_trainer.fit(
        generative_process,
        train_dataloader,
        val_dataloader,
        ckpt_path=cfg.run.ckpt_path,
        weights_only=False,
    )

    log.info("Training completed.")


@hydra.main(config_path="configs", version_base="1.2", config_name="base")
def sample(cfg):

    log.info("Running on host: " + str(socket.gethostname()))
    log.info("Starting inference for run: {}".format(cfg.run.id))
    if cfg.get("print_config", True):
        fields = (
            "run",
            "generative_model",
            "model",
            "dataset",
            "seed",
        )
        print_config(cfg, fields=fields, resolve=False)

    ########## Hyperparameters and settings ##########
    log.info("Setting random seed to {}".format(cfg.seed))
    pl.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision("medium")

    ########## Dataset ##########
    test_dataset = instantiate(cfg.dataset.test_dataset)
    test_dataloader = instantiate(cfg.dataset.test_dataloader, dataset=test_dataset)
    atoms_dataset = test_dataset.get_dataset_as_atoms()

    generative_process = instantiate(cfg.generative_model)
    log.info("Loading model checkpoint: <{}>".format(cfg.generative_model.pretrained))
    state_dict = torch.load(
        cfg.generative_model.pretrained, weights_only=False, map_location=device
    )["state_dict"]
    generative_process.load_state_dict(state_dict)
    generative_process.to(device)
    generative_process.eval()

    nfe = getattr(cfg.generative_model, "n_sample_steps", 1)
    guidance_scale = getattr(cfg.generative_model, "guidance_scale", 0.0)
    conditioned = getattr(cfg.generative_model, "conditioned", True)
    no_of_samples = getattr(cfg.generative_model, "no_of_samples", 1)
    threshold = getattr(cfg.generative_model, "threshold", 0.5)
    prior_type = generative_process.prior_sampler.type
    save_folder = f"nfe_{nfe}_gs_{guidance_scale}"
    
    log.info(
        f"Sampling {no_of_samples} time(s) with:\n"
        f"  NFE = {nfe} "
        f"  guidance_scale = {guidance_scale}"
        f"  conditioned = {conditioned}"
        f"  prior_type = {prior_type}"
    )

    sample_seed = getattr(cfg, "sample_seed", None)
    if sample_seed is not None:
        log.info(f"Setting sample seed for every batch to {sample_seed}")

    metrics = {}
    atoms_generated = []
    for batch in tqdm(test_dataloader, desc="Evaluating test dataset"):
        batch = batch.to(device)
        batch_atoms_generated, batch_metrics = generative_process.sample(
            batch_pos=batch,
            num_steps=nfe,
            n_samples=no_of_samples,
            save_folder=save_folder,
            save_trajectory=getattr(cfg, "save_trajectory", False),
            save_pca_plot=getattr(cfg, "save_pca_plot", False),
            conditioned=conditioned,
            guidance_scale=guidance_scale,
            # Fix initial seed for each batch (in case of gaussian prior, this would be
            # the same prior for each batch).
            seed=sample_seed, 
        )
        atoms_generated.extend(batch_atoms_generated)
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

    # Evaluate coverage and matching for the whole dataset
    ratio = getattr(cfg.generative_model, "ratio", 2.0)
    log.info(
        f"Analysing coverage and matching (threshold: {threshold:.2f}, ratio: {ratio:.0f}):"
    )
    results = evaluate_covmat(
        atoms_generated,
        atoms_dataset,
        thresholds=np.arange(0.05, 3.05, 0.05),
        num_workers=getattr(cfg.generative_model, "num_workers", 8),
        worker_fn_type=getattr(
            cfg.generative_model, "worker_fn_type", "rmsd_rdkit_wo_h"
        ),
        ratio=ratio,  # only keep at most 2*n_conformers predictions per reference
    )
    df, metrics_cov = print_covmat_results(results, threshold=threshold)
    
    # Log results
    for k, v in metrics_cov.items():
        log.info(f"{k}: {v}")
        
    df.to_csv(os.path.join(save_folder, "covmat_results.csv"), index=False)
    with open(os.path.join(save_folder, "covmat_metrics.json"), "w") as f:
        json.dump({"Ratio": ratio, **metrics_cov}, f, indent=4)
    log.info("Inference completed.")


def run_covmat_evaluation(
    path_generated: str,
    path_dataset: str,
    num_workers: int = 8,
    worker_fn_type: str = "rmsd_rdkit_wo_h",
    threshold: float = 0.5,
    ratio: float = 2.0,
    save_folder: str = "covmat_evaluation_results",
):
    from ase.io import read

    log.info("Reading generated and dataset conformers from .xyz files...")
    
    log.info(f"Generated conformers path: {path_generated}")
    atoms_generated = read(path_generated, ":")

    log.info(f"Dataset conformers path: {path_dataset}")
    atoms_dataset = read(path_dataset, ":")

    log.info(
        f"Loaded {len(atoms_generated)} generated conformers and {len(atoms_dataset)} reference conformers."
    )
    log.info(
        f"Analysing coverage and matching (threshold: {threshold:.2f}, ratio: {ratio:.0f}):"
    )
    results = evaluate_covmat(
        atoms_generated,
        atoms_dataset,
        thresholds=np.arange(0.05, 3.05, 0.05),
        num_workers=num_workers,
        worker_fn_type=worker_fn_type,
        ratio=ratio,  # only keep at most 2*n_conformers predictions per reference
    )
    df, metrics_cov = print_covmat_results(results, threshold=threshold)
    
    # Log results
    for k, v in metrics_cov.items():
        log.info(f"{k}: {v}")

    # Save results
    os.makedirs(save_folder, exist_ok=True)
    df.to_csv(os.path.join(save_folder, "covmat_results.csv"), index=False)
    with open(os.path.join(save_folder, "covmat_metrics.json"), "w") as f:
        json.dump({"Ratio": ratio, **metrics_cov}, f, indent=4)
    

    log.info("Analysis completed.")
