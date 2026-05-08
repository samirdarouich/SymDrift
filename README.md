# SymDrift: One-Shot Generative Modeling under Symmetries

Official implementation of  
**"SymDrift: One-Shot Generative Modeling under Symmetries"**

> Generative modeling of physical systems often requires respecting global symmetries such as rotations and translations. SymDrift introduces symmetry-aware drifting models for efficient one-shot generation in molecular systems, enabling up to **40× faster inference** compared to multi-step diffusion and flow matching approaches while maintaining competitive performance.

---

## Overview

SymDrift addresses a fundamental challenge in one-shot generative modeling under symmetries:  
even if a generator is equivariant, the corresponding drifting field is generally **not** invariant under symmetrization of the target distribution.

To overcome this issue, SymDrift introduces two complementary approaches:

- **Symmetrized coordinate-space drifting**
  - based on optimal alignment of molecular structures
- **G-invariant latent embeddings**
  - removing symmetry ambiguity by construction

---

## Installation

Clone the repository:

```bash
git clone <repo-url>
cd symdrift
```

Create an environment and install dependencies:

```bash
conda create -n symdrift python=3.10
conda activate symdrift
pip install -e .
pip install torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
```

### CUDA-Accelerated Hungarian Algorithm

For improved performance of the Hungarian matching step, we use the CUDA-compatible implementation provided by:

- [torch-linear-assignment](https://github.com/ivan-chai/torch-linear-assignment/tree/main?utm_source=chatgpt.com)

In some environments, the default installation may fail to properly detect or compile the CUDA kernels. If this occurs, first uninstall the existing package before reinstalling it from source:

```bash
pip uninstall torch-linear-assignment
```

and reinstall using:

```bash
pip install --no-build-isolation torch-linear-assignment
```

---

## Training

Training is managed using Hydra configs.

```bash
symdrift_train experiment=geom_qm9_equi_coordinate_space
```

Outputs, checkpoints, and logs are stored automatically in Hydra output directories.

---

## Sampling

Generate samples from a trained checkpoint:

```bash
symdrift_sample experiment=geom_qm9_equi_coordinate_space generative_model.pretrained=/your/custom/reference/path
```

## Reproduction

Pretrained models and processed data will be published soon.

---

## Citation

If you use this work, please cite:

```bibtex
@article{darouich2026symdriftoneshotgenerativemodeling,
    title={SymDrift: One-Shot Generative Modeling under Symmetries}, 
    author={Samir Darouich and Vinh Tong and Lluís Pastor-Pérez and Tanja Bien and Loay Mualem and Mathias Niepert},
    year={2026},
    eprint={2605.06140},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    url={https://arxiv.org/abs/2605.06140}, 
}
```
---

## License

This project is released under the MIT License.