# Conformer Generation Baselines — QM9

Benchmark scripts for 5 molecular conformer generation methods evaluated on QM9.

## Methods

| Method | Env Name | Python | Checkpoint Source |
|--------|----------|--------|-------------------|
| GeoDiff | `geodiff` | 3.7 | Google Drive |
| GeoMol | `geomol` | ≥3.7.9 | **Manual** (see note below) |
| MCF | `mcf` | 3.10 | Apple CDN |
| Torsional Diffusion | `torsional_diffusion` | 3.9 | Google Drive |
| ET-Flow | `etflow` | ≥3.8 | Zenodo |

## Prerequisites

- Conda (Miniconda or Anaconda) with `conda` on PATH
- CUDA-compatible GPU recommended
- `gdown` for Google Drive downloads:
  ```bash
  pip install gdown
  ```
- `wget` available (macOS: `brew install wget`)

## Directory Layout (after cloning)

```
baselines/
├── README.md
├── setup_all.sh          # install all conda environments
├── download_all.sh       # download all QM9 checkpoints
├── benchmark.sh          # run 50-sample timing for all methods
├── scripts/              # per-method setup / download / sample scripts
├── GeoDiff/              # cloned repo
├── GeoMol/
├── ml-mcf/
├── torsional-diffusion/
└── ETFlow/
```

## Quick Start

```bash
# 1. Clone repos (run from project root)
bash clone_baselines.sh

# 2. Set up all conda environments
cd baselines
bash setup_all.sh

# 3. Download all QM9 checkpoints
bash download_all.sh

# 4. Run 50-sample timing benchmark
bash benchmark.sh
```

Results are written to `benchmark_results.txt` and printed to stdout.

## Running Methods Individually

Each method has three scripts in `scripts/`:

```bash
bash scripts/<method>_setup.sh      # create conda env
bash scripts/<method>_download.sh   # download checkpoint
bash scripts/<method>_sample.sh     # sample 50 molecules + report timing
```

Where `<method>` is one of: `geodiff`, `geomol`, `mcf`, `tordiff`, `etflow`.

## GeoMol — No Public Checkpoint

GeoMol does **not** provide a public pretrained checkpoint. You must either:

- Request weights from the authors (see their GitHub issues)
- Train from scratch:
  ```bash
  conda run -n geomol python GeoMol/train.py --dataset qm9 --data_dir GeoMol/data/QM9
  ```

Then place the model at `GeoMol/trained_models/qm9/best_model.pt` and the config at
`GeoMol/trained_models/qm9/model_parameters.yml`.

## Timing Methodology

Each `_sample.sh` script:
1. Runs sampling on **50 QM9 test molecules** (1 conformer per molecule)
2. Measures wall-clock time using Python's `time.perf_counter()`
3. Prints `Total: Xs | Avg/sample: Yms`

The benchmark aggregates all results into a summary table.
