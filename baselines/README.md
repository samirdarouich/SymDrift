# Conformer Generation Baselines — QM9

Inference speed benchmarks for 5 molecular conformer generation methods on QM9. All benchmarks use **randomly initialized weights** — the goal is to measure sampling algorithm speed, not output quality. No checkpoint downloads required.

## Methods

| Method | Repo | Conda Env | Python | Sampling Algorithm |
|--------|------|-----------|--------|--------------------|
| GeoDiff | [MinkaiXu/GeoDiff](https://github.com/MinkaiXu/GeoDiff) | `geodiff` | 3.7 | Langevin dynamics (5000 steps) |
| GeoMol | [PattanaikL/GeoMol](https://github.com/PattanaikL/GeoMol) | `geomol` | ≥3.7.9 | Torsion-angle prediction (single pass) |
| MCF | [apple/ml-mcf](https://github.com/apple/ml-mcf) | `mcf` | 3.10 | DDIM (50 steps, PerceiverIO) |
| Torsional Diffusion | [gcorso/torsional-diffusion](https://github.com/gcorso/torsional-diffusion) | `torsional_diffusion` | 3.9 | Score-based diffusion on torsion angles (20 steps) |
| ET-Flow | [shenoynikhil/ETFlow](https://github.com/shenoynikhil/ETFlow) | `etflow` | ≥3.8 | Flow matching ODE (50 steps) |

## Prerequisites

- Conda (Miniconda or Anaconda) with `conda` on PATH
- CUDA-compatible GPU recommended
- `wget` (macOS: `brew install wget`)

## Directory Layout

```
baselines/
├── README.md
├── setup_all.sh          # create all conda environments
├── benchmark.sh          # run inference speed benchmark for all methods
├── scripts/
│   ├── geodiff_setup.sh
│   ├── geodiff_sample.sh
│   ├── geomol_setup.sh
│   ├── geomol_sample.sh
│   ├── mcf_setup.sh
│   ├── mcf_sample.sh
│   ├── tordiff_setup.sh
│   ├── tordiff_sample.sh
│   ├── etflow_setup.sh
│   └── etflow_sample.sh
├── GeoDiff/              # cloned repo
├── GeoMol/
├── ml-mcf/
├── torsional-diffusion/
└── ETFlow/
```

## Quick Start

```bash
# 1. Clone all repos (from project root)
bash clone_baselines.sh

# 2. Set up conda environments
cd baselines
bash setup_all.sh

# 3. Run benchmark
bash benchmark.sh
```

Results are printed to stdout and saved to `benchmark_results.txt`.

## Running a Single Method

```bash
bash scripts/<method>_setup.sh    # create conda env (one-time)
bash scripts/<method>_sample.sh   # run benchmark
```

Where `<method>` is one of: `geodiff`, `geomol`, `mcf`, `tordiff`, `etflow`.

## Timing Methodology

Each `_sample.sh` script:
1. **Warmup** — runs 5 molecules to load the model onto GPU and prime caches (not timed)
2. **Benchmark** — runs 50 molecules, wall-clock time measured with `time.perf_counter()` with `torch.cuda.synchronize()` before/after
3. **Output** — prints `Avg inference speed: X ms/sample  (total Ys for 50 samples)`

Models are instantiated with **random weights** using each repo's default QM9 architecture config. Inputs are real QM9-like SMILES processed through each method's own data pipeline.


GeoDiff              | [GeoDiff] Avg inference speed: 21.6 ms/sample  
GeoMol               | [GeoMol] Avg inference speed: 2.6 ms/sample  
MCF                  | [MCF] Avg inference speed: 4482.0 ms/sample 
ETFlow               | [ETFlow] Avg inference speed: 20.5 ms/sample  