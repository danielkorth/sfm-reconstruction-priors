<div align="center">

# Revisiting Structure from Motion with 3D Reconstruction Priors
[Daniel Korth](https://danielkorth.io/), [Matthias Niessner](https://www.niessnerlab.org/) <br>
Technical University of Munich

[Report](docs/static/pdfs/report.pdf) | [Project Page](https://danielkorth.github.io/sfm-reconstruction-priors/) | [Slides](docs/static/pdfs/slides.pdf)

<img src="docs/static/images/pipeline.png" alt="Pipeline" width="100%">
</div>

We aim to add 3D constraints to the incremental SfM pipeline by proposing a new point-to-point global optimization term.

## Installation

### Environment Setup

```bash
# clone project
git clone https://github.com/danielkorth/sfm-reconstruction-priors
cd sfm-reconstruction-priors

# create conda environment and install dependencies
conda env create -f environment.yaml -n guided

# activate conda environment
conda activate guided
```

### Installing Submodules

```bash
git submodule update --init --recursive
```

### Download pre-trained models

First, download the required model weights / the one you want to use:

```bash
mkdir -p checkpoints/
# DUSt3R
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth -P checkpoints/dust3r
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_linear.pth -P checkpoints/dust3r
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth -P checkpoints/dust3r

# MASt3R
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/mast3r

# VGGT
wget https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt -P checkpoints/vggt
```

### Development Setup

For development, install additional dependencies:

```bash
# Install development dependencies
pip install -e ".[dev]"
```

## Data Structure

The project uses the ScanNet++ v2 dataset. The data folder structure should be organized as follows:

```
data/
└── scannetppv2/
    ├── data/                    # Scene data
    │   └── <scene_id>/         # Each scene has a unique ID
    │       ├── iphone/         # iPhone camera data
    │       ├── dslr/           # DSLR camera data
    │       └── scans/          # 3D scan data
    ├── metadata/               # Dataset metadata
    │   ├── semantic_classes.txt
    │   ├── instance_classes.txt
    │   ├── scene_types.json
    │   └── semantic_benchmark/
    └── splits/                 # Dataset splits
        ├── nvs_sem_train.txt
        ├── nvs_sem_val.txt
        ├── nvs_test.txt
        ├── nvs_test_small.txt
        └── sem_test.txt
```

Each scene directory contains data from different sensors (iPhone, DSLR, and 3D scans). The metadata folder contains class definitions and scene information, while the splits folder contains the train/val/test splits for different tasks.

## Reproduce Experiments from Report

All the experiments for the report are in the `scripts/` folder.

Before running them, you need to precompute the matches for the datasets.

```bash
sh scripts/precompute_matches.sh
```

Afterwards, you can run all the experiments for the report, for example for the main table:

```bash
sh scripts/main_table/run_all.sh
```

Experiments were conducted on Ubuntu 22.04 LTS with an Intel Xeon W-1370P CPU (8 cores, 16 threads) and NVIDIA RTX 3080 GPU (10GB VRAM).
