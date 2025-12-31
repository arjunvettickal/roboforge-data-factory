# RoboForge Data Factory
*A customizable Synthetic Data Generation pipeline for physical .*

RoboForge Data Factory is a synthetic data generation (SDG) pipeline built on **NVIDIA Isaac Sim** and **Omniverse Replicator** to generate photorealistic images and annotations for **object detection** (and optionally pose-centric workflows) in robotics / industrial scenes.

It supports:
- **Static pose randomization** (fast, non-physics)
- **Physics-based (PhysX) object dropping** (realistic pile-ups)
- **Material, lighting, camera, and environment randomization**
- Export in **COCO** and **YOLO** formats (including a custom YOLO writer)

---

## Table of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation & Environment](#installation--environment)
- [Getting Started (Recommended Tutorial: Public Dataset)](#getting-started-recommended-tutorial-public-dataset)
- [Customize Assets (Replace Default URLs)](#customize-assets-replace-default-urls)
- [Run the Pipeline](#run-the-pipeline)
- [Configuration](#configuration)
- [Training, Validation, Deployment (Docs)](#training-validation-deployment-docs)
- [References & Disclaimer](#references--disclaimer)
- [License](#license)

---

## Features

### Multi-Mode SDG
- **Pose-Based** (`object_based_sdg.py`)  
  Fast, non-physics randomization using continuous rotation and non-overlapping placement.

- **Physics-Based** (`object_based_sdg_physX.py`)  
  Simulates objects falling and settling on a floor for realistic pile-ups, including a specialized **“Dark Mode”** with spot lighting.

- **Minimalist** (`object_based_sdg_min.py`)  
  A streamlined version for quick testing and lightweight deployments (under development).

### Dynamic Domain Randomization
- Automatic swapping of floor textures and USD environments  
- MDL material randomization for both labeled assets and distractors  
- Lighting randomization (Key/Fill + Dome light) and camera pose jittering  

### Dataset Export
- COCO-style labeled outputs
- YOLO-format outputs (via the repo’s YOLO workflow / custom writer)

### Training Integration
- `train_script.py` enables quick training of YOLO models (Ultralytics) on generated synthetic data.

---

## Prerequisites

### 1) Install NVIDIA Isaac Sim
Install Isaac Sim for your platform (Windows/Linux, workstation/container, etc.):

- Isaac Sim Installation Docs: https://docs.isaacsim.omniverse.nvidia.com/ (navigate to **Installation**)

> **Tip:** Cloning this repo *inside your Isaac Sim install directory* makes it easier to run scripts using Isaac Sim’s bundled Python.

### 2) Install Conda (recommended)
Install **Miniconda** or **Anaconda** for environment management:
- Miniconda: https://docs.conda.io/en/latest/miniconda.html

### 3) Python version must match Isaac Sim
Isaac Sim is built against a specific Python version. Make sure your environment matches your Isaac Sim major version:
- **Isaac Sim 4.x → Python 3.10**
- **Isaac Sim 5.x+ → Python 3.11**

(If you run scripts using Isaac Sim’s `python.sh`/`python.bat`, you’re automatically using the correct Python.)

---

## Installation & Environment

### Option A (Recommended): Run with Isaac Sim’s bundled Python
This is the least painful path for Replicator pipelines.

1) **Clone this repo** (see next section)
2) Install Python deps into Isaac Sim’s Python:
```bash
# From your Isaac Sim install root (where python.sh/python.bat lives)
./python.sh -m pip install -r requirements.txt
```

Windows users typically run:
```bash
python.bat -m pip install -r requirements.txt
```
### Option B: Conda environment (useful for training / tooling)

Create a conda env that matches your Isaac Sim Python version, then install repo deps:
```bash
# Isaac Sim 5.x example (Python 3.11)
conda create -n roboforge-sdg python=3.11 -y
conda activate roboforge-sdg
pip install -r requirements.txt
```

If you’re on Isaac Sim 4.x, use ```bash python=3.10. ```

Note: Running Replicator standalone scripts still typically works best via Isaac Sim’s ```bash python.sh/python.bat ```(Option A). Many users keep Conda mainly for training/analysis, and use Isaac Sim’s Python for SDG.
---
## Getting Started (Recommended Tutorial: Public Dataset)

The fastest way to confirm your setup is to start with the public dataset included in this repository (recommended tutorial workflow), then switch to your own assets.

### 1) Clone the repo (Preferred: inside Isaac Sim install folder)

Preferred
```bash
# Example: go to your Isaac Sim install root
cd <PATH_TO_ISAAC_SIM_INSTALL>
git clone <YOUR_REPO_URL> roboforge-data-factory
cd roboforge-data-factory
```
Alternative (any location)
```bash
git clone <YOUR_REPO_URL>
cd <REPO_FOLDER>
```
### 2) Find the public dataset config

Locate the included public dataset config in```bash config/```(look for something like *public*dataset*.yaml) and use it as your baseline tutorial config.

### 3) Run a small test generation
```bash
# Pose-based
./python.sh object_based_sdg.py --config config/<PUBLIC_DATASET_CONFIG>.yaml

# Physics-based (optional)
./python.sh object_based_sdg_physX.py --config config/<PUBLIC_DATASET_PHYSX_CONFIG>.yaml
```
### 4) Verify outputs

By default, outputs typically land under something like:

```bash
coco_out/ 
```
```bash
yolo_out/
```
(Exact folders depend on your writer settings in the YAML.)

### 5) Train a quick YOLO baseline (optional)
```bash
./python.sh train_script.py yolo_out --model yolov8n.pt
```
---
## Customize Assets (Replace Default URLs)

Once the public dataset tutorial works end-to-end, switch to your own assets.

### What you’ll change

In your selected YAML under `config/`, update fields like:

- `labeled_assets_and_properties` (USD URLs + labels)
- `distractor_assets` (optional)
- environment / floor USDs (if applicable)

### Asset URL types you can use

- **Local filesystem**  
  e.g. `file:///.../my_asset.usd`

- **Nucleus server**  
  e.g. `omniverse://<server>/Projects/.../my_asset.usd`

- **Local Nucleus** (commonly starts as)  
  `omniverse://localhost/`

### Step-by-step instructions

Follow this guide for replacing the default URLs with your own asset URLs (local or Nucleus-hosted:  
**Asset URL Customization Guide:** (Link to instructions for these steps)

> If you’re using Nucleus, make sure the server is running and your paths are valid. You can usually copy Omniverse URLs directly from the Omniverse Content Browser / connected tools.

---

## Run the Pipeline

### Pose-based generation

```bash
./python.sh object_based_sdg.py --config config/object_based_sdg_config.yaml
```

### Physics-based generation (object dropping)
```bash
./python.sh object_based_sdg_physX.py --config config/object_based_sdg_physX_config.yaml
```
###Minimal (WIP)
```bash
./python.sh object_based_sdg_min.py --config config/object_based_sdg_min_config.yaml
```
---
## Configuration

Configs live in config/ and are designed to be the main customization surface.

Typical knobs you can tune:

- **Output**

  - **output directory**
  
  - **image resolution**
  
  - **subframes / render quality**

  - **annotation types (2D boxes, segmentation, etc.)**

  - **dataset split ratios (train/val/test)**

- **Scene layout**

  -**working_area_size / spawn bounds**

  -**collision walls / floor plane**

  -**number of distractors, clutter level**

-**Domain randomization**

  -**lighting (dome, key/fill, spot)**

  -**camera pose jitter, focal length/DoF (if enabled)**

  -**materials (MDL libraries), texture swaps**

  -**environment/background USD swapping**

-**Physics**

  -**drop height, settle time, friction/restition, rigid body parameters**

  -**“Dark Mode” lighting (if using PhysX script)**
---
## Training, Validation, Deployment (Docs)

This repo includes additional documentation pages to take you from SDG → training → evaluation → deployment:

-Train the model on generated datasets: (link)

-Validate & improve YOLO via structured ablation + logging: (link)

-Deployment strategies for real-world robotics: (link)

-Further learning / reference links: (link)
---
## References & Disclaimer
### Foundation tutorials (NVIDIA)

This pipeline is based on NVIDIA’s official Replicator / Isaac Sim tutorial:

-“Object Based Synthetic Data Generation” (foundation for SDG workflow):
https://docs.isaacsim.omniverse.nvidia.com/
---
### YOLO writer note

A YOLO format writer was developed following NVIDIA Omniverse documentation on creating custom writers:

-Custom writer reference (Omniverse Replicator docs):
https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/custom_writer.html

**This repo’s YOLO output pipeline follows the official writer patterns and adapts them for YOLO directory/layout + annotation formatting.**
---
## License

Apache License, Version 2.0

This project builds upon official NVIDIA Omniverse and Isaac Sim resources:

- Isaac Sim Installation Documentation:  
  https://docs.isaacsim.omniverse.nvidia.com/

- Omniverse Replicator – Object-Based Synthetic Dataset Generation:  
  https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/tutorials.html

- Omniverse Replicator Custom Writer Documentation:  
  https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/custom_writer.html
---
