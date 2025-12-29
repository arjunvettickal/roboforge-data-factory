# RoboForge Data Factory

**RoboForge Data Factory** is a synthetic data generation (SDG) pipeline built on **NVIDIA Isaac Sim** and **Omniverse Replicator**. It generates high-quality, annotated datasets (**COCO** and **YOLO** formats) for **industrial object detection** and **pose estimation**.

It supports:
- **Static pose randomization**
- **Physics-based (PhysX) object dropping**
- **Material, lighting, and camera randomizations**

---

##  Features

### Multi-Mode SDG
- **Pose-Based** (`object_based_sdg.py`)  
  Fast, non-physics randomization using continuous rotation and non-overlapping placement.

- **Physics-Based** (`object_based_sdg_physX.py`)  
  Simulates objects falling and settling on a floor for realistic pile-ups, including a specialized **"Dark Mode"** with spot lighting.

- **Minimalist** (`object_based_sdg_min.py`)  
  A streamlined version for quick testing and lightweight deployments (under development).

### Dynamic Randomization
- Automatic swapping of floor textures and USD environments  
- MDL material randomization for both labeled assets and distractors  
- Lighting randomization (Key/Fill + Dome light) and camera pose jittering  

### Dataset Export
- Direct export to **COCO format**
- Automatic post-processing conversion to **YOLO format** with configurable **train/val/test splits**

### Training Integration
- `train_script.py` allows immediate training of YOLO models (via **Ultralytics**) on generated synthetic data.

---

##  Project Structure

| Path | Description |
|------|-------------|
| `object_based_sdg.py` | Main script for pose-randomized synthetic data generation |
| `object_based_sdg_physX.py` | SDG script featuring PhysX dynamics and complex lighting modes |
| `object_based_sdg_min.py` | Minimal SDG script for quick testing |
| `object_based_sdg_utils.py` | Core utility functions for USD transformations, physics setup, and Replicator integration |
| `train_script.py` | Training pipeline for YOLOv8/YOLO11 using generated datasets |
| `config/` | YAML configs for assets, labels, simulation parameters, and writer settings |
| `my_WS.code-workspace` | VS Code workspace configuration |

---

##  Setup & Requirements

### NVIDIA Isaac Sim
- Ensure a working installation of **Isaac Sim** (**v4.0+ recommended**).

### Dependencies
- Standard Isaac Sim libraries
- `yaml`
- `ultralytics` (for training)

### Assets (NVIDIA Nucleus)
This pipeline pulls assets from NVIDIA Nucleus by default:

- `omniverse://localhost/`

Make sure:
- Your **local Nucleus server is running**
- Asset paths in `config/` match your setup

---

##  Usage

### 1) Generating Data

Run the SDG pipeline by specifying a configuration file.

**Pose-based generation:**
```bash
python object_based_sdg.py --config config/object_based_sdg_config.yaml

**Physics-based generation (object dropping):**
```bash
python object_based_sdg_physX.py --config config/object_based_sdg_physX_config.yaml

### 2) Training a Model

After generation + YOLO conversion (usually in yolo_out/), run:

python train_script.py yolo_out --model yolov8n.pt


Training outputs will be saved under:

yolo_out/runs/

 Configuration

The pipeline is modular via YAML files in config/. You can customize:

labeled_assets_and_properties: URLs for USD parts and semantic labels

working_area_size: Bounds of the simulation floor

asset_mdl_materials: Library of MDL materials randomly applied to assets

Writer settings: Output directory, image resolution, subframe counts for raytracing

📝 License

This project is licensed under the Apache License, Version 2.0.
