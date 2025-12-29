RoboForge Data Factory

RoboForge Data Factory is a synthetic data generation (SDG) pipeline built on NVIDIA Isaac Sim and Omniverse Replicator. It is designed to generate high-quality, annotated datasets (COCO and YOLO formats) for industrial object detection and pose estimation. The factory supports both static pose randomization and physics-based (PhysX) object dropping, along with material, lighting, and camera randomizations.
Features

    Multi-Mode SDG:

        Pose-Based (object_based_sdg.py): Fast, non-physics randomization using continuous rotation and non-overlapping placement.

        Physics-Based (object_based_sdg_physX.py): Simulates objects falling and settling on a floor for realistic pile-ups, including a specialized "Dark Mode" with spot lighting.

        Minimalist (object_based_sdg_min.py): A streamlined version for quick testing and lightweight deployments.

    Dynamic Randomization:

        Automatic swapping of floor textures and USD environments.

        MDL material randomization for both labeled assets and distractors.

        Lighting randomization (Key/Fill and Dome light) and camera pose jittering.

    Dataset Export:

        Direct export to COCO format.

        Automatic post-processing conversion to YOLO format with configurable train/val/test splits.

    Training Integration:

        train_script.py allows for immediate training of YOLO models (via Ultralytics) on the generated synthetic data.

Project Structure

    object_based_sdg.py: Main script for pose-randomized synthetic data generation.

    object_based_sdg_physX.py: SDG script featuring PhysX dynamics and complex lighting modes.

    object_based_sdg_utils.py: Core utility functions for USD transformations, physics setup, and Replicator integration.

    train_script.py: Training pipeline for YOLOv8/YOLO11 using the generated datasets.

    config/: Contains YAML configuration files to define asset URLs, labels, and simulation parameters.

    my_WS.code-workspace: VS Code workspace configuration.

Setup & Requirements

    NVIDIA Isaac Sim: Ensure you have a working installation of Isaac Sim (v4.0+ recommended).

    Dependencies: The scripts require standard Isaac Sim libraries, yaml, and ultralytics for training.

    Assets: This pipeline is configured to pull assets from NVIDIA Nucleus (omniverse://localhost/). Ensure your local Nucleus server is running.

Usage
1. Generating Data

Run the SDG pipeline by specifying a configuration file. For example:
Bash

python object_based_sdg.py --config config/object_based_sdg_config.yaml

2. Training a Model

Once your data is generated and converted to YOLO format, run the training script:
Bash

python train_script.py yolo_out --model yolov8n.pt

Configuration

The pipeline is modular via YAML files in the config/ directory. You can customize:

    labeled_assets_and_properties: URLs for your USD parts and their semantic labels.

    working_area_size: The bounds of the simulation floor.

    asset_mdl_materials: A library of MDL materials to be randomly applied to assets.

    Writer Settings: Output directory, image resolution, and subframe counts.

License

This project is licensed under the Apache License, Version 2.0.
