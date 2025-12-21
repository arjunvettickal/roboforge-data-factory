"""
Train and evaluate a YOLO model on the synthetic dataset saved in ``yolo_out``.

- Saves all runs/checkpoints under ``yolo_out/runs`` so everything stays together.
- Adds ``images/test``/``labels/test`` as the held-out split for post-train evaluation.
- Tries to resume from the latest checkpoint if it exists (see the block below).
"""

from __future__ import annotations

import time
import pathlib
import yaml
import argparse
from ultralytics import YOLO

# ---- defaults for this workflow ----
# Use a valid Ultralytics model name (e.g., "yolov8n.pt" or "yolo11n.pt") or an absolute/relative path to a .pt file.
MODEL = "yolov8n.pt"
EPOCHS = 500
IMGSZ = 720
BATCH = 8
WORKERS = 4
DEVICE = 0
PATIENCE = 30  # early-stop patience; keep the request's default
RUN_NAME = "train_results"


ABLATION_ROOT = pathlib.Path(r"E:\Arjun\Ablation Study")

def resolve_dataset_root(dataset_arg: str) -> pathlib.Path:
    """
    Resolve the dataset root directory.
    1. Checks if the argument is an absolute path or exists relative to CWD.
    2. Checks if it exists inside ABLATION_ROOT.
    3. Fallback: attempts to find 'yolo_out' (default) relative to the script.
    """
    # 1. Direct path check
    candidate = pathlib.Path(dataset_arg)
    if candidate.is_dir():
        return candidate.resolve()

    # 2. Check inside Ablation Study folder
    candidate = ABLATION_ROOT / dataset_arg
    if candidate.is_dir():
        return candidate.resolve()

    # 3. Legacy fallback (only if default "yolo_out" was requested but not found yet)
    if dataset_arg == "yolo_out":
        here = pathlib.Path(__file__).resolve()
        for parent in [here.parent, *here.parents]:
            candidate = parent / "yolo_out"
            if candidate.is_dir():
                return candidate
                
    # Return directly so error can be raised by caller if missing
    return pathlib.Path(dataset_arg).resolve()


def load_data_config(yolo_out_root: pathlib.Path) -> tuple[pathlib.Path, dict]:
    """
    Load data.yaml, sanity-check required keys/directories, and return the YAML path plus the parsed dict.
    If the YAML path is relative, rewrite it in place to an absolute path so Ultralytics resolves splits correctly.
    If split folders are missing, they are created so training can proceed.
    """
    data_cfg_path = yolo_out_root / "data.yaml"
    if not data_cfg_path.is_file():
        raise FileNotFoundError(f"data.yaml not found at {data_cfg_path}")

    with data_cfg_path.open("r") as f:
        data_cfg = yaml.safe_load(f) or {}

    required_keys = ("path", "train", "val", "test", "nc", "names")
    missing = [k for k in required_keys if k not in data_cfg]
    if missing:
        raise ValueError(f"data.yaml at {data_cfg_path} is missing keys: {missing}. Regenerate or edit the file.")

    # Resolve dataset root from data.yaml; if relative, make it absolute and rewrite data.yaml in place.
    path_field = pathlib.Path(data_cfg["path"])
    dataset_root = path_field if path_field.is_absolute() else (data_cfg_path.parent / path_field).resolve()
    if not path_field.is_absolute():
        data_cfg["path"] = str(dataset_root)
        with data_cfg_path.open("w") as f:
            yaml.safe_dump(data_cfg, f, sort_keys=False)
        print(f"[train_script] Updated data.yaml path to absolute: {data_cfg['path']}")

    # Ensure required split folders exist; create if missing to avoid failures.
    created_dirs = []
    for split_key in ("train", "val", "test"):
        rel = data_cfg[split_key]
        split_dir = pathlib.Path(rel)
        if not split_dir.is_absolute():
            split_dir = dataset_root / split_dir
        if not split_dir.is_dir():
            split_dir.mkdir(parents=True, exist_ok=True)
            created_dirs.append(split_dir)
    if created_dirs:
        print(f"[train_script] Created missing split directories: {created_dirs}")

    return data_cfg_path, data_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO on a specific dataset for ablation study.")
    parser.add_argument("dataset", nargs="?", default="yolo_out", help="Folder name (e.g. yolo_out_00) in Ablation dir, or path.")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model name or path (e.g. yolov8s.pt)")
    args = parser.parse_args()

    yolo_out_root = resolve_dataset_root(args.dataset)
    if not yolo_out_root.is_dir():
        print(f"Error: Dataset directory not found: {yolo_out_root}")
        print(f"Search base: {ABLATION_ROOT}")
        return

    # Derive run name from model name to keep results separate
    model_name_stem = pathlib.Path(args.model).stem
    run_name = f"train_{model_name_stem}"

    print(f"Training on dataset: {yolo_out_root}")
    print(f"Model: {args.model} -> Run Folder: {run_name}")

    data_cfg_path, data_cfg = load_data_config(yolo_out_root)

    project_dir = yolo_out_root / "runs"
    save_dir = project_dir / "detect" / run_name
    last_ckpt = save_dir / "weights" / "last.pt"

    # If last.pt exists, we resume training in-place so it continues the same run directory.
    if last_ckpt.is_file():
        print(f"Resuming training from: {last_ckpt}")
        model = YOLO(str(last_ckpt))
        resume_flag = True
    else:
        print("No previous checkpoint found; starting fresh")
        try:
            model = YOLO(args.model)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"MODEL '{args.model}' not found. Use a valid Ultralytics model alias (e.g., 'yolov8n.pt' or 'yolo11n.pt') "
                "or provide an absolute/relative path to an existing .pt checkpoint."
            ) from exc
        resume_flag = False

    train_kwargs = dict(
        # Ultralytics expects a path/str, not a dict, for data in this version.
        data=str(data_cfg_path),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        name=run_name,
        workers=WORKERS,
        device=DEVICE,
        project=str(project_dir / "detect"),
        patience=PATIENCE,
        resume=resume_flag,
    )

    t0 = time.time()
    results = model.train(**train_kwargs)
    print(f"\nTraining time: {(time.time() - t0) / 60:.1f} minutes")

    # Evaluate the best weights on the held-out test split.
    best_weights = pathlib.Path(model.trainer.save_dir) / "weights" / "best.pt"
    if not best_weights.is_file():
        raise FileNotFoundError(f"best.pt not found at {best_weights}")

    test_model = YOLO(best_weights)
    metrics = test_model.val(
        # Use the YAML path (string) for validation to match Ultralytics expectations.
        data=str(data_cfg_path),
        split="test",
        imgsz=IMGSZ,
        device=DEVICE,
        project=str(project_dir / "detect"),
        name=f"validation_results",
        save_json=True,
    )

    print(f"Test set mAP50-95: {metrics.box.map:.4f}")
    print(f"Run artifacts saved under: {model.trainer.save_dir}")
    val_save_dir = getattr(metrics, "save_dir", None)
    if val_save_dir is None:
        val_save_dir = project_dir / "detect" / f"{run_name}_test_eval"
    print(f"Test evaluation saved under: {val_save_dir}")


if __name__ == "__main__":
    main()
