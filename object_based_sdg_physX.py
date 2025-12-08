# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
from pathlib import Path

import yaml
from isaacsim import SimulationApp
import re

# Default config dict, can be updated/replaced using json/yaml config files ('--config' cli argument)
config = {
    "launch_config": {
        "renderer": "RaytracedLighting",
        "headless": False,
    },
    "env_url": "",
    "working_area_size": (4, 4, 3),
    "rt_subframes": 4,
    "num_frames": 10,
    "num_cameras": 3,
    "camera_collider_radius": 0.5,
    "disable_render_products_between_captures": False,
    "simulation_duration_between_captures": 0.05,
    "resolution": (640, 480),
    "camera_properties_kwargs": {
        "focalLength": 24.0,
        "focusDistance": 400,
        "fStop": 0.0,
        "clippingRange": (0.01, 10000),
    },
    "camera_look_at_target_offset": 0.15,
    "camera_distance_to_target_min_max": (0.25, 0.75),
    "writer_type": "PoseWriter",
    "writer_kwargs": {
        "output_dir": "_out_obj_based_sdg_pose_writer",
        "format": None,
        "use_subfolders": False,
        "write_debug_images": True,
        "skip_empty_frames": False,
    },
    "labeled_assets_and_properties": [
        {
            "url": "/Isaac/Props/YCB/Axis_Aligned/008_pudding_box.usd",
            "label": "pudding_box",
            "count": 5,
            "floating": True,
            "scale_min_max": (0.85, 1.25),
        },
        {
            "url": "/Isaac/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd",
            "label": "mustard_bottle",
            "count": 7,
            "floating": True,
            "scale_min_max": (0.85, 1.25),
        },
    ],
    "shape_distractors_types": ["capsule", "cone", "cylinder", "sphere", "cube"],
    "shape_distractors_scale_min_max": (0.015, 0.15),
    "shape_distractors_num": 350,
    "mesh_distractors_urls": [
        "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_04_1847.usd",
        "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxA_01_414.usd",
        "/Isaac/Environments/Simple_Warehouse/Props/S_TrafficCone.usd",
    ],
    "mesh_distractors_scale_min_max": (0.35, 1.35),
    "mesh_distractors_num": 75,
    "yolo_output_dir": "_out_yolo",
    "yolo_split_ratio": 0.8,
    "yolo_split_seed": 42,
    "pathtracing": {
        "enabled": False,
        "spp": 256,
        "total_spp": 256,
        "denoiser": True,
    },
}

import carb

# Check if there are any config files (yaml or json) are passed as arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=False, help="Include specific config parameters (json or yaml))")
args, unknown = parser.parse_known_args()
args_config = {}
if args.config and os.path.isfile(args.config):
    with open(args.config, "r") as f:
        if args.config.endswith(".json"):
            args_config = json.load(f)
        elif args.config.endswith(".yaml"):
            args_config = yaml.safe_load(f)
        else:
            carb.log_warn(f"File {args.config} is not json or yaml, will use default config")
else:
    carb.log_warn(f"File {args.config} does not exist, will use default config")

# Update the default config dict with the external one
config.update(args_config)

print(f"[SDG] Using config:\n{config}")

launch_config = config.get("launch_config", {})
simulation_app = SimulationApp(launch_config=launch_config)

import random
import time
from itertools import chain

import carb.settings
import numpy as np
import re

# Custom util functions for the example
import object_based_sdg_utils
import omni.replicator.core as rep
import omni.timeline
import omni.usd
import usdrt
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.utils.semantics import add_labels, remove_labels, upgrade_prim_semantics_to_labels
from isaacsim.storage.native import get_assets_root_path
from pxr import PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, Gf
from pxr import UsdShade

# Isaac nucleus assets root path
assets_root_path = get_assets_root_path()
stage = None

# Global RTX shadow softness (helps avoid razor-sharp shadows)
settings = carb.settings.get_settings()
settings.set("/rtx/shadows/softShadows", True)
settings.set("/rtx/shadows/sunArea", 0.01)   
pt_cfg = config.get("pathtracing", {})
if pt_cfg.get("enabled"):
    settings.set("/rtx/rendermode", "PathTracing")
    settings.set("/rtx/pathtracing/spp", pt_cfg.get("spp", 256))
    settings.set("/rtx/pathtracing/totalSpp", pt_cfg.get("total_spp", pt_cfg.get("spp", 256)))
    settings.set("/rtx/pathtracing/optixDenoiser/enabled", 1 if pt_cfg.get("denoiser", True) else 0)
else:
    settings.set("/rtx/rendermode", "RaytracedLighting")

# Volumetric Fog
settings.set("/rtx/fog/enabled", True)
settings.set("/rtx/fog/fogType", "linear")
settings.set("/rtx/fog/density", 0.002)
settings.set("/rtx/fog/startDistance", 0.0)
settings.set("/rtx/fog/endDistance", 100.0)
settings.set("/rtx/fog/fogColor", (0.8, 0.85, 0.9))

# Film Grain (Sensor Noise)
settings.set("/rtx/post/tonemap/filmIso", 800.0)
settings.set("/rtx/post/tonemap/enable", True)

# ENVIRONMENT
env_url = config.get("env_url", "")
if env_url:
    env_path = env_url if env_url.startswith("omniverse://") else assets_root_path + env_url
    omni.usd.get_context().open_stage(env_path)
    stage = omni.usd.get_context().get_stage()
else:
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
# -------------------------------------------------------
# FIXED TOP-DOWN CAMERA SETUP (define once)
# -------------------------------------------------------

CAMERA_DISTANCE_ABOVE_OBJECTS = 6.0   # tweak this

# Centralized light tweakables
LIGHT_SETTINGS = {
    "distant_intensity": 7000,
    "random_light_color_min": (0.9, 0.9, 0.9),
    "random_light_color_max": (1.0, 1.0, 1.0),
    "random_light_temperature_mean": 6500,
    "random_light_temperature_std": 500,
    "random_light_intensity_mean": 6000,
    "random_light_intensity_std": 500,
    "random_light_count": 3,
}

def update_camera_pose():
    if not labeled_prims:
        return

    # Look at the center of the workspace where objects are dropped
    target = np.array([0.0, 0.0, floor_height])

    # Use the configured distance range
    min_dist, max_dist = config.get("camera_distance_to_target_min_max", (0.25, 0.75))
    dist = random.uniform(min_dist, max_dist)

    # Top-down view (high elevation) to match reference images
    azimuth = random.uniform(0, 2 * np.pi)
    # Elevation between 70 and 90 degrees (0 is horizontal, 90 is vertical)
    elevation = random.uniform(np.deg2rad(70), np.deg2rad(90))

    x = dist * np.cos(elevation) * np.cos(azimuth)
    y = dist * np.cos(elevation) * np.sin(azimuth)
    z = dist * np.sin(elevation)

    cam_loc = target + np.array([x, y, z])

    # Ensure camera is above floor
    if cam_loc[2] < floor_height + 0.1:
        cam_loc[2] = floor_height + 0.1

    # Calculate orientation (Quaternion)
    eye = Gf.Vec3d(cam_loc.tolist())
    center = Gf.Vec3d(target.tolist())
    up = Gf.Vec3d(0, 0, 1)

    # Gf.Matrix4d().SetLookAt creates a view matrix (world to camera). We want camera to world.
    m = Gf.Matrix4d().SetLookAt(eye, center, up)
    m = m.GetInverse()

    # Extract rotation
    q = m.ExtractRotation().GetQuat()

    # Set for all cameras
    for cam_prim in cameras:
        object_based_sdg_utils.set_transform_attributes(
            cam_prim,
            location=tuple(cam_loc),
            orientation=Gf.Quatf(q)
        )

    print(f"[INFO] Camera updated to look at {target} from {cam_loc}")



# Add a distant light to the empty stage
distant_light = stage.DefinePrim("/World/Lights/DistantLight", "DistantLight")
distant_light.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(LIGHT_SETTINGS["distant_intensity"])
# Increase angular size to soften shadows
distant_light.CreateAttribute("inputs:angle", Sdf.ValueTypeNames.Float).Set(0.5)
if not distant_light.HasAttribute("xformOp:rotateXYZ"):
    UsdGeom.Xformable(distant_light).AddRotateXYZOp()
distant_light.GetAttribute("xformOp:rotateXYZ").Set((0, 60, 0))

# Get the working area size and bounds (width=x, depth=y, height=z)
working_area_size = config.get("working_area_size", (3, 3, 3))

# Horizontal extents stay the same
min_x = -working_area_size[0] / 2.0
max_x =  working_area_size[0] / 2.0
min_y = -working_area_size[1] / 2.0
max_y =  working_area_size[1] / 2.0

# Floor is at the very bottom of the working area
floor_height = -working_area_size[2] / 2.0

# objects will float slightly above the floor
OBJECT_Z = floor_height + 1.2    # 5 cm above floor




# We only spawn objects in a thin band above the floor, e.g. 0.1–0.4 units
spawn_min_z = floor_height + 0.80
spawn_max_z = floor_height + 1.50

working_area_min = (min_x, min_y, spawn_min_z)
working_area_max = (max_x, max_y, spawn_max_z)

def build_mdl_material_library(stage, mdl_entries, prefix="AssetMat"):
    """
    Given a list of dicts:
        [{ "mdl_url": "...", "subidentifier": "..." }, ...]
    create UsdShade.Materials under /World/Looks and return them.
    """
    if not mdl_entries:
        return []

    stage.DefinePrim("/World/Looks", "Scope")
    materials = []
    for idx, entry in enumerate(mdl_entries):
        mdl_url = entry.get("mdl_url")
        subid = entry.get("subidentifier")
        if not mdl_url:
            continue
        if not subid:
            subid = mdl_url.split("/")[-1].replace(".mdl", "")

        mtl_path = Sdf.Path(f"/World/Looks/{prefix}_{idx}")
        mtl = UsdShade.Material.Define(stage, mtl_path)
        shader = UsdShade.Shader.Define(stage, mtl_path.AppendPath("Shader"))
        shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
        shader.SetSourceAsset(mdl_url, "mdl")
        shader.SetSourceAssetSubIdentifier(subid, "mdl")
        mtl.GetPrim().CreateAttribute("user:mdl_url", Sdf.ValueTypeNames.Asset).Set(mdl_url)
        mtl.GetPrim().CreateAttribute("user:mdl_subidentifier", Sdf.ValueTypeNames.String).Set(subid)

        surf_out = mtl.CreateSurfaceOutput("mdl")
        surf_out.ConnectToSource(shader.ConnectableAPI(), "out")
        disp_out = mtl.CreateDisplacementOutput("mdl")
        disp_out.ConnectToSource(shader.ConnectableAPI(), "out")
        vol_out = mtl.CreateVolumeOutput("mdl")
        vol_out.ConnectToSource(shader.ConnectableAPI(), "out")
        materials.append(mtl)
    return materials


def randomize_asset_materials(prims, materials):
    """Randomly bind one of the given materials to each prim in 'prims'."""
    if not materials:
        return
    for prim in prims:
        mat = random.choice(materials)
        UsdShade.MaterialBindingAPI(prim).Bind(mat)


def bind_material_to_meshes(root_prim, material):
    """Bind a material to all mesh/gprim descendants of the given root."""
    if material is None:
        return
    for desc in Usd.PrimRange(root_prim):
        if desc.IsA(UsdGeom.Gprim):
            UsdShade.MaterialBindingAPI(desc).Bind(material)


def randomize_ground_plane_material(materials):
    """Randomize material on the ground plane meshes."""
    if not materials:
        return
    mat = random.choice(materials)
    # Bind at root and all mesh descendants to ensure MDL sticks
    UsdShade.MaterialBindingAPI(ground_plane_prim).Bind(mat)
    bind_material_to_meshes(ground_plane_prim, mat)
    mdl_url = mat.GetPrim().GetAttribute("user:mdl_url").Get()
    subid = mat.GetPrim().GetAttribute("user:mdl_subidentifier").Get()
    print(f"[SDG] GroundPlane material -> {subid} ({mdl_url})")


def convert_coco_to_yolo_and_split(coco_root_path, yolo_root_path=None, train_ratio=0.8, seed=42):
    """Convert COCO annotations to YOLO labels and split into train/val folders (no ultralytics dependency)."""

    def _pick_coco_json(search_root: Path, preferred_stem: str = "coco_annotations"):
        cand = []
        for p in search_root.glob("*.json"):
            cand.append(p)
        rep_dir = search_root / "Replicator"
        if rep_dir.exists():
            cand.extend(rep_dir.glob("*.json"))
        if not cand:
            return None
        def keyfunc(p):
            stem_ok = p.stem.startswith(preferred_stem)
            return (not stem_ok, -p.stat().st_size, -p.stat().st_mtime)
        cand.sort(key=keyfunc)
        return cand[0]

    coco_root = Path(coco_root_path)
    ann_path = None
    if coco_root.exists():
        ann_path = _pick_coco_json(coco_root)
    if ann_path is None and coco_root.parent.exists():
        siblings = sorted(coco_root.parent.glob(f"{coco_root.name}*"))
        for sib in reversed(siblings):
            ann_path = _pick_coco_json(sib)
            if ann_path:
                coco_root = sib
                break
    if ann_path is None:
        print(f"[YOLO-CONVERT] No COCO json found under {coco_root_path} or siblings, skipping conversion.")
        return

    try:
        with open(ann_path, "r") as f:
            coco = json.load(f)
    except Exception as exc:
        print(f"[YOLO-CONVERT] Failed to read {ann_path}: {exc}")
        return

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])
    if not images or not annotations or not categories:
        print(f"[YOLO-CONVERT] Empty COCO annotations at {ann_path}, skipping.")
        return

    cat_id_to_yolo = {cat["id"]: idx for idx, cat in enumerate(categories)}
    yolo_id_to_name = [cat.get("name", f"class_{i}") for i, cat in enumerate(categories)]

    ann_by_image = {}
    for ann in annotations:
        img_id = ann.get("image_id")
        ann_by_image.setdefault(img_id, []).append(ann)

    # Deterministic split
    yolo_root = Path(yolo_root_path) if yolo_root_path else coco_root
    img_train_dir = yolo_root / "images" / "train"
    img_val_dir = yolo_root / "images" / "val"
    lbl_train_dir = yolo_root / "labels" / "train"
    lbl_val_dir = yolo_root / "labels" / "val"
    for d in [img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir]:
        d.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed if seed is not None else 42)
    rng.shuffle(images)
    n_total = len(images)
    n_train = max(0, min(n_total, int(round(n_total * train_ratio))))
    train_images = set(img.get("id") for img in images[:n_train])

    for img in images:
        file_name = img.get("file_name")
        width = img.get("width", 1)
        height = img.get("height", 1)
        img_src = coco_root / file_name
        anns = ann_by_image.get(img.get("id"), [])
        yolo_lines = []
        for ann in anns:
            bbox = ann.get("bbox", [0, 0, 0, 0])
            if len(bbox) != 4 or width <= 0 or height <= 0:
                continue
            x, y, w, h = bbox
            cx = (x + w / 2) / width
            cy = (y + h / 2) / height
            nw = w / width
            nh = h / height
            cid = ann.get("category_id")
            yid = cat_id_to_yolo.get(cid)
            if yid is None:
                continue
            yolo_lines.append(f"{yid} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        if img.get("id") in train_images:
            img_dst = img_train_dir / img_src.name
            lbl_dst = lbl_train_dir / (img_src.stem + ".txt")
        else:
            img_dst = img_val_dir / img_src.name
            lbl_dst = lbl_val_dir / (img_src.stem + ".txt")

        if img_src.exists():
            img_dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                img_dst.write_bytes(img_src.read_bytes())
            except Exception as exc:
                print(f"[YOLO-CONVERT] Failed to copy image {img_src} -> {img_dst}: {exc}")
        lbl_dst.parent.mkdir(parents=True, exist_ok=True)
        lbl_dst.write_text("\n".join(yolo_lines))

    # data.yaml
    data_yaml = yolo_root / "data.yaml"
    names_lines = "\n".join([f"  {i}: {n}" for i, n in enumerate(yolo_id_to_name)])
    data_yaml.write_text(
        f"train: images/train\nval: images/val\nnc: {len(yolo_id_to_name)}\nnames:\n{names_lines}\n"
    )
    print(f"[YOLO-CONVERT] YOLO dataset at: {yolo_root}")


# Create a collision box area around the assets to prevent them from drifting away
object_based_sdg_utils.create_collision_box_walls(
    stage, "/World/CollisionWalls", working_area_size[0], working_area_size[1], working_area_size[2]
)

# ---- FLOOR PLANE -------------------------------------------------
# Use a physics-enabled ground plane instead of a flattened cube
floor_height = -working_area_size[2] / 2.0  # bottom of the collision box
ground_plane_size = max(working_area_size[0], working_area_size[1]) * 4.0

def create_ground_plane_mesh(stage, path, size, height):
    """Create a simple quad mesh for the ground plane with UVs."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    half_size = size / 2.0
    
    # Vertices for a quad
    points = [
        (-half_size, -half_size, 0),
        (half_size, -half_size, 0),
        (half_size, half_size, 0),
        (-half_size, half_size, 0),
    ]
    mesh.CreatePointsAttr(points)
    
    # Face vertex counts (one face with 4 vertices)
    mesh.CreateFaceVertexCountsAttr([4])
    
    # Face vertex indices
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    
    # Normals
    mesh.CreateNormalsAttr([(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)])
    
    # Texture coordinates (UVs)
    # Scale UVs to match the increased ground plane size (4x) to avoid stretched textures
    uv_scale = 4.0
    tex_coords = [(0, 0), (uv_scale, 0), (uv_scale, uv_scale), (0, uv_scale)]
    primvars_api = UsdGeom.PrimvarsAPI(mesh)
    pv = primvars_api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    pv.Set(tex_coords)
    
    # Set transform
    object_based_sdg_utils.set_transform_attributes(
        mesh.GetPrim(), location=(0.0, 0.0, height)
    )
    
    return mesh.GetPrim()

ground_plane_prim = create_ground_plane_mesh(stage, "/World/groundPlane", ground_plane_size, floor_height)
# Add collision
object_based_sdg_utils.add_colliders(ground_plane_prim)
# ------------------------------------------------------------------



# Create a physics scene to add or modify custom physics settings
usdrt_stage = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())
physics_scenes = usdrt_stage.GetPrimsWithAppliedAPIName("PhysxSceneAPI")
if physics_scenes:
    physics_scene = physics_scenes[0]
else:
    physics_scene = UsdPhysics.Scene.Define(stage, "/PhysicsScene")
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/PhysicsScene"))
physx_scene.GetTimeStepsPerSecondAttr().Set(60)


# TRAINING ASSETS
labeled_assets_and_properties = config.get("labeled_assets_and_properties", [])
labeled_prims = []
drop_height_min = floor_height + 0.5
drop_height_max = floor_height + working_area_size[2] * 0.8
mesh_distractors = []


def clear_labeled_assets():
    """Remove existing labeled assets from the stage."""
    global labeled_prims
    if stage.GetPrimAtPath("/World/Labeled").IsValid():
        stage.RemovePrim("/World/Labeled")
    stage.DefinePrim("/World/Labeled", "Xform")
    labeled_prims = []


def spawn_labeled_assets(max_label_types=None):
    """Spawn labeled assets at random XY and height, enable physics, and prepare to drop.
    Optionally limit to a subset of label types (random choice).
    """
    global labeled_prims
    clear_labeled_assets()
    spawned = []
    # Choose which label types to spawn this frame
    if max_label_types is not None:
        chosen_objs = random.sample(
            labeled_assets_and_properties, k=min(max_label_types, len(labeled_assets_and_properties))
        )
    else:
        chosen_objs = labeled_assets_and_properties

    # Force spawning at the center to create a pile, matching reference images
    # Define a small box around the center (0,0)
    spawn_width = 0.6
    spawn_height = 0.6
    spawn_min_x, spawn_max_x = -spawn_width/2, spawn_width/2
    spawn_min_y, spawn_max_y = -spawn_height/2, spawn_height/2
    print(f"[SDG] Spawning objects at center with spread {spawn_width}x{spawn_height}")

    for obj in chosen_objs:
        obj_url = obj.get("url", "")
        label = obj.get("label", "unknown")
        count = obj.get("count", 1)
        for _ in range(count):
            rand_loc, rand_rot, _ = object_based_sdg_utils.get_random_transform_values(
                loc_min=(spawn_min_x, spawn_min_y, drop_height_min),
                loc_max=(spawn_max_x, spawn_max_y, drop_height_max),
                scale_min_max=(1, 1),  # keep authored scale
            )
            prim_path = omni.usd.get_stage_next_free_path(stage, f"/World/Labeled/{label}", False)
            prim = stage.DefinePrim(prim_path, "Xform")
            asset_path = obj_url if obj_url.startswith("omniverse://") else assets_root_path + obj_url
            prim.GetReferences().AddReference(asset_path)
            object_based_sdg_utils.set_transform_attributes(prim, location=rand_loc, rotation=rand_rot)
            object_based_sdg_utils.add_colliders(prim)
            object_based_sdg_utils.add_rigid_body_dynamics(prim, disable_gravity=False)
            add_labels(prim, labels=[label], instance_name="class")
            # zero velocities
            if prim.HasAttribute("physics:velocity"):
                prim.GetAttribute("physics:velocity").Set((0.0, 0.0, 0.0))
            if prim.HasAttribute("physics:angularVelocity"):
                prim.GetAttribute("physics:angularVelocity").Set((0.0, 0.0, 0.0))
            spawned.append(prim)
    labeled_prims = spawned
    return spawned


def wait_for_labeled_assets_to_settle(timeline, max_duration=3.0, lin_thresh=0.02, ang_thresh=1.0, check_interval=10):
    """Simulate until labeled assets rest on the ground plane or timeout."""
    if not labeled_prims and not mesh_distractors:
        return
    if not timeline.is_playing():
        timeline.play()
    elapsed = 0.0
    previous_time = timeline.get_current_time()
    steps = 0
    while elapsed < max_duration:
        simulation_app.update()
        steps += 1
        
        # Only check for settling every 'check_interval' steps to save performance
        if steps % check_interval != 0:
            current_time = timeline.get_current_time()
            elapsed += current_time - previous_time
            previous_time = current_time
            continue

        all_settled = True
        for prim in labeled_prims + mesh_distractors:
            lin_vel = prim.GetAttribute("physics:velocity").Get() if prim.HasAttribute("physics:velocity") else (0, 0, 0)
            ang_vel = (
                prim.GetAttribute("physics:angularVelocity").Get()
                if prim.HasAttribute("physics:angularVelocity")
                else (0, 0, 0)
            )
            if any(abs(v) > lin_thresh for v in lin_vel) or any(abs(v) > ang_thresh for v in ang_vel):
                all_settled = False
                break
        current_time = timeline.get_current_time()
        elapsed += current_time - previous_time
        previous_time = current_time
        if all_settled:
            break



# DISTRACTORS
# Add shape distractors to the environment as floating or falling objects
shape_distractors_types = config.get("shape_distractors_types", ["capsule", "cone", "cylinder", "sphere", "cube"])
shape_distractors_scale_min_max = config.get("shape_distractors_scale_min_max", (0.02, 0.2))
shape_distractors_num = config.get("shape_distractors_num", 350)
shape_distractors = []
floating_shape_distractors = []
falling_shape_distractors = []
for i in range(shape_distractors_num):
    rand_loc, rand_rot, rand_scale = object_based_sdg_utils.get_random_transform_values(
        loc_min=working_area_min, loc_max=working_area_max, scale_min_max=shape_distractors_scale_min_max
    )
    rand_loc = (rand_loc[0], rand_loc[1], OBJECT_Z)
    rand_shape = random.choice(shape_distractors_types)
    prim_path = omni.usd.get_stage_next_free_path(stage, f"/World/Distractors/{rand_shape}", False)
    prim = stage.DefinePrim(prim_path, rand_shape.capitalize())
    object_based_sdg_utils.set_transform_attributes(prim, location=rand_loc, rotation=rand_rot, scale=rand_scale)
    object_based_sdg_utils.add_colliders(prim)
    disable_gravity = random.choice([True, False])
    object_based_sdg_utils.add_rigid_body_dynamics(prim, disable_gravity)
    if disable_gravity:
        floating_shape_distractors.append(prim)
    else:
        falling_shape_distractors.append(prim)
    shape_distractors.append(prim)

# Add mesh distractors to the environment as floating of falling objects
mesh_distactors_urls = config.get("mesh_distractors_urls", [])
mesh_distactors_scale_min_max = config.get("mesh_distractors_scale_min_max", (0.1, 2.0))
mesh_distactors_num = config.get("mesh_distactors_num", 20)
mesh_distractors = []


def clear_mesh_distractors():
    """Remove existing mesh distractors from the stage."""
    global mesh_distractors
    for prim in mesh_distractors:
        stage.RemovePrim(prim.GetPath())
    if stage.GetPrimAtPath("/World/Distractors/Mesh").IsValid():
        stage.RemovePrim("/World/Distractors/Mesh")
    stage.DefinePrim("/World/Distractors/Mesh", "Xform")
    mesh_distractors = []

def spawn_mesh_distractors():
    """Spawn mesh distractors at random XY/height with physics enabled, similar to labeled assets."""
    global mesh_distractors
    clear_mesh_distractors()
    
    # Force spawning at the center to create a pile, matching reference images
    spawn_width = 0.8
    spawn_height = 0.8
    spawn_min_x, spawn_max_x = -spawn_width/2, spawn_width/2
    spawn_min_y, spawn_max_y = -spawn_height/2, spawn_height/2

    for i in range(mesh_distactors_num):
        rand_loc, rand_rot, rand_scale = object_based_sdg_utils.get_random_transform_values(
            loc_min=(spawn_min_x, spawn_min_y, drop_height_min),
            loc_max=(spawn_max_x, spawn_max_y, drop_height_max),
            scale_min_max=mesh_distactors_scale_min_max,
        )
        mesh_url = random.choice(mesh_distactors_urls)
        prim_name_raw = os.path.basename(mesh_url).split(".")[0]
        prim_name = re.sub(r"[^A-Za-z0-9_]", "_", prim_name_raw)
        if not prim_name or not (prim_name[0].isalpha() or prim_name[0] == "_"):
            prim_name = f"mesh_{prim_name}"
        prim_path = omni.usd.get_stage_next_free_path(stage, f"/World/Distractors/Mesh/{prim_name}", False)
        prim = stage.DefinePrim(prim_path, "Xform")
        asset_path = mesh_url if mesh_url.startswith("omniverse://") else assets_root_path + mesh_url
        prim.GetReferences().AddReference(asset_path)
        object_based_sdg_utils.set_transform_attributes(prim, location=rand_loc, rotation=rand_rot, scale=rand_scale)
        object_based_sdg_utils.add_colliders(prim)
        object_based_sdg_utils.add_rigid_body_dynamics(prim, disable_gravity=False)
        # zero velocities
        if prim.HasAttribute("physics:velocity"):
            prim.GetAttribute("physics:velocity").Set((0.0, 0.0, 0.0))
        if prim.HasAttribute("physics:angularVelocity"):
            prim.GetAttribute("physics:angularVelocity").Set((0.0, 0.0, 0.0))
        mesh_distractors.append(prim)
        upgrade_prim_semantics_to_labels(prim, include_descendants=True)
        remove_labels(prim, include_descendants=True)

# REPLICATOR
# Disable capturing every frame (capture will be triggered manually using the step function)
rep.orchestrator.set_capture_on_play(False)

# Create the camera prims and their properties
cameras = []
num_cameras = config.get("num_cameras", 1)
camera_properties_kwargs = config.get("camera_properties_kwargs", {})
for i in range(num_cameras):
    # Create camera and add its properties (focal length, focus distance, f-stop, clipping range, etc.)
    cam_prim = stage.DefinePrim(f"/World/Cameras/cam_{i}", "Camera")
    for key, value in camera_properties_kwargs.items():
        if cam_prim.HasAttribute(key):
            cam_prim.GetAttribute(key).Set(value)
        else:
            print(f"Unknown camera attribute with {key}:{value}")
    cameras.append(cam_prim)

# Add collision spheres (disabled by default) to cameras to avoid objects overlaping with the camera view
camera_colliders = []
camera_collider_radius = config.get("camera_collider_radius", 0)
if camera_collider_radius > 0:
    for cam in cameras:
        cam_path = cam.GetPath()
        cam_collider = stage.DefinePrim(f"{cam_path}/CollisionSphere", "Sphere")
        cam_collider.GetAttribute("radius").Set(camera_collider_radius)
        object_based_sdg_utils.add_colliders(cam_collider)
        collision_api = UsdPhysics.CollisionAPI(cam_collider)
        collision_api.GetCollisionEnabledAttr().Set(False)
        UsdGeom.Imageable(cam_collider).MakeInvisible()
        camera_colliders.append(cam_collider)

# Wait an app update to ensure the prim changes are applied
simulation_app.update()

# Create render products using the cameras
render_products = []
resolution = config.get("resolution", (640, 480))
for cam in cameras:
    rp = rep.create.render_product(cam.GetPath(), resolution)
    render_products.append(rp)

# Enable rendering only at capture time
disable_render_products_between_captures = config.get("disable_render_products_between_captures", True)
if disable_render_products_between_captures:
    object_based_sdg_utils.set_render_products_updates(render_products, False, include_viewport=False)

# Create the writer and attach the render products
writer_type = config.get("writer_type", "PoseWriter")
writer_kwargs = config.get("writer_kwargs", {})
# If not an absolute path, set it relative to the current working directory
if out_dir := writer_kwargs.get("output_dir"):
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(os.getcwd(), out_dir)
        writer_kwargs["output_dir"] = out_dir
    print(f"[SDG] Writing data to: {out_dir}")
if writer_type is not None and len(render_products) > 0:
    writer = rep.writers.get(writer_type)
    writer.initialize(**writer_kwargs)
    writer.attach(render_products)


# Randomize camera poses to look at a random target asset (random distance and center offset)
camera_distance_to_target_min_max = config.get("camera_distance_to_target_min_max", (0.1, 0.5))
camera_look_at_target_offset = config.get("camera_look_at_target_offset", 0.2)


#def randomize_camera_poses():
#for cam in cameras:
        # Get a random target asset to look at
        #target_asset = random.choice(labeled_prims)
        # Add a look_at offset so the target is not always in the center of the camera view
        #loc_offset = (
            #random.uniform(-camera_look_at_target_offset, camera_look_at_target_offset),
            #random.uniform(-camera_look_at_target_offset, camera_look_at_target_offset),
            #random.uniform(-camera_look_at_target_offset, camera_look_at_target_offset),
        #)
        #target_loc = target_asset.GetAttribute("xformOp:translate").Get() + loc_offset
        # Get a random distance to the target asset
        #distance = random.uniform(camera_distance_to_target_min_max[0], camera_distance_to_target_min_max[1])
        # Get a random pose of the camera looking at the target asset from the given distance
        #cam_loc, quat = object_based_sdg_utils.get_random_pose_on_sphere(origin=target_loc, radius=distance)
        #object_based_sdg_utils.set_transform_attributes(cam, location=cam_loc, orientation=quat)


# Temporarily enable camera colliders and simulate for the given number of frames to push out any overlapping objects
def simulate_camera_collision(num_frames=1):
    for cam_collider in camera_colliders:
        collision_api = UsdPhysics.CollisionAPI(cam_collider)
        collision_api.GetCollisionEnabledAttr().Set(True)
    if not timeline.is_playing():
        timeline.play()
    for _ in range(num_frames):
        simulation_app.update()
    for cam_collider in camera_colliders:
        collision_api = UsdPhysics.CollisionAPI(cam_collider)
        collision_api.GetCollisionEnabledAttr().Set(False)


# Create a randomizer for the shape distractors colors, manually triggered at custom events
with rep.trigger.on_custom_event(event_name="randomize_shape_distractor_colors"):
    shape_distractors_paths = [prim.GetPath() for prim in chain(floating_shape_distractors, falling_shape_distractors)]
    shape_distractors_group = rep.create.group(shape_distractors_paths)
    with shape_distractors_group:
        rep.randomizer.color(colors=rep.distribution.uniform((0, 0, 0), (1, 1, 1)))


# Create a randomizer for the lights, manually triggered at custom events
with rep.trigger.on_custom_event(event_name="randomize_lights"):
    # Randomize Distant Light
    distant_light_rep = rep.get.prims(path_pattern="/World/Lights/DistantLight")
    with distant_light_rep:
        # Constrain rotation to keep light coming from above (Y rotation 10-80 degrees, Z rotation 0-360)
        rep.modify.pose(rotation=rep.distribution.uniform((0, 10, 0), (0, 80, 360)))
        rep.randomizer.color(colors=rep.distribution.uniform(LIGHT_SETTINGS["random_light_color_min"], LIGHT_SETTINGS["random_light_color_max"]))
        rep.modify.attribute("inputs:intensity", rep.distribution.normal(LIGHT_SETTINGS["random_light_intensity_mean"], LIGHT_SETTINGS["random_light_intensity_std"]))
        rep.modify.attribute("inputs:colorTemperature", rep.distribution.normal(LIGHT_SETTINGS["random_light_temperature_mean"], LIGHT_SETTINGS["random_light_temperature_std"]))
        # Randomize shadow softness (angle) to match various lighting conditions (sharp vs soft shadows)
        rep.modify.attribute("inputs:angle", rep.distribution.uniform(0.1, 1.5))





# Create a randomizer for the dome background, manually triggered at custom events
dome_textures = [
    assets_root_path + "/NVIDIA/Assets/Skies/Indoor/autoshop_01_4k.hdr",
    assets_root_path + "/NVIDIA/Assets/Skies/Indoor/carpentry_shop_01_4k.hdr",
    assets_root_path + "/NVIDIA/Assets/Skies/Indoor/hotel_room_4k.hdr",
    assets_root_path + "/NVIDIA/Assets/Skies/Indoor/wooden_lounge_4k.hdr",
]


def randomize_dome_background():
    tex = random.choice(dome_textures)
    dome_path = "/World/DomeLight"
    if not stage.GetPrimAtPath(dome_path).IsValid():
        dome_light = UsdLux.DomeLight.Define(stage, dome_path)
    else:
        dome_light = UsdLux.DomeLight(stage.GetPrimAtPath(dome_path))
    dome_light.CreateIntensityAttr().Set(1000.0)
    dome_light.CreateTextureFileAttr().Set(Sdf.AssetPath(tex))
    dome_light.CreateColorAttr().Set((1.0, 1.0, 1.0))
    dome_light.CreateDiffuseAttr().Set(1.0)
    dome_light.CreateSpecularAttr().Set(1.0)
    dome_light.CreateTextureFormatAttr().Set("latlong")
    # optional: random rotation
    xformable = UsdGeom.Xformable(dome_light)
    ops = xformable.GetOrderedXformOps()
    rot_op = None
    for op in ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeRotateZ:
            rot_op = op
            break
    if rot_op is None:
        rot_op = xformable.AddRotateZOp()
    rot_op.Set(random.uniform(0, 360))
    print(f"[SDG] Dome texture -> {tex}")


# Capture motion blur by combining the number of pathtraced subframes samples simulated for the given duration
def capture_with_motion_blur_and_pathtracing(duration=0.05, num_samples=8, spp=64):
    # For small step sizes the physics FPS needs to be temporarily increased to provide movements every syb sample
    orig_physics_fps = physx_scene.GetTimeStepsPerSecondAttr().Get()
    target_physics_fps = 1 / duration * num_samples
    if target_physics_fps > orig_physics_fps:
        print(f"[SDG] Changing physics FPS from {orig_physics_fps} to {target_physics_fps}")
        physx_scene.GetTimeStepsPerSecondAttr().Set(target_physics_fps)

    # Enable motion blur (if not enabled)
    is_motion_blur_enabled = carb.settings.get_settings().get("/omni/replicator/captureMotionBlur")
    if not is_motion_blur_enabled:
        carb.settings.get_settings().set("/omni/replicator/captureMotionBlur", True)
    # Number of sub samples to render for motion blur in PathTracing mode
    carb.settings.get_settings().set("/omni/replicator/pathTracedMotionBlurSubSamples", num_samples)

    # Set the render mode to PathTracing
    prev_render_mode = carb.settings.get_settings().get("/rtx/rendermode")
    carb.settings.get_settings().set("/rtx/rendermode", "PathTracing")
    carb.settings.get_settings().set("/rtx/pathtracing/spp", spp)
    carb.settings.get_settings().set("/rtx/pathtracing/totalSpp", spp)
    carb.settings.get_settings().set("/rtx/pathtracing/optixDenoiser/enabled", 0)

    # Make sure the timeline is playing
    if not timeline.is_playing():
        timeline.play()

    # Capture the frame by advancing the simulation for the given duration and combining the sub samples
    rep.orchestrator.step(delta_time=duration, pause_timeline=False)

    # Restore the original physics FPS
    if target_physics_fps > orig_physics_fps:
        print(f"[SDG] Restoring physics FPS from {target_physics_fps} to {orig_physics_fps}")
        physx_scene.GetTimeStepsPerSecondAttr().Set(orig_physics_fps)

    # Restore the previous render and motion blur  settings
    carb.settings.get_settings().set("/omni/replicator/captureMotionBlur", is_motion_blur_enabled)
    print(f"[SDG] Restoring render mode from 'PathTracing' to '{prev_render_mode}'")
    carb.settings.get_settings().set("/rtx/rendermode", prev_render_mode)


# SDG
# Number of frames to capture
num_frames = config.get("num_frames", 10)

# Increase subframes if materials are not loaded on time, or ghosting artifacts appear on moving objects,
# see: https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/subframes_examples.html
rt_subframes = config.get("rt_subframes", -1)

# Initial trigger for randomizers before the SDG loop with several app updates (ensures materials/textures are loaded)
rep.utils.send_og_event(event_name="randomize_shape_distractor_colors")
randomize_dome_background()
for _ in range(5):
    simulation_app.update()

# Build MDL materials from config for per-frame asset material randomization
asset_mdl_entries = config.get("asset_mdl_materials", [])
asset_mdl_materials = build_mdl_material_library(stage, asset_mdl_entries, prefix="AssetMat")
floor_mdl_entries = config.get("floor_mdl_materials", [])
floor_mdl_materials = build_mdl_material_library(stage, floor_mdl_entries, prefix="FloorMat")

# Set the timeline parameters (start, end, no looping) and start the timeline
timeline = omni.timeline.get_timeline_interface()
timeline.set_start_time(0)
timeline.set_end_time(1000000)
timeline.set_looping(False)
# If no custom physx scene is created, a default one will be created by the physics engine once the timeline starts
timeline.play()
timeline.commit()
simulation_app.update()

# Store the wall start time for stats
wall_time_start = time.perf_counter()

# Set camera once in a fixed top-down pose
# Set camera once in a fixed top-down pose (initial)
update_camera_pose()
# Run the simulation and capture data triggering randomizations and actions at custom frame intervals
for i in range(num_frames):
    print(f"[SDG] Spawning and dropping labeled assets for frame {i}")
    # For roughly half of the frames, limit to at most 5 label types; otherwise spawn all
    max_label_types = 5 if i < (num_frames // 2) else None
    spawn_labeled_assets(max_label_types=max_label_types)
    spawn_mesh_distractors()
    # Randomize MDL materials on labeled assets every frame
    randomize_asset_materials(labeled_prims, asset_mdl_materials)
    randomize_ground_plane_material(floor_mdl_materials)
    # Increase wait time and strictness to ensure objects are not floating
    wait_for_labeled_assets_to_settle(timeline, max_duration=5.0, lin_thresh=0.15, ang_thresh=0.15, check_interval=5)
    update_camera_pose()

    # Randomize lights locations and colors
    if i % 5 == 0:
        print(f"\t Randomizing lights")
        rep.utils.send_og_event(event_name="randomize_lights")

    # Randomize the colors of the primitive shape distractors
    if i % 15 == 0:
        print(f"\t Randomizing shape distractors colors")
        rep.utils.send_og_event(event_name="randomize_shape_distractor_colors")

    # Randomize the texture of the dome background
    if i % 25 == 0:
        print(f"\t Randomizing dome background")
        randomize_dome_background()

    # Randomly disable dome light for 30% of frames to get darker images
    if stage.GetPrimAtPath("/World/DomeLight").IsValid():
        dome_light = UsdLux.DomeLight(stage.GetPrimAtPath("/World/DomeLight"))
        if random.random() < 0.3:
            dome_light.GetIntensityAttr().Set(0.0)
        else:
            dome_light.GetIntensityAttr().Set(1000.0)

    # Enable render products only at capture time
    if disable_render_products_between_captures:
        object_based_sdg_utils.set_render_products_updates(render_products, True, include_viewport=False)

    # Capture the current frame
    print(f"[SDG] Capturing frame {i}/{num_frames}, at simulation time: {timeline.get_current_time():.2f}")
    if i % 5 == 0:
        capture_with_motion_blur_and_pathtracing(duration=0.025, num_samples=4, spp=64)
    else:
        rep.orchestrator.step(delta_time=0.0, rt_subframes=rt_subframes, pause_timeline=False)

    # Disable render products between captures
    if disable_render_products_between_captures:
        object_based_sdg_utils.set_render_products_updates(render_products, False, include_viewport=False)

# Wait for the data to be written (default writer backends are asynchronous)
rep.orchestrator.wait_until_complete()

# Convert COCO output to YOLO format and split train/val if using CocoWriter
yolo_output_dir = config.get("yolo_output_dir")
yolo_split_ratio = config.get("yolo_split_ratio", 0.8)
yolo_split_seed = config.get("yolo_split_seed", 42)
if writer_type == "CocoWriter" and out_dir:
    convert_coco_to_yolo_and_split(out_dir, yolo_output_dir, train_ratio=yolo_split_ratio, seed=yolo_split_seed)

# Get the stats
wall_duration = time.perf_counter() - wall_time_start
sim_duration = timeline.get_current_time()
avg_frame_fps = num_frames / wall_duration
num_captures = num_frames * num_cameras
avg_capture_fps = num_captures / wall_duration
print(
    f"[SDG] Captured {num_frames} frames, {num_captures} entries (frames * cameras) in {wall_duration:.2f} seconds.\n"
    f"\t Simulation duration: {sim_duration:.2f}\n"
    f"\t Average frame FPS: {avg_frame_fps:.2f}\n"
    f"\t Average capture entries (frames * cameras) FPS: {avg_capture_fps:.2f}\n"
)

simulation_app.update()
timeline.stop()

simulation_app.close()
