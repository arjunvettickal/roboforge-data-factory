# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import random
import time
import math
import yaml
import numpy as np
from pathlib import Path

import carb
import carb.settings
from isaacsim import SimulationApp
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# 1. SETUP & CONFIGURATION
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Run the SynthRender SDG pipeline.")
parser.add_argument("--config", required=True, help="Path to the YAML configuration file.")
args, unknown = parser.parse_known_args()

config = {}
if os.path.isfile(args.config):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
else:
    raise FileNotFoundError(f"Config file not found: {args.config}")

print(f"[SynthRender] Loaded config from {args.config}")

# Start Simulation App
launch_config = config.get("launch_config", {"renderer": "RaytracedLighting", "headless": False})
simulation_app = SimulationApp(launch_config=launch_config)

# Import Omniverse/Isaac modules after app start
import omni.replicator.core as rep
import omni.timeline
import omni.usd
import usdrt
from isaacsim.core.utils.semantics import add_labels, remove_labels, upgrade_prim_semantics_to_labels
from isaacsim.storage.native import get_assets_root_path
from pxr import Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, Gf, UsdShade, PhysxSchema

import object_based_sdg_utils

assets_root_path = get_assets_root_path()
stage = omni.usd.get_context().get_stage()

# ---------------------------------------------------------------------------
# 2. SCENE & ENVIRONMENT
# ---------------------------------------------------------------------------
# Setup Physics
physics_scene = UsdPhysics.Scene.Define(stage, "/PhysicsScene")
physx_scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/PhysicsScene"))
physx_scene.GetTimeStepsPerSecondAttr().Set(60)

# Setup Workspace (Floor & Walls)
working_area_size = config.get("working_area_size", (5, 5, 1))
floor_height = -working_area_size[2] / 2.0
object_based_sdg_utils.create_collision_box_walls(
    stage, "/World/CollisionWalls", working_area_size[0], working_area_size[1], working_area_size[2]
)

# Ground Plane
ground_plane_size = 20.0
ground_plane_path = "/World/groundPlane"
ground_plane_mesh = UsdGeom.Mesh.Define(stage, ground_plane_path)
half = ground_plane_size / 2.0
points = [(-half, -half, 0), (half, -half, 0), (half, half, 0), (-half, half, 0)]
ground_plane_mesh.CreatePointsAttr(points)
ground_plane_mesh.CreateFaceVertexCountsAttr([4])
ground_plane_mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
ground_plane_mesh.CreateNormalsAttr([(0, 0, 1)] * 4)
# UVs
uv_scale = 4.0
primvars_api = UsdGeom.PrimvarsAPI(ground_plane_mesh)
primvars_api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set([(0, 0), (uv_scale, 0), (uv_scale, uv_scale), (0, uv_scale)])
object_based_sdg_utils.set_transform_attributes(ground_plane_mesh.GetPrim(), location=(0.0, 0.0, floor_height))
object_based_sdg_utils.add_colliders(ground_plane_mesh.GetPrim(), approximation_shape="none", rest_offset=-0.002)

# ---------------------------------------------------------------------------
# 3. LIGHTING (Standard & Dark Mode)
# ---------------------------------------------------------------------------
light_settings = config.get("light_settings", {})
std_lights = light_settings.get("standard", {})
dark_lights = light_settings.get("dark_mode", {})

# Distant Light (Sun) - On for Standard, Off/Low for Dark Mode
distant_light = stage.DefinePrim("/World/Lights/DistantLight", "DistantLight")
distant_light.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(std_lights.get("distant_intensity", 500.0))
distant_light.CreateAttribute("inputs:angle", Sdf.ValueTypeNames.Float).Set(0.5)

# Dome Light (Environment)
dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
dome_light.CreateIntensityAttr().Set(std_lights.get("dome_intensity", 500.0))

# Spot Light (Dark Mode) - Initially OFF
spot_light = UsdLux.DiskLight.Define(stage, "/World/Lights/SpotLight")
spot_light.CreateIntensityAttr().Set(0.0) 
spot_light.CreateRadiusAttr().Set(dark_lights.get("spot_radius", 3.0))
object_based_sdg_utils.set_transform_attributes(spot_light.GetPrim(), location=(0, 0, 8.0)) # High up
# Shaping API
shaping = UsdLux.ShapingAPI.Apply(spot_light.GetPrim())
shaping.CreateShapingConeAngleAttr().Set(dark_lights.get("spot_cone_angle", 20.0))
shaping.CreateShapingConeSoftnessAttr().Set(0.2)
shaping.CreateShapingFocusAttr().Set(10.0)
# Point down
if not spot_light.GetPrim().HasAttribute("xformOp:rotateXYZ"):
    UsdGeom.Xformable(spot_light).AddRotateXYZOp()
spot_light.GetPrim().GetAttribute("xformOp:rotateXYZ").Set((0,0,0))

def set_lighting_mode(mode):
    """Switches lighting between 'standard' and 'dark_mode'."""
    if mode == "dark_mode":
        # Turn off sun/ambient
        distant_light.GetAttribute("inputs:intensity").Set(0.0)
        dome_light.GetIntensityAttr().Set(10.0) # Very faint ambient
        # Turn on spot
        spot_light.GetIntensityAttr().Set(dark_lights.get("spot_intensity", 60000.0))
    else:
        # Standard
        distant_light.GetAttribute("inputs:intensity").Set(std_lights.get("distant_intensity", 500.0))
        dome_light.GetIntensityAttr().Set(std_lights.get("dome_intensity", 500.0))
        spot_light.GetIntensityAttr().Set(0.0)

# ---------------------------------------------------------------------------
# 4. MATERIALS and ASSET LOADING
# ---------------------------------------------------------------------------
def build_mdl_library(stage, mdl_list, prefix):
    materials = []
    if not mdl_list: return []
    stage.DefinePrim(f"/World/Looks/{prefix}", "Scope")
    for i, entry in enumerate(mdl_list):
        url = entry.get("mdl_url")
        subid = entry.get("subidentifier", "")
        path = f"/World/Looks/{prefix}/Mat_{i}"
        mtl = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, mtl.GetPath().AppendPath("Shader"))
        shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
        shader.SetSourceAsset(url, "mdl")
        shader.SetSourceAssetSubIdentifier(subid, "mdl")
        mtl.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
        materials.append(mtl)
    return materials

floor_materials = build_mdl_library(stage, config.get("materials", {}).get("floor", []), "Floor")
asset_materials = build_mdl_library(stage, config.get("materials", {}).get("assets", []), "Assets")

labeled_assets_config = config.get("labeled_assets", [])
spawned_assets = []
mesh_distractors = []
shape_distractors = []

def clear_assets():
    global spawned_assets, mesh_distractors, shape_distractors
    # Delete previous containers
    for path in ["/World/Labeled", "/World/Distractors/Mesh", "/World/Distractors/Shape"]:
        if stage.GetPrimAtPath(path).IsValid():
            stage.RemovePrim(path)
    spawned_assets = []
    mesh_distractors = []
    shape_distractors = []

def spawn_frame_assets(asset_subset):
    """Spawns valid labeled assets."""
    global spawned_assets
    if not asset_subset: return
    
    container = stage.DefinePrim("/World/Labeled", "Xform")
    spawned = []
    
    # Spawn area
    spawn_min = (-0.5, -0.5, floor_height + 0.8)
    spawn_max = (0.5, 0.5, floor_height + 1.5)

    for item in asset_subset:
        label = item["label"]
        count = item.get("count", 1)
        url = item["url"]
        scale_mm = [1.0, 1.0] # Scaling disabled as per user request

        for _ in range(count):
            path = omni.usd.get_stage_next_free_path(stage, f"/World/Labeled/{label}", False)
            prim = stage.DefinePrim(path, "Xform")
            
            # Reference
            full_url = url if "omniverse://" in url else assets_root_path + url
            prim.GetReferences().AddReference(full_url)
            
            # Random Transform
            loc, rot, scale = object_based_sdg_utils.get_random_transform_values(
                loc_min=spawn_min, loc_max=spawn_max, scale_min_max=scale_mm
            )
            object_based_sdg_utils.set_transform_attributes(prim, location=loc, rotation=rot, scale=scale)
            
            # Physics
            object_based_sdg_utils.add_colliders(prim)
            object_based_sdg_utils.add_rigid_body_dynamics(prim)
            
            # Semantic Label
            add_labels(prim, [label], "class")
            # Apply to children meshes just to be safe
            for child in Usd.PrimRange(prim):
                if child.IsA(UsdGeom.Gprim):
                    add_labels(child, [label], "class")
            
            # Tag for material grouping
            prim.CreateAttribute("user:type_label", Sdf.ValueTypeNames.String).Set(label)
            spawned.append(prim)
            
    spawned_assets = spawned



def apply_materials_logic():
    """
    Applies consistency logic:
    - 3 Handles -> All get same random Material X
    - 2 Clamps -> All get same random Material Y (X != Y ideally)
    
    SPECIAL CASE: 'ballstem'
    - ballstem -> Material A
    - ballstem_001 -> Material B
    """
    if not asset_materials or not spawned_assets: return
    
    # Group by type
    groups = {}
    for prim in spawned_assets:
        lbl = prim.GetAttribute("user:type_label").Get()
        if lbl not in groups: groups[lbl] = []
        groups[lbl].append(prim)
    
    unique_types = list(groups.keys())
    # Shuffle materials
    avail_mats = asset_materials[:]
    random.shuffle(avail_mats)
    
    for i, lbl in enumerate(unique_types):
        if lbl == "ballstem":
             # Special Case: ballstem has 2 parts ("ballstem" and "ballstem_001")
             # We want 2 different materials for these parts.
            
             # Pick first material
             mat1 = avail_mats[i % len(avail_mats)]
            
             # Pick second material (ensure it's different)
             offset = 1
             mat2 = avail_mats[(i + offset) % len(avail_mats)]
             while mat2.GetPath() == mat1.GetPath() and len(avail_mats) > 1:
                 offset += 1
                 if offset >= len(avail_mats): break
                 mat2 = avail_mats[(i + offset) % len(avail_mats)]
             
             for prim in groups[lbl]:
                 parts_found = 0
                 # Bind specific parts
                 for descendant in Usd.PrimRange(prim):
                     name = descendant.GetName()
                     if name == "ballstem":
                         UsdShade.MaterialBindingAPI(descendant).Bind(mat1)
                         parts_found += 1
                     elif name == "ballstem_001":
                         UsdShade.MaterialBindingAPI(descendant).Bind(mat2)
                         parts_found += 1
                 
                 if parts_found == 0:
                     # Fallback if specific parts not found (maybe different hierarchy)
                     UsdShade.MaterialBindingAPI(prim).Bind(mat1)
                     for desc in Usd.PrimRange(prim):
                         if desc.IsA(UsdGeom.Gprim):
                             UsdShade.MaterialBindingAPI(desc).Bind(mat1)

        else:
            # Standard logic for other assets
            mat = avail_mats[i % len(avail_mats)] # Wrap if not enough materials
            for prim in groups[lbl]:
                UsdShade.MaterialBindingAPI(prim).Bind(mat)
                # Bind to all children meshes too for safety
                for desc in Usd.PrimRange(prim):
                    if desc.IsA(UsdGeom.Gprim):
                        UsdShade.MaterialBindingAPI(desc).Bind(mat)
    
    print(f"[SynthRender] Applied materials to {len(unique_types)} distinct asset types.")

def randomize_floor():
    if not floor_materials: return
    mat = random.choice(floor_materials)
    UsdShade.MaterialBindingAPI(ground_plane_mesh.GetPrim()).Bind(mat)

def spawn_distractors():
    """Spawns mesh and shape distractors."""
    global mesh_distractors, shape_distractors
    # Mesh Distractors
    d_conf = config.get("distractors", {})
    mesh_urls = d_conf.get("mesh_distractors_urls", [])
    if mesh_urls:
        n_mesh = d_conf.get("mesh_distractors_num", 5)
        container = stage.DefinePrim("/World/Distractors/Mesh", "Xform")
        for _ in range(n_mesh):
            url = random.choice(mesh_urls)
            name = url.split("/")[-1].split(".")[0]
            # Verify valid USD identifier
            if not name or not (name[0].isalpha() or name.startswith("_")):
                name = f"mesh_{name}"
            # Replace invalid characters (like dashes) with underscores
            name = name.replace("-", "_").replace(" ", "_")
            
            path = omni.usd.get_stage_next_free_path(stage, f"/World/Distractors/Mesh/{name}", False)
            prim = stage.DefinePrim(path, "Xform")
            full_url = url if "omniverse://" in url else assets_root_path + url
            prim.GetReferences().AddReference(full_url)
            
            loc, rot, scl = object_based_sdg_utils.get_random_transform_values(
                loc_min=(-1, -1, floor_height+1), loc_max=(1, 1, floor_height+2), 
                scale_min_max=d_conf.get("mesh_distractors_scale_min_max", [0.5, 1])
            )
            object_based_sdg_utils.set_transform_attributes(prim, location=loc, rotation=rot, scale=scl)
            object_based_sdg_utils.add_colliders(prim)
            object_based_sdg_utils.add_rigid_body_dynamics(prim)
            mesh_distractors.append(prim)

    # Shape Distractors (Simple)
    n_shape = d_conf.get("shape_distractors_num", 2)
    shapes = ["cube", "sphere", "cone"]
    container_s = stage.DefinePrim("/World/Distractors/Shape", "Xform")
    for _ in range(n_shape):
         shp = random.choice(shapes)
         path = omni.usd.get_stage_next_free_path(stage, f"/World/Distractors/Shape/{shp}", False)
         prim = stage.DefinePrim(path, shp.capitalize())
         loc, rot, scl = object_based_sdg_utils.get_random_transform_values(
             loc_min=(-1.5, -1.5, floor_height+0.5), loc_max=(1.5, 1.5, floor_height+1),
             scale_min_max=d_conf.get("shape_distractors_scale_min_max", [0.05, 0.15])
         )
         object_based_sdg_utils.set_transform_attributes(prim, location=loc, rotation=rot, scale=scl)
         object_based_sdg_utils.add_colliders(prim)
         object_based_sdg_utils.add_rigid_body_dynamics(prim)
         
         # Random Color
         color = (random.random(), random.random(), random.random())
         if prim.IsA(UsdGeom.Imageable):
             UsdGeom.Gprim(prim).CreateDisplayColorAttr().Set([color])
             
         shape_distractors.append(prim)

# ---------------------------------------------------------------------------
# 5. CAMERA & WRITER
# ---------------------------------------------------------------------------
rep.orchestrator.set_capture_on_play(False) # Prevent capturing during physics settling

camera = stage.DefinePrim("/World/Camera", "Camera")
rep_res = config.get("resolution", [720, 720])
rp = rep.create.render_product(camera.GetPath(), rep_res)

writer_conf = config.get("writer_type", "CocoWriter")
kwargs = config.get("writer_kwargs", {})
out_dir = kwargs.get("output_dir", "output")
if not os.path.isabs(out_dir): kwargs["output_dir"] = os.path.join(os.getcwd(), out_dir)
print(f"[SynthRender] Output Directory: {kwargs['output_dir']}")

writer = rep.writers.get(writer_conf)
writer.initialize(**kwargs)
print(f"[SynthRender] Created Render Product: {rp}")
writer.attach([rp])

def update_camera():
    # 4 Corner logic
    dist_min, dist_max = config.get("input_camera", {}).get("distance_min_max", [10, 16])
    corner_angles = [45, 135, 225, 315]
    angle_base = random.choice(corner_angles)
    angle_rad = math.radians(angle_base + random.uniform(-15, 15))
    elev_rad = math.radians(random.uniform(60, 80)) # high angle
    dist = random.uniform(dist_min, dist_max)
    
    x = dist * math.cos(angle_rad) * math.cos(elev_rad)
    y = dist * math.sin(angle_rad) * math.cos(elev_rad)
    z = dist * math.sin(elev_rad)
    
    eye = Gf.Vec3d(x, y, z)
    target = Gf.Vec3d(0, 0, 0)
    up = Gf.Vec3d(0, 0, 1)
    m = Gf.Matrix4d().SetLookAt(eye, target, up).GetInverse()
    q = m.ExtractRotation().GetQuat()
    
    object_based_sdg_utils.set_transform_attributes(camera, location=(x,y,z), orientation=Gf.Quatf(q))

# ---------------------------------------------------------------------------
# 6. MAIN LOOP
# ---------------------------------------------------------------------------
timeline = omni.timeline.get_timeline_interface()
timeline.play()

probs = config.get("probabilities", {})
p_ground = probs.get("ground_plane_only", 0.08)
p_dark = probs.get("dark_mode", 0.25)
# p_standard = remainder

num_frames = config.get("num_frames", 100)
min_assets = config.get("assets", {}).get("min_types_per_frame", 3)
max_assets = config.get("assets", {}).get("max_types_per_frame", 8)

for i in range(num_frames):
    print(f"--- Frame {i+1}/{num_frames} ---")
    clear_assets()
    
    # 1. Determine Frame Type
    rnd = random.random()
    if rnd < p_ground:
        frame_type = "GROUND_ONLY"
    elif rnd < (p_ground + p_dark):
        frame_type = "DARK_MODE"
    else:
        frame_type = "STANDARD"
        
    print(f"    Type: {frame_type}")
    
    # 2. Setup Lighting
    if frame_type == "DARK_MODE":
        set_lighting_mode("dark_mode")
    else:
        set_lighting_mode("standard")
        
    # 3. Spawn Assets (unless Ground Only)
    if frame_type != "GROUND_ONLY":
        # Pick random subset of assets
        # For STANDARD mode, 20% chance to have ALL assets
        if frame_type == "STANDARD" and random.random() < 0.20:
             subset = labeled_assets_config # All of them
        else:
             num_types = random.randint(min_assets, min(len(labeled_assets_config), max_assets))
             subset = random.sample(labeled_assets_config, num_types)
        spawn_frame_assets(subset)
    
    # 4. Spawn Distractors (Always)
    spawn_distractors()
    
    # 5. Apply Materials
    randomize_floor()
    apply_materials_logic()
    
    # 6. Move Camera
    update_camera()
    
    # 7. Settle Physics
    for _ in range(120): # Increased settling time
        simulation_app.update()
        
    # 8. Capture
    rep.orchestrator.step(delta_time=0.1, pause_timeline=False) # Trigger writer with time advance
    
print("[SynthRender] DONE.")

# ---------------------------------------------------------------------------
# YOLO CONVERSION
# ---------------------------------------------------------------------------
rep.orchestrator.wait_until_complete()

def convert_coco_to_yolo_and_split(coco_root_path, yolo_root_path=None, train_ratio=0.8, seed=None):
    """Converts the generated COCO data to YOLO format."""
    
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
    
    # Sort categories to ensure consistent class IDs
    categories.sort(key=lambda x: x.get("id", 0))
    
    if not images or not annotations or not categories:
        print(f"[YOLO-CONVERT] Empty COCO annotations at {ann_path}, skipping.")
        return

    cat_id_to_yolo = {cat["id"]: idx for idx, cat in enumerate(categories)}
    yolo_id_to_name = [cat.get("name", f"class_{i}") for i, cat in enumerate(categories)]

    ann_by_image = {}
    for ann in annotations:
        img_id = ann.get("image_id")
        ann_by_image.setdefault(img_id, []).append(ann)

    # Setup directories
    yolo_root = Path(yolo_root_path) if yolo_root_path else coco_root
    img_train_dir = yolo_root / "images" / "train"
    img_val_dir = yolo_root / "images" / "val"
    lbl_train_dir = yolo_root / "labels" / "train"
    lbl_val_dir = yolo_root / "labels" / "val"
    for d in [img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Random split
    rng = random.Random(seed)
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

    # Create test directories
    img_test_dir = yolo_root / "images" / "test"
    lbl_test_dir = yolo_root / "labels" / "test"
    img_test_dir.mkdir(parents=True, exist_ok=True)
    lbl_test_dir.mkdir(parents=True, exist_ok=True)

    # Create data.yaml
    data_yaml = yolo_root / "data.yaml"
    names_lines = "\n".join([f"  {i}: {n}" for i, n in enumerate(yolo_id_to_name)])
    data_yaml.write_text(
        f"path: .\ntrain: images/train\nval: images/val\ntest: images/test\n# Number of classes\nnc: {len(yolo_id_to_name)}\nnames:\n{names_lines}\n"
    )
    print(f"[YOLO-CONVERT] YOLO dataset at: {yolo_root}")

# Run conversion if needed
writer_type = writer_conf # Map existing variable
yolo_output_dir = config.get("yolo_output_dir")
yolo_split_ratio = config.get("yolo_split_ratio", 0.8)
yolo_split_seed = config.get("yolo_split_seed")
if writer_type == "CocoWriter" and out_dir:
    convert_coco_to_yolo_and_split(out_dir, yolo_output_dir, train_ratio=yolo_split_ratio, seed=yolo_split_seed)

simulation_app.close()
