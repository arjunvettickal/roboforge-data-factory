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

import yaml
from isaacsim import SimulationApp

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




floor_urls = config.get("floors", [])

if not floor_urls:
    print("[WARN] No floors found in config['floors']")
else:
    print("[INFO] Loaded floors from config:")
    for url in floor_urls:
        print("   ", url)

simulation_app = SimulationApp(launch_config=launch_config)

import random
import time
from itertools import chain
import math
import colorsys

import carb.settings

# Custom util functions for the example
import object_based_sdg_utils
import omni.replicator.core as rep
import omni.timeline
import omni.usd
import usdrt
from isaacsim.core.utils.semantics import add_labels, remove_labels, upgrade_prim_semantics_to_labels
from isaacsim.storage.native import get_assets_root_path
from omni.physx import get_physx_interface, get_physx_scene_query_interface
from pxr import PhysxSchema, Sdf, UsdGeom, UsdPhysics, Usd, Gf, UsdShade



# Isaac nucleus assets root path
assets_root_path = get_assets_root_path()
stage = None

# -------------------------------------------------------------------
# Small helpers for pose randomization (no physics)
# -------------------------------------------------------------------

# bbox_cache and labeled_radii will be initialized later, after assets are created
bbox_cache = None
labeled_radii = {}

labeled_yaw_offsets = {}
ROTATION_SPEED_DEG_PER_FRAME = 26.0  # tweak speed as you like
RADIUS_SCALE = 1.2      # was 1.1 inside the function
LABEL_MARGIN = 0.05     # extra gap between objects in meters
MAX_PLACEMENT_TRIES = 100


TINT_STRENGTH = 1.0  # 0 = no tint (pure white), 1 = full crazy color

def sample_light_color():
    """Return a soft-tinted (r, g, b) in 0–1 range."""
    # Still random hue
    h = random.random()

    # Lower saturation & slightly dimmer value than before
    s = random.uniform(0.4, 0.7)   # was 0.6–1.0
    v = random.uniform(0.7, 0.9)   # was 0.8–1.0

    r, g, b = colorsys.hsv_to_rgb(h, s, v)

    # Mix with white so the tint is not overwhelming
    t = TINT_STRENGTH
    r = 1.0 * (1 - t) + r * t
    g = 1.0 * (1 - t) + g * t
    b = 1.0 * (1 - t) + b * t

    return r, g, b




def sample_positions_for_prims(prims, radii, min_x, max_x, min_y, max_y, margin=LABEL_MARGIN, max_tries=MAX_PLACEMENT_TRIES):
    """Sample 2D positions so that discs (radii per prim) don't overlap.

    If a prim has no precomputed radius, compute it once and cache it.
    """
    positions = {}
    for prim in prims:
        # get radius for this prim; compute if missing
        r = radii.get(prim)
        if r is None:
            bbox = bbox_cache.ComputeWorldBound(prim)
            size = bbox.GetRange().GetSize()
            r = 0.5 * max(size[0], size[1]) * RADIUS_SCALE
            radii[prim] = r

        for _ in range(max_tries):
            x = random.uniform(min_x + r, max_x - r)
            y = random.uniform(min_y + r, max_y - r)
            if all(
                (x - px) ** 2 + (y - py) ** 2 >= (r + radii[other] + margin) ** 2
                for other, (px, py) in positions.items()
            ):
                positions[prim] = (x, y)
                break
        else:
            # fall back: put it somewhere, even if overlap
            print(f"[WARN] Forcing overlapped placement for {prim.GetPath()}")
            positions[prim] = (
                random.uniform(min_x, max_x),
                random.uniform(min_y, max_y),
            )
    return positions


def randomize_labeled_poses(labeled_prims, labeled_radii, min_x, max_x, min_y, max_y, object_z, frame_idx):
    """Randomize positions & *continuous* yaw using per-prim radii to avoid overlap."""
    if not labeled_prims:
        return

    positions = sample_positions_for_prims(
        prims=labeled_prims,
        radii=labeled_radii,
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
    )

    for prim, (x, y) in positions.items():
        # base offset per asset
        base_yaw = labeled_yaw_offsets.get(prim, 0.0)
        # advance with frame index – continuous spin
        yaw = (base_yaw + frame_idx * ROTATION_SPEED_DEG_PER_FRAME) % 360.0

        object_based_sdg_utils.set_transform_attributes(
            prim,
            location=(x, y, object_z),
            rotation=(yaw, 0.0, yaw),  # Z-axis yaw (assuming Z-up in your scene)
        )



def randomize_distractor_poses(prims, min_x, max_x, min_y, max_y, object_z):
    """Randomize distractors' positions & yaw."""
    if not prims:
        return

    for prim in prims:
        x = random.uniform(min_x, max_x)
        y = random.uniform(min_y, max_y)
        yaw = random.uniform(0.0, 360.0)

        object_based_sdg_utils.set_transform_attributes(
            prim,
            location=(x, y, object_z),
            rotation=(0.0, 0.0, yaw),
        )


# ENVIRONMENT
env_url = config.get("env_url", "")
if env_url:
    env_path = env_url if env_url.startswith("omniverse://") else assets_root_path + env_url
    omni.usd.get_context().open_stage(env_path)
    stage = omni.usd.get_context().get_stage()
else:
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()

# ----------------------------------------------------------------------
# MDL material helpers for labeled assets
# ----------------------------------------------------------------------

def build_mdl_material_library(stage, mdl_entries):
    """
    Given a list of dicts:
        [{ "mdl_url": "...", "subidentifier": "..." }, ...]
    create UsdShade.Materials under /World/Looks and return them.
    """
    if not mdl_entries:
        return []

    # Just a container prim for neatness
    stage.DefinePrim("/World/Looks", "Scope")

    materials = []

    for idx, entry in enumerate(mdl_entries):
        mdl_url = entry.get("mdl_url")
        subid = entry.get("subidentifier")

        if not mdl_url:
            continue
        if not subid:
            # Fallback: guess from filename (Copper_Brushed.mdl -> Copper_Brushed)
            subid = mdl_url.split("/")[-1].replace(".mdl", "")

        mtl_path = Sdf.Path(f"/World/Looks/AssetMat_{idx}")
        mtl = UsdShade.Material.Define(stage, mtl_path)

        # Shader that reads from MDL
        shader = UsdShade.Shader.Define(stage, mtl_path.AppendPath("Shader"))
        shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
        # MDL shaders use sourceType 'mdl'
        shader.SetSourceAsset(mdl_url, "mdl")
        shader.SetSourceAssetSubIdentifier(subid, "mdl")

        # MDL materials use renderContext 'mdl'
        surf_out = mtl.CreateSurfaceOutput("mdl")
        surf_out.ConnectToSource(shader.ConnectableAPI(), "out")

        disp_out = mtl.CreateDisplacementOutput("mdl")
        disp_out.ConnectToSource(shader.ConnectableAPI(), "out")

        vol_out = mtl.CreateVolumeOutput("mdl")
        vol_out.ConnectToSource(shader.ConnectableAPI(), "out")

        materials.append(mtl)

    return materials


def randomize_asset_materials(prims, materials):
    """
    Randomly bind one of the given materials to each prim in 'prims'.
    """
    if not materials:
        return

    for prim in prims:
        mat = random.choice(materials)
        UsdShade.MaterialBindingAPI(prim).Bind(mat)



# -------------------------------------------------------
# FIXED CAMERA SETUP (top-down-ish, with randomization)
# -------------------------------------------------------

# These are the bounds for the random camera distance
CAMERA_DISTANCE_MIN = 3.0
CAMERA_DISTANCE_MAX = 6.0

# This will be updated every frame before capture
CAMERA_DISTANCE_ABOVE_OBJECTS = CAMERA_DISTANCE_MIN



def set_fixed_topdown_camera():
    cam_path = "/World/Cameras/cam_0"
    cam_prim = stage.GetPrimAtPath(cam_path)

    if not cam_prim.IsValid():
        print(f"[WARN] Camera prim {cam_path} not found")
        return

    cam_loc = (0.0, 0.0, OBJECT_Z + CAMERA_DISTANCE_ABOVE_OBJECTS)
    cam_rot = (0.0, 0.0, 0.0)  # in your scene this already looks "down" at the objects

    object_based_sdg_utils.set_transform_attributes(
        cam_prim,
        location=cam_loc,
        rotation=cam_rot
    )

    print(f"[INFO] Fixed camera set at {cam_loc}, rot={cam_rot}")


# Add a distant light to the empty stage
distant_light = stage.DefinePrim("/World/Lights/DistantLight", "DistantLight")
distant_light.CreateAttribute("inputs:angle", Sdf.ValueTypeNames.Float).Set(7.0)  # try 2–5
distant_light.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(5000.0)
if not distant_light.HasAttribute("xformOp:rotateXYZ"):
    UsdGeom.Xformable(distant_light).AddRotateXYZOp()
distant_light.GetAttribute("xformOp:rotateXYZ").Set((0, 60, 0))
# Second distant light – softer "fill sun" from a slightly different direction
distant_light2 = stage.DefinePrim("/World/Lights/DistantLight2", "DistantLight")

# Base intensity; will be overridden per-frame, but good to have a default
distant_light2.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(3000.0)

# Give it its own direction (a bit lower and from another side)
if not distant_light2.HasAttribute("xformOp:rotateXYZ"):
    UsdGeom.Xformable(distant_light2).AddRotateXYZOp()
distant_light2.GetAttribute("xformOp:rotateXYZ").Set((-15, -30, 0))

# Slightly larger angle -> softer, blurrier shadows
distant_light2.CreateAttribute("inputs:angle", Sdf.ValueTypeNames.Float).Set(10.0)


# Get the working area size and bounds (width=x, depth=y, height=z)
working_area_size = config.get("working_area_size", (3, 3, 3))

# Horizontal extents stay the same
min_x = -working_area_size[0] / 2.0
max_x = working_area_size[0] / 2.0
min_y = -working_area_size[1] / 2.0
max_y = working_area_size[1] / 2.0

# Floor is at the very bottom of the working area
floor_height = -working_area_size[2] / 2.0

# objects will float slightly above the floor – you tuned this value
OBJECT_Z = floor_height + 2.0

# NEW: shape distractors sit lower than labeled assets
SHAPE_DISTRACTOR_Z_OFFSET = -0.4   # 40 cm below OBJECT_Z (tweak as you like)

floor_material_urls = config.get("floor_material_urls", [])

# We only spawn objects in a band above the floor (used for initial placement)
spawn_min_z = floor_height + 0.80
spawn_max_z = floor_height + 1.50

working_area_min = (min_x, min_y, spawn_min_z)
working_area_max = (max_x, max_y, spawn_max_z)

# Create a collision box area around the assets (kept, but no physics motion now)
object_based_sdg_utils.create_collision_box_walls(
    stage, "/World/CollisionWalls", working_area_size[0], working_area_size[1], working_area_size[2]
)

# ---- USD FLOOR FROM CONFIG -------------------------------
if not floor_urls:
    raise RuntimeError("No floors found in config['floors'].")

# Create the floor prim once
floor_prim = stage.DefinePrim("/World/Floor", "Xform")
xform = UsdGeom.Xformable(floor_prim)

# Position it like the old cube floor
xform.AddTranslateOp().Set((0.0, 0.0, floor_height - 0.025))

# Get a reference proxy so we can swap floors later
floor_refs = floor_prim.GetReferences()

# Initialize with the first floor, just so the stage isn't empty
initial_floor = floor_urls[0]
print("[INFO] Initial floor:", initial_floor)
floor_refs.ClearReferences()
floor_refs.AddReference(initial_floor)
# ----------------------------------------------------------


# Create a physics scene (mostly unused now, but harmless)
usdrt_stage = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())
physics_scenes = usdrt_stage.GetPrimsWithAppliedAPIName("PhysxSceneAPI")
if physics_scenes:
    physics_scene = physics_scenes[0]
else:
    physics_scene = UsdPhysics.Scene.Define(stage, "/PhysicsScene")
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/PhysicsScene"))
physx_scene.GetTimeStepsPerSecondAttr().Set(60)

# TRAINING ASSETS
# Add the objects to be trained in the environment with their labels and properties
labeled_assets_and_properties = config.get("labeled_assets_and_properties", [])
floating_labeled_prims = []
falling_labeled_prims = []
labeled_prims = []

# --- build a grid of slots across the working area for initial placement ---
total_labeled = sum(obj.get("count", 1) for obj in labeled_assets_and_properties)

if total_labeled > 0:
    grid_cols = math.ceil(math.sqrt(total_labeled))
    grid_rows = math.ceil(total_labeled / grid_cols)

    gx_min, gy_min, gz_min = working_area_min
    gx_max, gy_max, gz_max = working_area_max

    step_x = (gx_max - gx_min) / grid_cols
    step_y = (gy_max - gy_min) / grid_rows

    slot_idx = 0

    for obj in labeled_assets_and_properties:
        obj_url = obj.get("url", "")
        label = obj.get("label", "unknown")
        count = obj.get("count", 1)
        floating = obj.get("floating", False)
        scale_min_max = obj.get("scale_min_max", (1, 1))

        for _ in range(count):
            row = slot_idx // grid_cols
            col = slot_idx % grid_cols
            slot_idx += 1

            base_x = gx_min + (col + 0.5) * step_x
            base_y = gy_min + (row + 0.5) * step_y

            jitter_x = random.uniform(-0.3, 0.3) * step_x
            jitter_y = random.uniform(-0.3, 0.3) * step_y

            rand_loc, rand_rot, rand_scale = object_based_sdg_utils.get_random_transform_values(
                loc_min=working_area_min, loc_max=working_area_max, scale_min_max=scale_min_max
            )
            rand_loc = (base_x + jitter_x, base_y + jitter_y, OBJECT_Z)

            prim_path = omni.usd.get_stage_next_free_path(stage, f"/World/Labeled/{label}", False)
            prim = stage.DefinePrim(prim_path, "Xform")
            asset_path = obj_url if obj_url.startswith("omniverse://") else assets_root_path + obj_url
            prim.GetReferences().AddReference(asset_path)
            object_based_sdg_utils.set_transform_attributes(
                prim, location=rand_loc, rotation=rand_rot, scale=rand_scale
            )
            object_based_sdg_utils.add_colliders(prim)  # no rigid body: we move via transforms only
            add_labels(prim, labels=[label], instance_name="class")

            if floating:
                floating_labeled_prims.append(prim)
            else:
                falling_labeled_prims.append(prim)

labeled_prims = floating_labeled_prims + falling_labeled_prims

# Initialize per-prim yaw offsets so each asset rotates differently
for idx, prim in enumerate(labeled_prims):
    # Spread starting angles around the circle
    labeled_yaw_offsets[prim] = (idx * (360.0 / max(1, len(labeled_prims)))) % 360.0


# After all labeled_prims exist, we can set up the bbox cache
bbox_cache = UsdGeom.BBoxCache(
    Usd.TimeCode.Default(),
    [UsdGeom.Tokens.default_],
)

# DISTRACTORS
# Add shape distractors to the environment (static, pose-randomized each frame)
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
    rand_loc = (rand_loc[0], rand_loc[1], OBJECT_Z + SHAPE_DISTRACTOR_Z_OFFSET)
    rand_shape = random.choice(shape_distractors_types)
    prim_path = omni.usd.get_stage_next_free_path(stage, f"/World/Distractors/{rand_shape}", False)
    prim = stage.DefinePrim(prim_path, rand_shape.capitalize())
    object_based_sdg_utils.set_transform_attributes(prim, location=rand_loc, rotation=rand_rot, scale=rand_scale)
    object_based_sdg_utils.add_colliders(prim)
    shape_distractors.append(prim)

# Add mesh distractors to the environment
mesh_distactors_urls = config.get("mesh_distractors_urls", [])
mesh_distactors_scale_min_max = config.get("mesh_distractors_scale_min_max", (0.1, 2.0))
mesh_distactors_num = config.get("mesh_distractors_num", 10)
mesh_distractors = []
floating_mesh_distractors = []
falling_mesh_distractors = []
for i in range(mesh_distactors_num):
    rand_loc, rand_rot, rand_scale = object_based_sdg_utils.get_random_transform_values(
        loc_min=working_area_min, loc_max=working_area_max, scale_min_max=mesh_distactors_scale_min_max
    )
    rand_loc = (rand_loc[0], rand_loc[1], OBJECT_Z)
    mesh_url = random.choice(mesh_distactors_urls)
    prim_name = os.path.basename(mesh_url).split(".")[0]
    prim_path = omni.usd.get_stage_next_free_path(stage, f"/World/Distractors/{prim_name}", False)
    prim = stage.DefinePrim(prim_path, "Xform")
    asset_path = mesh_url if mesh_url.startswith("omniverse://") else assets_root_path + mesh_url
    prim.GetReferences().AddReference(asset_path)
    object_based_sdg_utils.set_transform_attributes(prim, location=rand_loc, rotation=rand_rot, scale=rand_scale)
    object_based_sdg_utils.add_colliders(prim)
    mesh_distractors.append(prim)
    # Optional: clear semantics on distractors
    # upgrade_prim_semantics_to_labels(prim, include_descendants=True)
    # remove_labels(prim, include_descendants=True)

# REPLICATOR
# Disable capturing every frame (capture will be triggered manually using the step function)
rep.orchestrator.set_capture_on_play(False)

# Create the camera prims and their properties
cameras = []
num_cameras = config.get("num_cameras", 1)
camera_properties_kwargs = config.get("camera_properties_kwargs", {})
for i in range(num_cameras):
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
if out_dir := writer_kwargs.get("output_dir"):
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(os.getcwd(), out_dir)
        writer_kwargs["output_dir"] = out_dir
    print(f"[SDG] Writing data to: {out_dir}")
if writer_type is not None and len(render_products) > 0:
    writer = rep.writers.get(writer_type)
    writer.initialize(**writer_kwargs)
    writer.attach(render_products)

# ----------------------------- RANDOMIZERS -----------------------------

# Area to check for overlapping objects (kept from original, but mostly unused without physics)
overlap_area_thickness = 0.1
overlap_area_origin = (0, 0, (-working_area_size[2] / 2) + (overlap_area_thickness / 2))
overlap_area_extent = (
    working_area_size[0] / 2 * 0.99,
    working_area_size[1] / 2 * 0.99,
    overlap_area_thickness / 2 * 0.99,
)


def on_overlap_hit(hit):
    # kept for completeness, but no rigid bodies now so no hits expected
    return True


def on_physics_step(dt: float):
    get_physx_scene_query_interface().overlap_box(
        carb.Float3(overlap_area_extent),
        carb.Float3(overlap_area_origin),
        carb.Float4(0, 0, 0, 1),
        on_overlap_hit,
        False,
    )


physx_sub = get_physx_interface().subscribe_physics_step_events(on_physics_step)

# Create a randomizer for the shape distractors colors
with rep.trigger.on_custom_event(event_name="randomize_shape_distractor_colors"):
    shape_distractors_paths = [prim.GetPath() for prim in shape_distractors]
    shape_distractors_group = rep.create.group(shape_distractors_paths)
    with shape_distractors_group:
        rep.randomizer.color(colors=rep.distribution.uniform((0, 0, 0), (1, 1, 1)))

# Create a randomizer for lights in the working area
light_min = (min_x, min_y, OBJECT_Z + CAMERA_DISTANCE_ABOVE_OBJECTS + 1.0)
light_max = (max_x, max_y, OBJECT_Z + CAMERA_DISTANCE_ABOVE_OBJECTS + 1.0)

with rep.trigger.on_custom_event(event_name="randomize_lights"):
    lights = rep.create.light(
        light_type="Sphere",
        color=rep.distribution.uniform((0.6, 0.6, 0.6), (1.0, 1.0, 1.0)),
        temperature=rep.distribution.normal(6500, 500),
        intensity=rep.distribution.normal(12000, 3000),
        # lights always behind the camera, never intersecting the floor
        position=rep.distribution.uniform(light_min, light_max),
        scale=rep.distribution.uniform(0.1, 0.4),
        count=3,
    )

# Create a randomizer for the dome background
#with rep.trigger.on_custom_event(event_name="randomize_dome_background"):
#    dome_textures = [
#        assets_root_path + "/NVIDIA/Assets/Skies/Indoor/autoshop_01_4k.hdr",
#        assets_root_path + "/NVIDIA/Assets/Skies/Indoor/carpentry_shop_01_4k.hdr",
#        assets_root_path + "/NVIDIA/Assets/Skies/Indoor/hotel_room_4k.hdr",
#        assets_root_path + "/NVIDIA/Assets/Skies/Indoor/wooden_lounge_4k.hdr",
#    ]
#    dome_light = rep.create.light(light_type="Dome")
#    with dome_light:
#        rep.modify.attribute("inputs:texture:file", rep.distribution.choice(dome_textures))
#        rep.randomizer.rotation()

# ----------------------------- SDG LOOP -----------------------------

num_frames = config.get("num_frames", 10)
rt_subframes = config.get("rt_subframes", -1)

# Initial triggers so textures/materials are ready
rep.utils.send_og_event(event_name="randomize_shape_distractor_colors")
#rep.utils.send_og_event(event_name="randomize_dome_background")
for _ in range(5):
    simulation_app.update()
if floor_material_urls:
    rep.utils.send_og_event(event_name="randomize_floor_material")
    

# Timeline still used by Replicator
timeline = omni.timeline.get_timeline_interface()
timeline.set_start_time(0)
timeline.set_end_time(1000000)
timeline.set_looping(False)
timeline.play()
timeline.commit()
simulation_app.update()

# Set camera once
wall_time_start = time.perf_counter()


asset_mdl_entries = config.get("asset_mdl_materials", [])
asset_mdl_materials = build_mdl_material_library(stage, asset_mdl_entries)


# Main capture loop – NO PHYSICS MOTION, only pose + lighting randomization
for i in range(num_frames):
    
    # Randomize camera distance for this frame
    CAMERA_DISTANCE_ABOVE_OBJECTS = random.uniform(CAMERA_DISTANCE_MIN, CAMERA_DISTANCE_MAX)
    set_fixed_topdown_camera()

    # ---- Randomize floor USD ----
    if floor_urls:
        selected_floor = random.choice(floor_urls)
        print(f"[INFO] Frame {i:04d} using floor: {selected_floor}")
        floor_refs.ClearReferences()
        floor_refs.AddReference(selected_floor)
    # ------------------------------
    # ---- Intensities ----
    main_intensity = random.uniform(600.0, 1500.0)
    distant_light.GetAttribute("inputs:intensity").Set(main_intensity)

    fill_ratio = random.uniform(0.25, 0.5)
    fill_intensity = main_intensity * fill_ratio
    distant_light2.GetAttribute("inputs:intensity").Set(fill_intensity)

    # ---- Colors ----
    # One overall mood color per frame (for “apple-to-apple” with Blender)
    r, g, b = sample_light_color()
    color_vec = Gf.Vec3f(r, g, b)

    # Main sun uses the color
    distant_light.GetAttribute("inputs:color").Set(color_vec)

    # Fill sun: either same color or slightly washed out
    desat = random.uniform(0.2, 0.5)
    rf = r + (1.0 - r) * desat
    gf = g + (1.0 - g) * desat
    bf = b + (1.0 - b) * desat

    fill_color = Gf.Vec3f(rf, gf, bf)
    distant_light2.GetAttribute("inputs:color").Set(fill_color)


    # --------------------------------------------------------

    # Per-frame pose randomization for targets and distractors
    randomize_labeled_poses(
        labeled_prims=labeled_prims,
        labeled_radii=labeled_radii,
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
        object_z=OBJECT_Z,
        frame_idx=i,
    )
        # Randomize MDL materials on labeled assets
    # (every frame, or use i % N if you want them to change less often)
    if i % 3 == 0:  # change every 3 frames
        randomize_asset_materials(labeled_prims, asset_mdl_materials)





    randomize_distractor_poses(
        prims=shape_distractors,
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
        object_z=OBJECT_Z + SHAPE_DISTRACTOR_Z_OFFSET,
    )

    randomize_distractor_poses(
        prims=mesh_distractors,
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
        object_z=OBJECT_Z,
    )

    # Randomize floor material every frame (optional)
    if floor_material_urls:
        rep.utils.send_og_event(event_name="randomize_floor_material")

    # Randomize lights
    if i % 5 == 0:
        print(f"\t Randomizing lights")
        rep.utils.send_og_event(event_name="randomize_lights")

    # Randomize shape distractor colors
    if i % 15 == 0:
        print(f"\t Randomizing shape distractors colors")
        rep.utils.send_og_event(event_name="randomize_shape_distractor_colors")

    # Randomize dome background
    #if i % 25 == 0:
    #    print(f"\t Randomizing dome background")
    #    rep.utils.send_og_event(event_name="randomize_dome_background")

    
    # Enable render products only at capture time
    if disable_render_products_between_captures:
        object_based_sdg_utils.set_render_products_updates(render_products, True, include_viewport=False)

    # Capture the current frame
    print(f"[SDG] Capturing frame {i}/{num_frames}, at simulation time: {timeline.get_current_time():.2f}")
    rep.orchestrator.step(delta_time=0.0, rt_subframes=rt_subframes, pause_timeline=False)

    # Disable render products between captures
    if disable_render_products_between_captures:
        object_based_sdg_utils.set_render_products_updates(render_products, False, include_viewport=False)


# Wait for the data to be written
rep.orchestrator.wait_until_complete()

# Stats
wall_duration = time.perf_counter() - wall_time_start
sim_duration = timeline.get_current_time()
if wall_duration > 0:
    avg_frame_fps = num_frames / wall_duration
    num_captures = num_frames * num_cameras
    avg_capture_fps = num_captures / wall_duration
else:
    avg_frame_fps = 0.0
    avg_capture_fps = 0.0
    num_captures = num_frames * num_cameras

print(
    f"[SDG] Captured {num_frames} frames, {num_captures} entries (frames * cameras) in {wall_duration:.2f} seconds.\n"
    f"\t Simulation duration: {sim_duration:.2f}\n"
    f"\t Average frame FPS: {avg_frame_fps:.2f}\n"
    f"\t Average capture entries (frames * cameras) FPS: {avg_capture_fps:.2f}\n"
)

# Cleanup
physx_sub.unsubscribe()
physx_sub = None
simulation_app.update()
timeline.stop()

simulation_app.close()
