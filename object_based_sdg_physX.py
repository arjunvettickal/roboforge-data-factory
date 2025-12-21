# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import random
import time
from itertools import chain
from pathlib import Path
import re

import carb
import carb.settings
import numpy as np
import yaml
from isaacsim import SimulationApp

# ---------------------------------------------------------------------------
# 1. CONFIGURATION SETUP
# ---------------------------------------------------------------------------

# Set up the argument parser to accept a configuration file
parser = argparse.ArgumentParser(description="Run the synthetic data generation pipeline.")
parser.add_argument("--config", required=True, help="Path to the YAML or JSON configuration file.")
args, unknown = parser.parse_known_args()

# Load the configuration file
config = {}
if os.path.isfile(args.config):
    with open(args.config, "r") as f:
        if args.config.endswith(".json"):
            config = json.load(f)
        elif args.config.endswith(".yaml"):
            config = yaml.safe_load(f)
        else:
            raise ValueError("Config file must be .json or .yaml")
else:
    raise FileNotFoundError(f"Config file not found: {args.config}")

print(f"[SDG] Loaded config from {args.config}")

# ---------------------------------------------------------------------------
# 2. START SIMULATION APP
# ---------------------------------------------------------------------------

# Start the Isaac Sim application with settings from the config
launch_config = config.get("launch_config", {"renderer": "RaytracedLighting", "headless": False})
simulation_app = SimulationApp(launch_config=launch_config)

# Import Omniverse and Isaac Sim modules AFTER starting the app
import omni.replicator.core as rep
import omni.timeline
import omni.usd
import usdrt
from isaacsim.core.utils.semantics import add_labels, remove_labels, upgrade_prim_semantics_to_labels
from isaacsim.storage.native import get_assets_root_path
from pxr import PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, Gf, UsdShade

# Custom utility functions
import object_based_sdg_utils

# Get the path to built-in assets
assets_root_path = get_assets_root_path()

# ---------------------------------------------------------------------------
# 3. SCENE SETTINGS (Graphics & Physics)
# ---------------------------------------------------------------------------

# Get access to the global settings
settings = carb.settings.get_settings()

# Make shadows softer (less sharp) for realism
settings.set("/rtx/shadows/softShadows", True)
settings.set("/rtx/shadows/sunArea", 0.01)

# Configure rendering mode (PathTracing or RayTracing)
pt_cfg = config.get("pathtracing", {})
if pt_cfg.get("enabled"):
    settings.set("/rtx/rendermode", "PathTracing")
    settings.set("/rtx/pathtracing/spp", pt_cfg.get("spp", 256))
    settings.set("/rtx/pathtracing/totalSpp", pt_cfg.get("total_spp", 256))
    settings.set("/rtx/pathtracing/optixDenoiser/enabled", 1 if pt_cfg.get("denoiser", True) else 0)
else:
    settings.set("/rtx/rendermode", "RaytracedLighting")

# Add Fog for depth
settings.set("/rtx/fog/enabled", True)
settings.set("/rtx/fog/fogType", "linear")
settings.set("/rtx/fog/density", 0.002)
settings.set("/rtx/fog/startDistance", 0.0)
settings.set("/rtx/fog/endDistance", 100.0)
settings.set("/rtx/fog/fogColor", (0.8, 0.85, 0.9))

# Add Film Grain (like a real camera sensor)
settings.set("/rtx/post/tonemap/filmIso", 800.0)
settings.set("/rtx/post/tonemap/enable", True)

# Add Vignette (darker corners)
settings.set("/rtx/post/vignette/enable", True)
settings.set("/rtx/post/vignette/intensity", 0.6)
settings.set("/rtx/post/vignette/radius", 0.7)

# Open or create the 3D stage (the world)
env_url = config.get("env_url", "")
if env_url:
    # Open an existing environment if provided
    env_path = env_url if env_url.startswith("omniverse://") else assets_root_path + env_url
    omni.usd.get_context().open_stage(env_path)
else:
    # Otherwise, create a new empty stage
    omni.usd.get_context().new_stage()

stage = omni.usd.get_context().get_stage()

# ---------------------------------------------------------------------------
# 4. LIGHTING SETUP
# ---------------------------------------------------------------------------

# Settings for the lights
LIGHT_SETTINGS = {
    # Standard Mode (Day)
    "distant_intensity": 500.0,
    "dome_intensity": 500.0,
    
    # Dark Mode (Night)
    "dark_mode_spot_intensity": 60000.0,
    "dark_mode_spot_cone_angle": 20.0,
    "dark_mode_spot_radius": 3.0,

    # Randomization (Subtle color/temp changes only, intensity fixed to standard)
    "random_light_color_min": (0.9, 0.9, 0.9),
    "random_light_color_max": (1.0, 1.0, 1.0),
    "random_light_temperature_mean": 6500,
    "random_light_temperature_std": 500,
}

# Create a "Distant Light" (like the sun)
distant_light = stage.DefinePrim("/World/Lights/DistantLight", "DistantLight")
distant_light.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(LIGHT_SETTINGS["distant_intensity"])
distant_light.CreateAttribute("inputs:angle", Sdf.ValueTypeNames.Float).Set(0.5) # Soften shadows

# Point the light straight down
if not distant_light.HasAttribute("xformOp:rotateXYZ"):
    UsdGeom.Xformable(distant_light).AddRotateXYZOp()
distant_light.GetAttribute("xformOp:rotateXYZ").Set((0, 0, 0))

# Create a Spot Light (DiskLight with Shaping API)
# It will be enabled only during "Dark Mode" frames
spot_light_path = "/World/Lights/SpotLight"
spot_light = UsdLux.DiskLight.Define(stage, spot_light_path)
spot_light.CreateIntensityAttr().Set(0.0) # Disabled by default
spot_light.CreateRadiusAttr().Set(LIGHT_SETTINGS.get("dark_mode_spot_radius", 1.5)) # Larger radius for soft shadows (like a studio softbox)
spot_light.CreateColorAttr().Set((1.0, 1.0, 1.0))

# Position above center
if not spot_light.GetPrim().HasAttribute("xformOp:translate"):
    UsdGeom.Xformable(spot_light).AddTranslateOp()
spot_light.GetPrim().GetAttribute("xformOp:translate").Set((0.0, 0.0, 8.0))

# Apply Shaping API to make it a spot light
# Note: In some USD versions UsdLux.ShapingAPI might be different, but this is standard
try:
    shaping_api = UsdLux.ShapingAPI.Apply(spot_light.GetPrim())
    shaping_api.CreateShapingConeAngleAttr().Set(LIGHT_SETTINGS.get("dark_mode_spot_cone_angle", 35.0)) # Narrow cone
    shaping_api.CreateShapingConeSoftnessAttr().Set(0.2)
    shaping_api.CreateShapingFocusAttr().Set(10.0)
    
    # Rotate to point down
    if not spot_light.GetPrim().HasAttribute("xformOp:rotateXYZ"):
        UsdGeom.Xformable(spot_light).AddRotateXYZOp()
    spot_light.GetPrim().GetAttribute("xformOp:rotateXYZ").Set((0, 0, 0))
except Exception as e:
    print(f"[SDG] Warning: Could not apply ShapingAPI to SpotLight: {e}")

# ---------------------------------------------------------------------------
# 5. WORKSPACE SETUP (Floor & Walls)
# ---------------------------------------------------------------------------

# Define the size of the area where objects will drop
working_area_size = config.get("working_area_size", (3, 3, 3))
min_x = -working_area_size[0] / 2.0
max_x =  working_area_size[0] / 2.0
min_y = -working_area_size[1] / 2.0
max_y =  working_area_size[1] / 2.0
floor_height = -working_area_size[2] / 2.0

# Define where objects spawn (above the floor)
spawn_min_z = floor_height + 0.80
spawn_max_z = floor_height + 1.50
working_area_min = (min_x, min_y, spawn_min_z)
working_area_max = (max_x, max_y, spawn_max_z)
OBJECT_Z = floor_height + 1.2

# Create invisible walls to keep objects inside the area
object_based_sdg_utils.create_collision_box_walls(
    stage, "/World/CollisionWalls", working_area_size[0], working_area_size[1], working_area_size[2]
)

# Create the floor (Ground Plane)
ground_plane_size = max(working_area_size[0], working_area_size[1]) * 4.0

def create_ground_plane_mesh(stage, path, size, height):
    """Creates a simple square floor."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    half_size = size / 2.0
    
    # 4 corners of the square
    points = [(-half_size, -half_size, 0), (half_size, -half_size, 0), (half_size, half_size, 0), (-half_size, half_size, 0)]
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateNormalsAttr([(0, 0, 1)] * 4)
    
    # Texture coordinates (UVs)
    uv_scale = 4.0
    tex_coords = [(0, 0), (uv_scale, 0), (uv_scale, uv_scale), (0, uv_scale)]
    primvars_api = UsdGeom.PrimvarsAPI(mesh)
    pv = primvars_api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    pv.Set(tex_coords)
    
    # Move it to the correct height
    object_based_sdg_utils.set_transform_attributes(mesh.GetPrim(), location=(0.0, 0.0, height))
    return mesh.GetPrim()

ground_plane_prim = create_ground_plane_mesh(stage, "/World/groundPlane", ground_plane_size, floor_height)
# Use exact mesh collision for the floor and negative rest offset to ensure assets settle flush (no floating gap)
object_based_sdg_utils.add_colliders(ground_plane_prim, approximation_shape="none", rest_offset=-0.002)

# Setup Physics (Gravity, Time steps)
usdrt_stage = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())
physics_scenes = usdrt_stage.GetPrimsWithAppliedAPIName("PhysxSceneAPI")
if physics_scenes:
    physics_scene = physics_scenes[0]
else:
    physics_scene = UsdPhysics.Scene.Define(stage, "/PhysicsScene")
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/PhysicsScene"))
physx_scene.GetTimeStepsPerSecondAttr().Set(60)

# ---------------------------------------------------------------------------
# 6. ASSET MANAGEMENT (Spawning & Materials)
# ---------------------------------------------------------------------------

labeled_assets_and_properties = config.get("labeled_assets_and_properties", [])
labeled_prims = []
drop_height_min = floor_height + 0.5
drop_height_max = floor_height + working_area_size[2] * 0.8

def clear_labeled_assets():
    """Removes all spawned objects from the previous frame."""
    global labeled_prims
    if stage.GetPrimAtPath("/World/Labeled").IsValid():
        stage.RemovePrim("/World/Labeled")
    stage.DefinePrim("/World/Labeled", "Xform")
    labeled_prims = []

def spawn_labeled_assets(assets_to_spawn=None):
    """Spawns a batch of objects in the air, ready to drop."""
    global labeled_prims
    clear_labeled_assets()
    spawned = []
    
    # Use the provided list of assets, or all of them if none provided
    chosen_objs = assets_to_spawn if assets_to_spawn else labeled_assets_and_properties

    # Spawn them in a small area at the center so they pile up
    spawn_width = 1.0
    spawn_height = 1.0
    spawn_min_x, spawn_max_x = -spawn_width/2, spawn_width/2
    spawn_min_y, spawn_max_y = -spawn_height/2, spawn_height/2
    print(f"[SDG] Spawning objects at center with spread {spawn_width}x{spawn_height}")

    for obj in chosen_objs:
        obj_url = obj.get("url", "")
        label = obj.get("label", "unknown")
        count = obj.get("count", 1)
        
        for _ in range(count):
            # Random position and rotation
            rand_loc, rand_rot, _ = object_based_sdg_utils.get_random_transform_values(
                loc_min=(spawn_min_x, spawn_min_y, drop_height_min),
                loc_max=(spawn_max_x, spawn_max_y, drop_height_max),
                scale_min_max=(1, 1),
            )
            
            # Create the object
            prim_path = omni.usd.get_stage_next_free_path(stage, f"/World/Labeled/{label}", False)
            prim = stage.DefinePrim(prim_path, "Xform")
            asset_path = obj_url if obj_url.startswith("omniverse://") else assets_root_path + obj_url
            prim.GetReferences().AddReference(asset_path)
            
            # Set position and physics
            object_based_sdg_utils.set_transform_attributes(prim, location=rand_loc, rotation=rand_rot)
            object_based_sdg_utils.add_colliders(prim)
            object_based_sdg_utils.add_rigid_body_dynamics(prim, disable_gravity=False)
            
            # Add semantic label (for training)
            add_labels(prim, labels=[label], instance_name="class")
            
            # Reset velocity
            if prim.HasAttribute("physics:velocity"):
                prim.GetAttribute("physics:velocity").Set((0.0, 0.0, 0.0))
            if prim.HasAttribute("physics:angularVelocity"):
                prim.GetAttribute("physics:angularVelocity").Set((0.0, 0.0, 0.0))
            spawned.append(prim)
            
    labeled_prims = spawned
    return spawned

def wait_for_labeled_assets_to_settle(timeline, max_duration=3.0, lin_thresh=0.02, ang_thresh=1.0, check_interval=10):
    """Runs the simulation until objects stop moving (settle on the floor)."""
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
        
        # Check every few steps to save performance
        if steps % check_interval != 0:
            current_time = timeline.get_current_time()
            elapsed += current_time - previous_time
            previous_time = current_time
            continue

        # Check if all objects have stopped moving
        all_settled = True
        for prim in labeled_prims + mesh_distractors:
            lin_vel = prim.GetAttribute("physics:velocity").Get() if prim.HasAttribute("physics:velocity") else (0, 0, 0)
            ang_vel = prim.GetAttribute("physics:angularVelocity").Get() if prim.HasAttribute("physics:angularVelocity") else (0, 0, 0)
            
            if any(abs(v) > lin_thresh for v in lin_vel) or any(abs(v) > ang_thresh for v in ang_vel):
                all_settled = False
                break
                
        current_time = timeline.get_current_time()
        elapsed += current_time - previous_time
        previous_time = current_time
        
        if all_settled:
            break

# ---------------------------------------------------------------------------
# 7. DISTRACTORS (Random objects to confuse the model)
# ---------------------------------------------------------------------------

# Shape Distractors (Simple shapes like cubes, spheres)
shape_distractors_types = config.get("shape_distractors_types", ["capsule", "cone", "cylinder", "sphere", "cube"])
shape_distractors_scale_min_max = config.get("shape_distractors_scale_min_max", (0.02, 0.2))
shape_distractors_num = config.get("shape_distractors_num", 350)
shape_distractors = []
falling_shape_distractors = []
floating_shape_distractors = [] # Kept for compatibility, though we make them all fall now

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
    
    # Enable gravity so they fall
    object_based_sdg_utils.add_rigid_body_dynamics(prim, disable_gravity=False)
    falling_shape_distractors.append(prim)
    shape_distractors.append(prim)

# Mesh Distractors (Complex 3D models)
mesh_distractors_urls = config.get("mesh_distractors_urls", [])
mesh_distractors_scale_min_max = config.get("mesh_distractors_scale_min_max", (0.1, 2.0))
mesh_distractors_num = config.get("mesh_distractors_num", 20)
mesh_distractors = []

def clear_mesh_distractors():
    """Removes mesh distractors from the previous frame."""
    global mesh_distractors
    for prim in mesh_distractors:
        stage.RemovePrim(prim.GetPath())
    if stage.GetPrimAtPath("/World/Distractors/Mesh").IsValid():
        stage.RemovePrim("/World/Distractors/Mesh")
    stage.DefinePrim("/World/Distractors/Mesh", "Xform")
    mesh_distractors = []

def spawn_mesh_distractors():
    """Spawns complex mesh distractors."""
    global mesh_distractors
    clear_mesh_distractors()
    
    spawn_width = 0.8
    spawn_height = 0.8
    spawn_min_x, spawn_max_x = -spawn_width/2, spawn_width/2
    spawn_min_y, spawn_max_y = -spawn_height/2, spawn_height/2

    for i in range(mesh_distractors_num):
        rand_loc, rand_rot, rand_scale = object_based_sdg_utils.get_random_transform_values(
            loc_min=(spawn_min_x, spawn_min_y, drop_height_min),
            loc_max=(spawn_max_x, spawn_max_y, drop_height_max),
            scale_min_max=mesh_distractors_scale_min_max,
        )
        mesh_url = random.choice(mesh_distractors_urls)
        
        # Clean up the name
        prim_name_raw = os.path.basename(mesh_url).split(".")[0]
        prim_name = re.sub(r"[^A-Za-z0-9_]", "_", prim_name_raw) # Note: 're' is not imported, need to import it or use string methods. I'll add 'import re' at top.
        if not prim_name or not (prim_name[0].isalpha() or prim_name[0] == "_"):
            prim_name = f"mesh_{prim_name}"
            
        prim_path = omni.usd.get_stage_next_free_path(stage, f"/World/Distractors/Mesh/{prim_name}", False)
        prim = stage.DefinePrim(prim_path, "Xform")
        
        asset_path = mesh_url if mesh_url.startswith("omniverse://") else assets_root_path + mesh_url
        prim.GetReferences().AddReference(asset_path)
        
        object_based_sdg_utils.set_transform_attributes(prim, location=rand_loc, rotation=rand_rot, scale=rand_scale)
        object_based_sdg_utils.add_colliders(prim)
        object_based_sdg_utils.add_rigid_body_dynamics(prim, disable_gravity=False)
        
        # Reset velocity
        if prim.HasAttribute("physics:velocity"):
            prim.GetAttribute("physics:velocity").Set((0.0, 0.0, 0.0))
        if prim.HasAttribute("physics:angularVelocity"):
            prim.GetAttribute("physics:angularVelocity").Set((0.0, 0.0, 0.0))
            
        mesh_distractors.append(prim)
        
        # Ensure they don't have labels (they are distractors)
        upgrade_prim_semantics_to_labels(prim, include_descendants=True)
        remove_labels(prim, include_descendants=True)

# ---------------------------------------------------------------------------
# 8. MATERIALS (Textures)
# ---------------------------------------------------------------------------

def build_mdl_material_library(stage, mdl_entries, prefix="AssetMat"):
    """Creates a list of materials from the config."""
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
        materials.append(mtl)
    return materials

def randomize_asset_materials(prims, materials):
    """Applies random materials to objects."""
    if not materials:
        return
    
    # Shuffle the materials to ensure variety
    pool = materials[:]
    random.shuffle(pool)

    for i, prim in enumerate(prims):
        mat = pool[i % len(pool)]
        UsdShade.MaterialBindingAPI(prim).Bind(mat)

def bind_material_to_meshes(root_prim, material):
    """Helper to apply material to all parts of an object."""
    if material is None:
        return
    for desc in Usd.PrimRange(root_prim):
        if desc.IsA(UsdGeom.Gprim):
            UsdShade.MaterialBindingAPI(desc).Bind(material)

def randomize_ground_plane_material(materials):
    """Applies a random material to the floor."""
    if not materials:
        return
    mat = random.choice(materials)
    UsdShade.MaterialBindingAPI(ground_plane_prim).Bind(mat)
    bind_material_to_meshes(ground_plane_prim, mat)
    
    mdl_url = mat.GetPrim().GetAttribute("user:mdl_url").Get()
    subid = mat.GetPrim().GetAttribute("user:mdl_subidentifier").Get()
    print(f"[SDG] GroundPlane material -> {subid} ({mdl_url})")

# ---------------------------------------------------------------------------
# 9. CAMERA SETUP
# ---------------------------------------------------------------------------

# Replicator setup
rep.orchestrator.set_capture_on_play(False)

cameras = []
num_cameras = config.get("num_cameras", 1)
camera_properties_kwargs = config.get("camera_properties_kwargs", {})

for i in range(num_cameras):
    cam_prim = stage.DefinePrim(f"/World/Cameras/cam_{i}", "Camera")
    for key, value in camera_properties_kwargs.items():
        if cam_prim.HasAttribute(key):
            cam_prim.GetAttribute(key).Set(value)
    cameras.append(cam_prim)

# Camera collision spheres (optional)
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

simulation_app.update()

# Create "Render Products" (the actual images to be captured)
render_products = []
resolution = config.get("resolution", (640, 480))
for cam in cameras:
    rp = rep.create.render_product(cam.GetPath(), resolution)
    render_products.append(rp)

# Configure the "Writer" (saves the data to disk)
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

    print(f"[INFO] Randomly updated {len(cameras)} cameras.")

def update_camera_pose(frame_idx=0):
    """Moves the cameras to 4 corners to avoid similar frames."""
    if not labeled_prims:
        return

    # Configuration for randomization
    dist_min, dist_max = config.get("camera_distance_to_target_min_max", (5.0, 10.0))
    target = np.array([0.0, 0.0, -0.5]) # Looking at floor center
    
    # Define base angles for 4 corners (45, 135, 225, 315 degrees)
    quarter_pi = np.pi / 4
    base_angles = [1 * quarter_pi, 3 * quarter_pi, 5 * quarter_pi, 7 * quarter_pi]

    for i, cam_prim in enumerate(cameras):
        # Assign each camera to a corner sequentially
        # If more than 4 cameras, it wraps around
        corner_idx = i % 4
        base_angle = base_angles[corner_idx]
        
        # Add slight randomness to the angle (+/- 15 degrees) so it's not perfectly identical every frame
        angle_jitter = random.uniform(-np.radians(15), np.radians(15))
        azimuth = base_angle + angle_jitter

        # Elevation: high angle look-down
        elevation = random.uniform(np.radians(60), np.radians(80))
        distance = random.uniform(dist_min, dist_max)

        # Spherical to Cartesian
        x = distance * np.cos(azimuth) * np.cos(elevation)
        y = distance * np.sin(azimuth) * np.cos(elevation)
        z = distance * np.sin(elevation)
        
        cam_loc = np.array([x, y, z]) + target

        # Calculate rotation to look at target
        eye = Gf.Vec3d(cam_loc.tolist())
        center = Gf.Vec3d(target.tolist())
        up = Gf.Vec3d(0, 0, 1)

        m = Gf.Matrix4d().SetLookAt(eye, center, up)
        m = m.GetInverse()
        q = m.ExtractRotation().GetQuat()

        object_based_sdg_utils.set_transform_attributes(
            cam_prim,
            location=tuple(cam_loc),
            orientation=Gf.Quatf(q)
        )
    print(f"[INFO] Updated {len(cameras)} cameras to 4 corners.")

# ---------------------------------------------------------------------------
# 10. RANDOMIZERS (Dome, Lights, Colors)
# ---------------------------------------------------------------------------

# Shape Distractor Colors
with rep.trigger.on_custom_event(event_name="randomize_shape_distractor_colors"):
    shape_distractors_paths = [prim.GetPath() for prim in chain(floating_shape_distractors, falling_shape_distractors)]
    shape_distractors_group = rep.create.group(shape_distractors_paths)
    with shape_distractors_group:
        rep.randomizer.color(colors=rep.distribution.uniform((0, 0, 0), (1, 1, 1)))

# Lights
with rep.trigger.on_custom_event(event_name="randomize_lights"):
    distant_light_rep = rep.get.prims(path_pattern="/World/Lights/DistantLight")
    with distant_light_rep:
        # Note: Rotation randomization removed to keep light fixed from top
        rep.randomizer.color(colors=rep.distribution.uniform(LIGHT_SETTINGS["random_light_color_min"], LIGHT_SETTINGS["random_light_color_max"]))
        rep.modify.attribute("inputs:intensity", LIGHT_SETTINGS["distant_intensity"])
        rep.modify.attribute("inputs:colorTemperature", rep.distribution.normal(LIGHT_SETTINGS["random_light_temperature_mean"], LIGHT_SETTINGS["random_light_temperature_std"]))
        rep.modify.attribute("inputs:angle", rep.distribution.uniform(0.1, 1.5))

# Dome Background
dome_textures = [
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/NVIDIA/Assets/Skies/Indoor/wooden_lounge_4k.hdr"
]

def randomize_dome_background():
    tex = random.choice(dome_textures)
    dome_path = "/World/DomeLight"
    if not stage.GetPrimAtPath(dome_path).IsValid():
        dome_light = UsdLux.DomeLight.Define(stage, dome_path)
    else:
        dome_light = UsdLux.DomeLight(stage.GetPrimAtPath(dome_path))
        
    dome_light.CreateIntensityAttr().Set(LIGHT_SETTINGS["dome_intensity"])
    dome_light.CreateTextureFileAttr().Set(Sdf.AssetPath(tex))
    dome_light.CreateColorAttr().Set((1.0, 1.0, 1.0))
    dome_light.CreateTextureFormatAttr().Set("latlong")
    
    # Random rotation for the dome
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

# ---------------------------------------------------------------------------
# 11. CAPTURE LOGIC
# ---------------------------------------------------------------------------

def capture_with_motion_blur_and_pathtracing(duration=0.05, num_samples=8, spp=64):
    """Captures a frame with high-quality settings and motion blur."""
    # Temporarily increase physics speed for smoother motion blur
    orig_physics_fps = physx_scene.GetTimeStepsPerSecondAttr().Get()
    target_physics_fps = 1 / duration * num_samples
    if target_physics_fps > orig_physics_fps:
        physx_scene.GetTimeStepsPerSecondAttr().Set(target_physics_fps)

    # Enable motion blur
    is_motion_blur_enabled = carb.settings.get_settings().get("/omni/replicator/captureMotionBlur")
    if not is_motion_blur_enabled:
        carb.settings.get_settings().set("/omni/replicator/captureMotionBlur", True)
    carb.settings.get_settings().set("/omni/replicator/pathTracedMotionBlurSubSamples", num_samples)

    # Switch to Path Tracing
    prev_render_mode = carb.settings.get_settings().get("/rtx/rendermode")
    carb.settings.get_settings().set("/rtx/rendermode", "PathTracing")
    carb.settings.get_settings().set("/rtx/pathtracing/spp", spp)
    carb.settings.get_settings().set("/rtx/pathtracing/totalSpp", spp)
    carb.settings.get_settings().set("/rtx/pathtracing/optixDenoiser/enabled", 0)

    if not timeline.is_playing():
        timeline.play()

    # Take the picture
    rep.orchestrator.step(delta_time=duration, pause_timeline=False)

    # Restore settings
    if target_physics_fps > orig_physics_fps:
        physx_scene.GetTimeStepsPerSecondAttr().Set(orig_physics_fps)

    carb.settings.get_settings().set("/omni/replicator/captureMotionBlur", is_motion_blur_enabled)
    carb.settings.get_settings().set("/rtx/rendermode", prev_render_mode)

# ---------------------------------------------------------------------------
# 12. MAIN SIMULATION LOOP
# ---------------------------------------------------------------------------

# Simulation parameters
num_frames = config.get("num_frames", 10)
rt_subframes = config.get("rt_subframes", -1)
disable_render_products_between_captures = config.get("disable_render_products_between_captures", True)

# Initial setup
rep.utils.send_og_event(event_name="randomize_shape_distractor_colors")
randomize_dome_background()
for _ in range(5):
    simulation_app.update()

# Load materials
asset_mdl_entries = config.get("asset_mdl_materials", [])
asset_mdl_materials = build_mdl_material_library(stage, asset_mdl_entries, prefix="AssetMat")
floor_mdl_entries = config.get("floor_mdl_materials", [])
floor_mdl_materials = build_mdl_material_library(stage, floor_mdl_entries, prefix="FloorMat")

# Load Dark Mode Material (Tire)
dark_mode_entries = [{"mdl_url": "omniverse://localhost/NVIDIA/Materials/vMaterials_2/Other/Rubber/Tire.mdl"}]
dark_mode_materials = build_mdl_material_library(stage, dark_mode_entries, prefix="DarkGroundMat")

# Start timeline
timeline = omni.timeline.get_timeline_interface()
timeline.set_start_time(0)
timeline.set_end_time(1000000)
timeline.set_looping(False)
timeline.play()
timeline.commit()
simulation_app.update()

wall_time_start = time.perf_counter()

# Initial camera position
update_camera_pose(frame_idx=0)

# Initialize balanced sampling for assets (Deck of Cards method)
all_asset_indices = list(range(len(labeled_assets_and_properties)))
random.shuffle(all_asset_indices)
asset_deck_ptr = 0

# Loop through frames
for i in range(num_frames):
    print(f"[SDG] Spawning and dropping labeled assets for frame {i}")
    
    # Check for Dark Mode early
    is_dark_mode = (i % 4 == 0)
    
    # Select up to 7 assets in a balanced way
    current_batch_assets = []
    if len(labeled_assets_and_properties) <= 7:
        current_batch_assets = labeled_assets_and_properties
    else:
        selected_indices = []
        while len(selected_indices) < 7:
            needed = 7 - len(selected_indices)
            available = len(all_asset_indices) - asset_deck_ptr
            take = min(needed, available)
            selected_indices.extend(all_asset_indices[asset_deck_ptr : asset_deck_ptr + take])
            asset_deck_ptr += take
            if asset_deck_ptr >= len(all_asset_indices):
                random.shuffle(all_asset_indices)
                asset_deck_ptr = 0
        current_batch_assets = [labeled_assets_and_properties[idx] for idx in selected_indices]

    # Spawn assets and distractors
    spawn_labeled_assets(assets_to_spawn=current_batch_assets)
    spawn_mesh_distractors()
    
    # Randomize materials
    randomize_asset_materials(labeled_prims, asset_mdl_materials)
    
    if is_dark_mode and dark_mode_materials:
        randomize_ground_plane_material(dark_mode_materials)
    else:
        randomize_ground_plane_material(floor_mdl_materials)
    
    # Wait for things to fall and settle
    wait_for_labeled_assets_to_settle(timeline, max_duration=5.0, lin_thresh=0.15, ang_thresh=0.15, check_interval=5)
    
    # Update camera
    update_camera_pose(frame_idx=i)

    # Periodic randomizations
    if i % 5 == 0:
        print(f"\t Randomizing lights")
        rep.utils.send_og_event(event_name="randomize_lights")

    if i % 15 == 0:
        print(f"\t Randomizing shape distractors colors")
        rep.utils.send_og_event(event_name="randomize_shape_distractor_colors")

    if i % 25 == 0:
        print(f"\t Randomizing dome background")
        randomize_dome_background()

    # Randomly turn off dome light for variety
    # Determine if this frame is distinct "Dark Mode" (25% of frames)
    # e.g. every 4th frame (0, 4, 8, 12...)
    # is_dark_mode already calculated at top of loop

    # Randomly turn off dome light for variety (Normal Mode only)
    # If Dark Mode, we force it off later
    current_dome_intensity = LIGHT_SETTINGS["dome_intensity"]
    if stage.GetPrimAtPath("/World/DomeLight").IsValid():
        dome_light = UsdLux.DomeLight(stage.GetPrimAtPath("/World/DomeLight"))
        if not is_dark_mode:
            # User request: In normal mode, remove all lights except dome and distant.
            # Set Dome Light to a good value (LIGHT_SETTINGS value).
            dome_light.GetIntensityAttr().Set(LIGHT_SETTINGS["dome_intensity"])

    # --- APPLY LIGHTING MODES ---
    if is_dark_mode:
        print(f"\t [Dark Mode] Frame {i}: Activating SpotLight only.")
        # Disable Dome
        if stage.GetPrimAtPath("/World/DomeLight").IsValid():
             UsdLux.DomeLight(stage.GetPrimAtPath("/World/DomeLight")).GetIntensityAttr().Set(0.0)
        
        # Disable Distant Light
        if stage.GetPrimAtPath("/World/Lights/DistantLight").IsValid():
            UsdLux.DistantLight(stage.GetPrimAtPath("/World/Lights/DistantLight")).GetIntensityAttr().Set(0.0)

        # Enable Spot Light
        if stage.GetPrimAtPath(spot_light_path).IsValid():
             # High intensity for the spot since it's the only source
             UsdLux.DiskLight(stage.GetPrimAtPath(spot_light_path)).GetIntensityAttr().Set(LIGHT_SETTINGS["dark_mode_spot_intensity"]) 

    else:
        # Normal Mode: Ensure standard lights are ON (or restored to random values)
        # Note: randomize_lights event (triggered above) sets intensity for DistantLight, so we don't force it here
        # unless we need to recover from 0. However, Replicator randomization applies per frame or when triggered.
        # Since we triggered it at i%5==0, the value persists until next trigger.
        # If previous frame was Dark Mode, we might need to restore DistantLight.
        
        # Restore Spot Light to OFF
        if stage.GetPrimAtPath(spot_light_path).IsValid():
             UsdLux.DiskLight(stage.GetPrimAtPath(spot_light_path)).GetIntensityAttr().Set(0.0)

        # Ensure DistantLight is set to its good value
        if stage.GetPrimAtPath("/World/Lights/DistantLight").IsValid():
            d_light = UsdLux.DistantLight(stage.GetPrimAtPath("/World/Lights/DistantLight"))
            d_light.GetIntensityAttr().Set(LIGHT_SETTINGS["distant_intensity"]) 
             
        # Dome light was handled above

    # Capture!
    print(f"[SDG] Capturing frame {i}/{num_frames}, at simulation time: {timeline.get_current_time():.2f}")
    if disable_render_products_between_captures:
        object_based_sdg_utils.set_render_products_updates(render_products, True, include_viewport=False)

    if i % 5 == 0:
        capture_with_motion_blur_and_pathtracing(duration=0.025, num_samples=4, spp=64)
    else:
        rep.orchestrator.step(delta_time=0.0, rt_subframes=rt_subframes, pause_timeline=False)

    if disable_render_products_between_captures:
        object_based_sdg_utils.set_render_products_updates(render_products, False, include_viewport=False)
    print(f"[SDG] Finished frame {i}")

# Finish up
rep.orchestrator.wait_until_complete()

# ---------------------------------------------------------------------------
# 13. POST-PROCESSING (YOLO Conversion)
# ---------------------------------------------------------------------------

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
yolo_output_dir = config.get("yolo_output_dir")
yolo_split_ratio = config.get("yolo_split_ratio", 0.8)
yolo_split_seed = config.get("yolo_split_seed")
if writer_type == "CocoWriter" and out_dir:
    convert_coco_to_yolo_and_split(out_dir, yolo_output_dir, train_ratio=yolo_split_ratio, seed=yolo_split_seed)

# Print statistics
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
