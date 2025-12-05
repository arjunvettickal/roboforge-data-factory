# Minimal object-based SDG pipeline with COCO + YOLO output
# Keeps: multiple floors, shape distractors, MDL materials, labeled assets, top-down camera, two lights, light randomization

import argparse
import json
import os
import random
import time
from pathlib import Path
import shutil

import yaml
from isaacsim import SimulationApp


# -------------- Config --------------
def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=False)
    args, _ = parser.parse_known_args()
    cfg = {}
    if args.config and os.path.isfile(args.config):
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) if args.config.endswith(".yaml") else json.load(f)
    return cfg


# -------------- Simple helpers --------------
def set_transform(prim, loc=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1)):
    from pxr import UsdGeom, Gf
    api = UsdGeom.XformCommonAPI(prim)
    api.SetTranslate(Gf.Vec3d(*loc))
    api.SetRotate(Gf.Vec3f(*rot))
    api.SetScale(Gf.Vec3f(*scale))


def random_pose_2d(count, radius, min_x, max_x, min_y, max_y, z):
    poses = []
    for _ in range(count):
        for _ in range(20):
            x = random.uniform(min_x + radius, max_x - radius)
            y = random.uniform(min_y + radius, max_y - radius)
            if all((x - px) ** 2 + (y - py) ** 2 > (radius * 2) ** 2 for px, py, _ in poses):
                poses.append((x, y, z))
                break
        else:
            poses.append((random.uniform(min_x, max_x), random.uniform(min_y, max_y), z))
    return poses


def build_mdl_materials(stage, entries):
    from pxr import Sdf, UsdShade
    materials = []
    stage.DefinePrim("/World/Looks", "Scope")
    for idx, entry in enumerate(entries or []):
        mdl_url = entry.get("mdl_url")
        subid = entry.get("subidentifier") or (mdl_url.split("/")[-1].replace(".mdl", "") if mdl_url else None)
        if not mdl_url or not subid:
            continue
        mtl_path = Sdf.Path(f"/World/Looks/AssetMat_{idx}")
        mtl = UsdShade.Material.Define(stage, mtl_path)
        shader = UsdShade.Shader.Define(stage, mtl_path.AppendPath("Shader"))
        shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
        shader.SetSourceAsset(mdl_url, "mdl")
        shader.SetSourceAssetSubIdentifier(subid, "mdl")
        surf_out = mtl.CreateSurfaceOutput("mdl")
        surf_out.ConnectToSource(shader.ConnectableAPI(), "out")
        materials.append(mtl)
    return materials


def convert_coco_to_yolo(coco_root, yolo_root=None, train_ratio=0.8, seed=None):
    coco_root = Path(coco_root)

    def find_coco_json(root: Path):
        jsons = list(root.glob("*.json"))
        if jsons:
            return max(jsons, key=lambda p: p.stat().st_size)
        # try siblings like coco_out_0001
        candidates = []
        for sib in root.parent.glob(f"{root.name}*"):
            candidates.extend(sib.glob("*.json"))
        if candidates:
            return max(candidates, key=lambda p: p.stat().st_mtime)
        return None

    ann = find_coco_json(coco_root)
    if ann is None:
        print(f"[YOLO] No COCO json in {coco_root} or siblings")
        return
    coco_root = ann.parent

    # Resolve YOLO root
    if yolo_root:
        yolo_root = Path(yolo_root)
        if not yolo_root.is_absolute():
            yolo_root = coco_root.parent / yolo_root
    else:
        yolo_root = coco_root.parent / "yolo_out"

    data = json.loads(ann.read_text())
    print(f"[YOLO] Using COCO annotations: {ann}")
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])
    cat_map = {c["id"]: i for i, c in enumerate(sorted(categories, key=lambda c: c["id"]))}
    names = [c["name"] for c in sorted(categories, key=lambda c: c["id"])]

    anns_by_img = {}
    for a in annotations:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    rng = random.Random(seed if seed is not None else time.time())
    rng.shuffle(images)
    split = int(len(images) * train_ratio)
    splits = {"train": images[:split], "val": images[split:]}

    yolo_root.mkdir(parents=True, exist_ok=True)
    (yolo_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (yolo_root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (yolo_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (yolo_root / "labels" / "val").mkdir(parents=True, exist_ok=True)

    for split_name, imgs in splits.items():
        for img in imgs:
            fname = img["file_name"]
            stem = Path(fname).stem
            src_img = next((p for p in [
                coco_root / fname,
                coco_root / "Replicator" / fname,
                coco_root / "rgb" / fname,
            ] if p.exists()), None)
            if src_img is None:
                continue
            shutil.copy2(src_img, yolo_root / "images" / split_name / src_img.name)
            W, H = img["width"], img["height"]
            lines = []
            for a in anns_by_img.get(img["id"], []):
                x, y, w, h = a["bbox"]
                cx, cy = (x + w / 2) / W, (y + h / 2) / H
                lines.append(f"{cat_map[a['category_id']]} {cx:.6f} {cy:.6f} {w/W:.6f} {h/H:.6f}")
            (yolo_root / "labels" / split_name / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

    names_str = "\n".join([f"  {i}: {n}" for i, n in enumerate(names)])
    rel_path = os.path.relpath(yolo_root, start=yolo_root)
    (yolo_root / "data.yaml").write_text(
        f"path: {rel_path}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"# Number of classes\n"
        f"nc: 10\n"
        f"names:\n{names_str}\n"
    )
    print(f"[YOLO] Wrote YOLO dataset to {yolo_root}")


# -------------- Main --------------
def main():
    cfg = load_config()
    launch = cfg.get("launch_config", {"renderer": "RaytracedLighting", "headless": True})
    sim = SimulationApp(launch_config=launch)

    # Delayed imports that require SimulationApp
    import omni.replicator.core as rep
    import omni.timeline
    import omni.usd
    from pxr import Sdf, UsdGeom, UsdShade, Gf, UsdPhysics, PhysxSchema
    from isaacsim.storage.native import get_assets_root_path
    from isaacsim.core.utils.semantics import add_labels

    assets_root = get_assets_root_path()
    stage = omni.usd.get_context().get_stage()

    # Stage setup
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()

    # Floors
    floors = cfg.get("floors", [])
    floor_prim = stage.DefinePrim("/World/Floor", "Xform")
    floor_refs = floor_prim.GetReferences()

    # Physics scene (minimal)
    phys_scene = UsdPhysics.Scene.Define(stage, "/PhysicsScene")
    PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/PhysicsScene"))

    # Camera
    cam = stage.DefinePrim("/World/Cameras/cam_0", "Camera")

    # Lights (key/fill)
    key = stage.DefinePrim("/World/Lights/Key", "DistantLight")
    UsdGeom.Xformable(key).AddRotateXYZOp().Set((0, 60, 0))
    key.CreateAttribute("inputs:angle", Sdf.ValueTypeNames.Float).Set(7.0)
    fill = stage.DefinePrim("/World/Lights/Fill", "DistantLight")
    UsdGeom.Xformable(fill).AddRotateXYZOp().Set((-15, -30, 0))
    fill.CreateAttribute("inputs:angle", Sdf.ValueTypeNames.Float).Set(10.0)

    # Materials
    mdl_mats = build_mdl_materials(stage, cfg.get("asset_mdl_materials", []))

    # Working area
    area = cfg.get("working_area_size", (4, 4, 3))
    min_x, max_x = -area[0] / 2, area[0] / 2
    min_y, max_y = -area[1] / 2, area[1] / 2
    floor_z = -area[2] / 2
    object_z = floor_z + 2.0  # lifted above floor

    # Labeled assets
    labeled_cfg = cfg.get("labeled_assets_and_properties", [])
    labeled_prims = []
    for obj in labeled_cfg:
        url = obj.get("url", "")
        label = obj.get("label", "obj")
        count = obj.get("count", 1)
        scale_min, scale_max = obj.get("scale_min_max", (1, 1))
        for _ in range(count):
            prim = stage.DefinePrim(omni.usd.get_stage_next_free_path(stage, f"/World/Labeled/{label}", False), "Xform")
            prim.GetReferences().AddReference(url if url.startswith("omniverse://") else assets_root + url)
            s = random.uniform(scale_min, scale_max)
            # initial placement roughly centered above floor to avoid penetration
            set_transform(prim, loc=(random.uniform(min_x, max_x), random.uniform(min_y, max_y), object_z),
                          rot=(0, 0, 0), scale=(s, s, s))
            add_labels(prim, labels=[label], instance_name="class")
            labeled_prims.append(prim)

    # Shape distractors
    shape_types = cfg.get("shape_distractors_types", ["cube", "sphere"])
    shape_num = cfg.get("shape_distractors_num", 0)
    shape_scale_min, shape_scale_max = cfg.get("shape_distractors_scale_min_max", (0.02, 0.15))
    shape_prims = []
    for i in range(shape_num):
        kind = random.choice(shape_types).capitalize()
        prim = stage.DefinePrim(omni.usd.get_stage_next_free_path(stage, f"/World/Distractors/{kind}", False), kind)
        s = random.uniform(shape_scale_min, shape_scale_max)
        set_transform(prim, scale=(s, s, s))
        shape_prims.append(prim)

    sim.update()

    # Render product + writer
    res = tuple(cfg.get("resolution", (640, 480)))
    rp = rep.create.render_product(cam.GetPath(), res)
    writer = rep.WriterRegistry.get(cfg.get("writer_type", "CocoWriter"))
    wargs = cfg.get("writer_kwargs", {})
    out_dir = wargs.get("output_dir", "_out")
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(os.getcwd(), out_dir)
    wargs["output_dir"] = out_dir
    writer.initialize(**wargs)
    writer.attach([rp])

    # Capture loop
    num_frames = cfg.get("num_frames", 10)
    for i in range(num_frames):
        # Floor swap
        if floors:
            floor_refs.ClearReferences()
            floor_refs.AddReference(random.choice(floors))

        # Camera set
        cam_z = random.uniform(cfg.get("camera_distance_to_target_min_max", [4, 6])[0],
                               cfg.get("camera_distance_to_target_min_max", [4, 6])[1])
        set_transform(cam, loc=(0, 0, cam_z + object_z), rot=(0, 0, 0))

        # Light update
        key_int = random.uniform(800, 1500)
        fill_ratio = random.uniform(0.3, 0.5)
        r, g, b = [random.uniform(0.6, 1.0) for _ in range(3)]
        key.GetAttribute("inputs:intensity").Set(key_int)
        key.GetAttribute("inputs:color").Set(Gf.Vec3f(r, g, b))
        fill.GetAttribute("inputs:intensity").Set(key_int * fill_ratio)
        fill.GetAttribute("inputs:color").Set(Gf.Vec3f(r + 0.1, g + 0.1, b + 0.1))

        # Randomize labeled positions
        poses = random_pose_2d(len(labeled_prims), radius=0.12, min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y, z=object_z)
        for prim, (x, y, z) in zip(labeled_prims, poses):
            set_transform(prim, loc=(x, y, z), rot=(random.uniform(0, 360), 0, random.uniform(0, 360)))

        # Randomize distractors
        for prim in shape_prims:
            set_transform(prim, loc=(random.uniform(min_x, max_x), random.uniform(min_y, max_y), object_z - 0.3),
                          rot=(0, 0, random.uniform(0, 360)))

        # Randomize MDL materials occasionally
        if mdl_mats and i % 3 == 0:
            for prim in labeled_prims:
                UsdShade.MaterialBindingAPI(prim).Bind(random.choice(mdl_mats))

        # Capture
        rep.orchestrator.step()

    rep.orchestrator.wait_until_complete()
    sim.close()

    # Convert to YOLO (optional output dir + split)
    convert_coco_to_yolo(out_dir,
                         cfg.get("yolo_output_dir"),
                         cfg.get("yolo_split_ratio", 0.8),
                         cfg.get("yolo_split_seed"))


if __name__ == "__main__":
    main()
