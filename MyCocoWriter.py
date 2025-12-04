import os, io, json, itertools
import numpy as np
import omni.kit
import omni.replicator.core as rep
from omni.replicator.core import Writer, AnnotatorRegistry, BackendDispatch

class MyCocoWriter(Writer):
    def __init__(self, output_dir, image_output_format="png", categories=None):
        self._out_dir = output_dir
        self._backend = BackendDispatch({"paths": {"out_dir": output_dir}})
        self._fmt = image_output_format
        self._frame_idx = 0

        # --- COCO accumulators ---
        self.images = []
        self.annotations = []
        # categories mapping: [{"id": 1, "name": "part_name", "supercategory": "part"}]
        if categories is None:
            # Minimal example; replace with your project’s categories
            categories = [
                {"id": 1, "name": "SM_bonnet", "supercategory": "part"},
                {"id": 2, "name": "SM_seat", "supercategory": "part"},
            ]
        self.categories = categories
        # name->id lookup
        self.cat_name_to_id = {c["name"]: c["id"] for c in self.categories}

        # IDs
        self._next_ann_id = itertools.count(1)
        self._next_img_id = itertools.count(1)

        # --- declare annotators we will consume in write() ---
        self.annotators = [
            AnnotatorRegistry.get_annotator("rgb"),
            # tight 2D bboxes for visible pixels only
            AnnotatorRegistry.get_annotator(
                "bounding_box_2d_tight",
                init_params={"semanticTypes": ["class"]}  # only "class" labels
            ),
        ]

    def _coco_bbox_from_xyxy(self, x_min, y_min, x_max, y_max):
        x = float(x_min)
        y = float(y_min)
        w = float(max(0.0, x_max - x_min))
        h = float(max(0.0, y_max - y_min))
        return [x, y, w, h], w * h

    def write(self, data):
        # 1) Save image
        rgba = data["rgb"]
        H, W = rgba.shape[0], rgba.shape[1]
        img_file = f"{self._frame_idx:06d}.{self._fmt}"
        self._backend.write_image(os.path.join("images", img_file), rgba)

        image_id = next(self._next_img_id)
        self.images.append({
            "id": image_id,
            "file_name": img_file,
            "width": W,
            "height": H,
        })

        # 2) Read bboxes + labels
        if "bounding_box_2d_tight" in data:
            bbox = data["bounding_box_2d_tight"]["data"]
            # per-prim mapping {"<instance_id>": {"class": "SM_bonnet"} ...}
            id_to_labels = data["bounding_box_2d_tight"]["info"]["idToLabels"]

            # Each bbox array is shaped (N,) for x_min/x_max/y_min/y_max
            xmins, ymins, xmaxs, ymaxs = bbox["x_min"], bbox["y_min"], bbox["x_max"], bbox["y_max"]

            for i in range(len(xmins)):
                # Skip invalid boxes (guard for sentinel values)
                if np.isinf([xmins[i], ymins[i], xmaxs[i], ymaxs[i]]).any():
                    continue

                # Find category name for this instance id
                inst_id = str(bbox["id"][i]) if "id" in bbox else None
                labels = id_to_labels.get(inst_id, {}) if inst_id else {}
                # Prefer the "class" semantic
                cat_name = labels.get("class") if isinstance(labels, dict) else None
                if not cat_name:
                    # If your assets use different semantic keys, adjust here (e.g., "coco", "thing")
                    continue
                if cat_name not in self.cat_name_to_id:
                    # Unknown class → skip or map to a default. Better: add it to categories.
                    continue

                coco_bbox, area = self._coco_bbox_from_xyxy(xmins[i], ymins[i], xmaxs[i], ymaxs[i])
                self.annotations.append({
                    "id": next(self._next_ann_id),
                    "image_id": image_id,
                    "category_id": self.cat_name_to_id[cat_name],
                    "bbox": [round(v, 2) for v in coco_bbox],
                    "area": round(float(area), 2),
                    "iscrowd": 0,
                    # Optional polygon/RLE segmentation can be added later
                })

        # 3) Optionally stream a rolling JSON for debugging
        partial = {
            "images": self.images,
            "annotations": self.annotations,
            "categories": self.categories,
            "licenses": [],
            "info": {"version": "1.0", "description": "My Replicator COCO"}
        }
        blob = io.BytesIO(json.dumps(partial, indent=2).encode("utf-8"))
        self._backend.write_blob("annotations_coco.json", blob.getvalue())

        self._frame_idx += 1

# --- Hook the writer to a render product ---
# Example scene omitted; just show attach:
# render_product = rep.create.render_product(camera, (1024, 1024))
# writer = rep.WriterRegistry.get("MyCocoWriter")
# writer.initialize(output_dir="out_coco", image_output_format="png")
# writer.attach([render_product])
