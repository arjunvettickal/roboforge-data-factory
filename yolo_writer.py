# yolo_writer.py — YOLO detection writer (debug build)
from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import omni.ext
import omni.replicator.core as rep

# ---------- small helpers ----------

def _pjoin(*parts: str) -> str:
    cleaned = [p.strip("/\\") for p in parts if p is not None and p != ""]
    return "/".join(cleaned)

def _as_numpy(a) -> np.ndarray:
    if a is None:
        return np.array([], dtype=np.float32)
    if isinstance(a, np.ndarray):
        return a.reshape(-1)
    return np.asarray(a).reshape(-1)

def _first_not_none(*vals):
    for v in vals:
        if v is not None:
            return v
    return None


class YoloWriter(rep.WriterBase):
    def __init__(self, **kwargs):
        super().__init__()
        self._root: str = "."
        self._split: str = "train"
        self._img_ext: str = "jpg"
        self._class_names: List[str] = []
        self._class_to_id: Dict[str, int] = {}
        self._wrote_yaml: bool = False
        self._img_index: int = 0
        self._write_yaml: bool = True

    def initialize(
        self,
        output_dir: str,
        split: str = "train",
        class_names: Optional[List[str]] = None,
        image_output_format: str = "jpg",
        write_data_yaml: bool = True,
        **_
    ):
        self._root = output_dir
        self._split = split
        self._img_ext = image_output_format.lower()
        self._class_names = list(class_names or [])
        self._class_to_id = {name: i for i, name in enumerate(self._class_names)}
        self._write_yaml = bool(write_data_yaml)

        self._backend.make_dir(_pjoin("images", self._split))
        self._backend.make_dir(_pjoin("labels", self._split))
        print(f"[YOLO] initialize: root={self._root} split={self._split} classes={self._class_names}")

    def attach(self, render_product_paths: List[str]):
        print(f"[YOLO] attach: RPs={render_product_paths}")
        return super().attach(render_product_paths)

    def _stem(self) -> str:
        s = f"{self._img_index:06d}"
        self._img_index += 1
        return s

    @staticmethod
    def _image_wh(rgb_payload: Dict[str, Any]) -> Tuple[int, int]:
        img = rgb_payload.get("data")
        return (img.shape[1], img.shape[0]) if img is not None else (0, 0)

    @staticmethod
    def _xyxy_to_yolo(x1, y1, x2, y2, W, H) -> Tuple[float, float, float, float]:
        x1 = max(0.0, min(float(x1), W - 1))
        x2 = max(0.0, min(float(x2), W - 1))
        y1 = max(0.0, min(float(y1), H - 1))
        y2 = max(0.0, min(float(y2), H - 1))
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        if w <= 0.0 or h <= 0.0 or W <= 0 or H <= 0:
            return 0.0, 0.0, 0.0, 0.0
        cx = (x1 + x2) / 2.0 / W
        cy = (y1 + y2) / 2.0 / H
        return cx, cy, (w / W), (h / H)

    def _resolve_class_id(self, maybe_names: List[str]) -> Optional[int]:
        for name in maybe_names:
            if name in self._class_to_id:
                return self._class_to_id[name]
        return None

    @staticmethod
    def _extract_bbox_arrays(bbox_data: Any) -> Dict[str, np.ndarray]:
        out = {"x_min": None, "y_min": None, "x_max": None, "y_max": None,
               "bbox_ids": None, "class_ids": None}
        if isinstance(bbox_data, dict):
            out["x_min"]    = _first_not_none(bbox_data.get("x_min"), bbox_data.get("xmin"), bbox_data.get("xMin"))
            out["y_min"]    = _first_not_none(bbox_data.get("y_min"), bbox_data.get("ymin"), bbox_data.get("yMin"))
            out["x_max"]    = _first_not_none(bbox_data.get("x_max"), bbox_data.get("xmax"), bbox_data.get("xMax"))
            out["y_max"]    = _first_not_none(bbox_data.get("y_max"), bbox_data.get("ymax"), bbox_data.get("yMax"))
            out["bbox_ids"] = _first_not_none(bbox_data.get("ids"), bbox_data.get("prim_ids"), bbox_data.get("bboxIds"))
            out["class_ids"]= _first_not_none(bbox_data.get("semanticId"), bbox_data.get("classId"), bbox_data.get("class_ids"))
        else:
            arr = bbox_data
            names = list(getattr(arr, "dtype", None).names or [])
            def col(*choices):
                for c in choices:
                    if c in names:
                        return arr[c]
                return None
            if names:
                out["x_min"]     = col("x_min","xmin","xMin")
                out["y_min"]     = col("y_min","ymin","yMin")
                out["x_max"]     = col("x_max","xmax","xMax")
                out["y_max"]     = col("y_max","ymax","yMax")
                out["bbox_ids"]  = col("ids","prim_ids","bboxIds")
                out["class_ids"] = col("semanticId","classId","class_ids")
        for k in out:
            out[k] = _as_numpy(out[k])
        return out

    def write(self, data: Dict[str, Any]) -> None:
        print(f"[YOLO] write called, keys in data: {list(data.keys())}")
        
        rgb = data.get("rgb")
        bb  = data.get("bounding_box_2d_tight") or {}
        
        if rgb is None or "data" not in rgb:
            print("[YOLO] ERROR: No RGB data found")
            return
            
        if "data" not in bb:
            print("[YOLO] ERROR: No bounding box data found")
            print(f"[YOLO] Bounding box keys: {list(bb.keys())}")
            return

        arrs = self._extract_bbox_arrays(bb["data"])
        xmins = arrs["x_min"]; ymins = arrs["y_min"]
        xmaxs = arrs["x_max"]; ymaxs = arrs["y_max"]
        bbox_ids = arrs["bbox_ids"]
        class_ids = arrs["class_ids"]

        n = int(min(len(xmins), len(ymins), len(xmaxs), len(ymaxs)))
        info = bb.get("info", {}) or {}
        labs = info.get("labels", {})
        
        print(f"[YOLO] Found {n} bounding boxes")
        print(f"[YOLO] bbox_ids: {bbox_ids}")
        print(f"[YOLO] class_ids: {class_ids}")
        print(f"[YOLO] labels type: {type(labs)}, content: {labs}")
        print(f"[YOLO] info keys: {list(info.keys())}")

        stem = self._stem()
        img_rel = _pjoin("images", self._split, f"{stem}.{self._img_ext}")
        lbl_rel = _pjoin("labels", self._split, f"{stem}.txt")

        # Write image
        self._backend.make_dir(_pjoin("images", self._split))
        self._backend.write_image(img_rel, rgb)

        # If no boxes, write empty label file and return
        if n == 0:
            self._backend.make_dir(_pjoin("labels", self._split))
            self._backend.write_text(lbl_rel, "")
            print(f"[YOLO] No boxes found, wrote empty label: {lbl_rel}")
            self._maybe_write_yaml()
            return

        # ---- Resolve class names ----
        labels_per_box: List[Optional[str]] = [None] * n

        # Case 1: labels is a dict keyed by bbox ID
        if isinstance(labs, dict) and bbox_ids.size >= n:
            print("[YOLO] Using label resolution method 1 (dict by bbox ID)")
            for i in range(n):
                key = str(int(bbox_ids[i]))
                val = labs.get(key)
                picked = None
                if isinstance(val, dict):
                    picked = val.get("class") or next(iter(val.values()), None)
                elif isinstance(val, str):
                    picked = val
                labels_per_box[i] = picked
                print(f"[YOLO] Box {i}, bbox_id {key} -> label: {picked}")

        # Case 2: labels is a flat list
        if all(v is None for v in labels_per_box) and isinstance(labs, list) and len(labs) >= n:
            print("[YOLO] Using label resolution method 2 (flat list)")
            labels_per_box = labs[:n]
            for i in range(n):
                print(f"[YOLO] Box {i} -> label: {labels_per_box[i]}")

        # Case 3: idToLabels mapping
        if all(v is None for v in labels_per_box):
            print("[YOLO] Using label resolution method 3 (idToLabels)")
            id_to_labels = info.get("idToLabels", {}) or {}
            for i in range(n):
                key = str(int(bbox_ids[i])) if i < len(bbox_ids) else str(i)
                val = id_to_labels.get(key, [])
                picked = None
                if isinstance(val, list):
                    for item in val:
                        if isinstance(item, dict) and item.get("semanticType") == "class":
                            picked = item.get("value"); break
                        if isinstance(item, str):
                            picked = item; break
                elif isinstance(val, dict):
                    picked = val.get("class") or next(iter(val.values()), None)
                labels_per_box[i] = picked
                print(f"[YOLO] Box {i}, id {key} -> label: {picked}")

        # Write labels
        self._backend.make_dir(_pjoin("labels", self._split))
        W, H = self._image_wh(rgb)
        lines: List[str] = []
        
        for i in range(n):
            cname = labels_per_box[i]
            if cname is None:
                print(f"[YOLO] WARNING: No class name found for box {i}")
                continue
                
            cls_id = self._resolve_class_id([cname])
            if cls_id is None:
                print(f"[YOLO] WARNING: Class '{cname}' not in class_names {self._class_names}")
                continue
                
            cx, cy, ww, hh = self._xyxy_to_yolo(xmins[i], ymins[i], xmaxs[i], ymaxs[i], W, H)
            if ww <= 0.0 or hh <= 0.0:
                print(f"[YOLO] WARNING: Invalid bbox dimensions for box {i}")
                continue
                
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}")
            print(f"[YOLO] Box {i}: class={cname}({cls_id}), bbox=({cx:.3f},{cy:.3f},{ww:.3f},{hh:.3f})")

        # Write label file
        label_content = "\n".join(lines) + ("\n" if lines else "")
        self._backend.write_text(lbl_rel, label_content)
        print(f"[YOLO] Wrote {len(lines)} labels to: {lbl_rel}")
        print(f"[YOLO] Label content: {label_content}")

        self._maybe_write_yaml()

    def _maybe_write_yaml(self):
        if self._wrote_yaml or not self._write_yaml:
            return
        yaml_rel = "data.yaml"
        names = "\n".join([f"  {i}: {n}" for i, n in enumerate(self._class_names)])
        content = (
            f"path: {self._root}\n"
            f"train: images/{self._split}\n"
            f"val: images/{self._split}\n"
            f"names:\n{names}\n"
        )
        self._backend.write_text(yaml_rel, content)
        self._wrote_yaml = True
        print("[YOLO] wrote data.yaml")

# ---------- registration footer ----------
try:
    rep.WriterRegistry.register_writer("YoloWriter", YoloWriter)
except Exception:
    try:
        rep.WriterRegistry.register(YoloWriter)
    except Exception:
        pass

class Extension(omni.ext.IExt):
    def on_startup(self, ext_id):
        try:
            rep.WriterRegistry.register_writer("YoloWriter", YoloWriter)
        except Exception:
            try:
                rep.WriterRegistry.register(YoloWriter)
            except Exception:
                pass
    def on_shutdown(self): pass