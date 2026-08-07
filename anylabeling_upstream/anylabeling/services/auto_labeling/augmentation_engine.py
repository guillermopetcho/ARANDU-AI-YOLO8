"""
Data Augmentation Engine for AnyLabeling (Object Detection & Instance Segmentation)
Inspired by Albumentations, Roboflow, and CVAT.
Provides geometric and photometric transformations on images and polygon / bbox coordinates.
"""

import os
import math
import json
import random
import copy
import numpy as np
import PIL.Image
import PIL.ImageEnhance
import PIL.ImageOps
import PIL.ImageFilter
import cv2


class AugmentationPreset:
    LIGHT = "light"
    ROBOFLOW_STANDARD = "roboflow_standard"
    HEAVY = "heavy"
    CUSTOM = "custom"


class AugmentationEngine:
    """Robust Data Augmentation Engine for images and annotation shapes."""

    def __init__(self, config=None):
        self.config = config or self.get_default_config(AugmentationPreset.ROBOFLOW_STANDARD)

    @staticmethod
    def get_default_config(preset=AugmentationPreset.ROBOFLOW_STANDARD):
        if preset == AugmentationPreset.LIGHT:
            return {
                "preset": AugmentationPreset.LIGHT,
                "flip_h": {"enabled": True, "prob": 0.5},
                "flip_v": {"enabled": False, "prob": 0.0},
                "rotation": {"enabled": True, "prob": 0.5, "range": [-15, 15]},
                "brightness": {"enabled": True, "prob": 0.5, "factor_range": [0.85, 1.15]},
                "contrast": {"enabled": True, "prob": 0.5, "factor_range": [0.85, 1.15]},
                "saturation": {"enabled": False, "prob": 0.0, "factor_range": [0.8, 1.2]},
                "sharpness": {"enabled": False, "prob": 0.0, "factor_range": [0.8, 1.2]},
                "blur": {"enabled": False, "prob": 0.0, "max_radius": 1.5},
                "noise": {"enabled": False, "prob": 0.0, "std_range": [5, 20]},
                "crop": {"enabled": False, "prob": 0.0, "scale_range": [0.85, 1.0]},
                "grayscale": {"enabled": False, "prob": 0.0},
            }
        elif preset == AugmentationPreset.HEAVY:
            return {
                "preset": AugmentationPreset.HEAVY,
                "flip_h": {"enabled": True, "prob": 0.5},
                "flip_v": {"enabled": True, "prob": 0.3},
                "rotation": {"enabled": True, "prob": 0.7, "range": [-45, 45]},
                "brightness": {"enabled": True, "prob": 0.7, "factor_range": [0.7, 1.3]},
                "contrast": {"enabled": True, "prob": 0.7, "factor_range": [0.7, 1.3]},
                "saturation": {"enabled": True, "prob": 0.5, "factor_range": [0.6, 1.4]},
                "sharpness": {"enabled": True, "prob": 0.5, "factor_range": [0.5, 2.0]},
                "blur": {"enabled": True, "prob": 0.4, "max_radius": 2.5},
                "noise": {"enabled": True, "prob": 0.4, "std_range": [10, 35]},
                "crop": {"enabled": True, "prob": 0.5, "scale_range": [0.75, 0.95]},
                "grayscale": {"enabled": True, "prob": 0.2},
            }
        else:  # ROBOFLOW_STANDARD
            return {
                "preset": AugmentationPreset.ROBOFLOW_STANDARD,
                "flip_h": {"enabled": True, "prob": 0.5},
                "flip_v": {"enabled": False, "prob": 0.0},
                "rotation": {"enabled": True, "prob": 0.5, "range": [-25, 25]},
                "brightness": {"enabled": True, "prob": 0.6, "factor_range": [0.8, 1.2]},
                "contrast": {"enabled": True, "prob": 0.6, "factor_range": [0.8, 1.2]},
                "saturation": {"enabled": True, "prob": 0.4, "factor_range": [0.8, 1.2]},
                "sharpness": {"enabled": True, "prob": 0.3, "factor_range": [0.8, 1.5]},
                "blur": {"enabled": True, "prob": 0.3, "max_radius": 2.0},
                "noise": {"enabled": True, "prob": 0.3, "std_range": [5, 25]},
                "crop": {"enabled": True, "prob": 0.4, "scale_range": [0.8, 0.98]},
                "grayscale": {"enabled": False, "prob": 0.0},
            }

    def apply_transformations(self, pil_img, shapes):
        """
        Apply a random augmentation pass according to self.config to PIL Image and shapes.
        Returns:
            aug_pil_img (PIL.Image): Augmented PIL image.
            aug_shapes (list of dict): Transformed shapes with updated coordinates.
        """
        if pil_img is None:
            return None, []

        img = pil_img.copy()
        w, h = img.size
        curr_shapes = copy.deepcopy(shapes)

        # 1. Flip Horizontal
        cfg_fh = self.config.get("flip_h", {})
        if cfg_fh.get("enabled") and random.random() < cfg_fh.get("prob", 0.5):
            img = img.transpose(PIL.Image.Transpose.FLIP_LEFT_RIGHT)
            for shape in curr_shapes:
                new_points = []
                for p in shape.get("points", []):
                    nx = w - p[0]
                    ny = p[1]
                    new_points.append([nx, ny])
                shape["points"] = self._normalize_points(shape.get("shape_type"), new_points)

        # 2. Flip Vertical
        cfg_fv = self.config.get("flip_v", {})
        if cfg_fv.get("enabled") and random.random() < cfg_fv.get("prob", 0.5):
            img = img.transpose(PIL.Image.Transpose.FLIP_TOP_BOTTOM)
            for shape in curr_shapes:
                new_points = []
                for p in shape.get("points", []):
                    nx = p[0]
                    ny = h - p[1]
                    new_points.append([nx, ny])
                shape["points"] = self._normalize_points(shape.get("shape_type"), new_points)

        # 3. Random Rotation
        cfg_rot = self.config.get("rotation", {})
        if cfg_rot.get("enabled") and random.random() < cfg_rot.get("prob", 0.5):
            angle_min, angle_max = cfg_rot.get("range", [-15, 15])
            angle = random.uniform(angle_min, angle_max)
            img, curr_shapes = self._rotate_img_and_shapes(img, curr_shapes, angle)

        # 4. Random Crop & Scale
        w, h = img.size
        cfg_crop = self.config.get("crop", {})
        if cfg_crop.get("enabled") and random.random() < cfg_crop.get("prob", 0.5):
            scale_min, scale_max = cfg_crop.get("scale_range", [0.8, 0.95])
            scale = random.uniform(scale_min, scale_max)
            new_w = max(10, int(w * scale))
            new_h = max(10, int(h * scale))

            # Random crop origin
            x0 = random.randint(0, w - new_w)
            y0 = random.randint(0, h - new_h)

            img = img.crop((x0, y0, x0 + new_w, y0 + new_h))
            img = img.resize((w, h), PIL.Image.Resampling.BILINEAR)

            # Adjust shape points
            valid_shapes = []
            for shape in curr_shapes:
                new_points = []
                for p in shape.get("points", []):
                    # Scale coordinates from cropped area back to full image bounds
                    nx = (p[0] - x0) * (w / float(new_w))
                    ny = (p[1] - y0) * (h / float(new_h))
                    new_points.append([nx, ny])

                shape["points"] = self._clip_and_normalize(shape.get("shape_type"), new_points, w, h)
                if self._is_valid_shape(shape, w, h):
                    valid_shapes.append(shape)
            curr_shapes = valid_shapes

        # 5. Photometric Adjustments (Brightness, Contrast, Saturation, Sharpness)
        cfg_br = self.config.get("brightness", {})
        if cfg_br.get("enabled") and random.random() < cfg_br.get("prob", 0.5):
            f_min, f_max = cfg_br.get("factor_range", [0.8, 1.2])
            factor = random.uniform(f_min, f_max)
            img = PIL.ImageEnhance.Brightness(img).enhance(factor)

        cfg_ct = self.config.get("contrast", {})
        if cfg_ct.get("enabled") and random.random() < cfg_ct.get("prob", 0.5):
            f_min, f_max = cfg_ct.get("factor_range", [0.8, 1.2])
            factor = random.uniform(f_min, f_max)
            img = PIL.ImageEnhance.Contrast(img).enhance(factor)

        cfg_st = self.config.get("saturation", {})
        if cfg_st.get("enabled") and random.random() < cfg_st.get("prob", 0.5):
            f_min, f_max = cfg_st.get("factor_range", [0.8, 1.2])
            factor = random.uniform(f_min, f_max)
            img = PIL.ImageEnhance.Color(img).enhance(factor)

        cfg_sh = self.config.get("sharpness", {})
        if cfg_sh.get("enabled") and random.random() < cfg_sh.get("prob", 0.5):
            f_min, f_max = cfg_sh.get("factor_range", [0.8, 1.5])
            factor = random.uniform(f_min, f_max)
            img = PIL.ImageEnhance.Sharpness(img).enhance(factor)

        # 6. Gaussian Blur
        cfg_blur = self.config.get("blur", {})
        if cfg_blur.get("enabled") and random.random() < cfg_blur.get("prob", 0.3):
            max_r = cfg_blur.get("max_radius", 2.0)
            radius = random.uniform(0.5, max_r)
            img = img.filter(PIL.ImageFilter.GaussianBlur(radius))

        # 7. Gaussian Noise
        cfg_noise = self.config.get("noise", {})
        if cfg_noise.get("enabled") and random.random() < cfg_noise.get("prob", 0.3):
            std_min, std_max = cfg_noise.get("std_range", [5, 25])
            std = random.uniform(std_min, std_max)
            arr = np.array(img).astype(np.float32)
            noise = np.random.normal(0, std, arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            img = PIL.Image.fromarray(arr)

        # 8. Grayscale
        cfg_gray = self.config.get("grayscale", {})
        if cfg_gray.get("enabled") and random.random() < cfg_gray.get("prob", 0.1):
            img = PIL.ImageOps.grayscale(img).convert("RGB")

        # Final pass: Ensure all shapes are clipped to image bounds
        w, h = img.size
        final_shapes = []
        for shape in curr_shapes:
            shape["points"] = self._clip_and_normalize(shape.get("shape_type"), shape.get("points", []), w, h)
            if self._is_valid_shape(shape, w, h):
                final_shapes.append(shape)

        return img, final_shapes

    def _rotate_img_and_shapes(self, img, shapes, angle):
        """Rotate PIL Image and transform shape points around center."""
        w, h = img.size
        cx, cy = w / 2.0, h / 2.0

        # Rotate image with expansion to avoid clipping content
        rotated_img = img.rotate(-angle, resample=PIL.Image.Resampling.BILINEAR, expand=True)
        nw, nh = rotated_img.size
        ncx, ncy = nw / 2.0, nh / 2.0

        rad = math.radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        new_shapes = []
        for shape in shapes:
            new_points = []
            for p in shape.get("points", []):
                x = p[0] - cx
                y = p[1] - cy
                # Standard 2D rotation formula around center
                rx = x * cos_a - y * sin_a
                ry = x * sin_a + y * cos_a
                nx = rx + ncx
                ny = ry + ncy
                new_points.append([nx, ny])

            shape_copy = copy.deepcopy(shape)
            shape_copy["points"] = self._normalize_points(shape.get("shape_type"), new_points)
            new_shapes.append(shape_copy)

        # Resize rotated image back to original dimensions (w, h)
        final_img = rotated_img.resize((w, h), PIL.Image.Resampling.BILINEAR)

        # Scale shape coordinates from expanded (nw, nh) back to (w, h)
        scale_x = w / float(nw)
        scale_y = h / float(nh)

        for shape in new_shapes:
            scaled_points = []
            for p in shape.get("points", []):
                scaled_points.append([p[0] * scale_x, p[1] * scale_y])
            shape["points"] = self._clip_and_normalize(shape.get("shape_type"), scaled_points, w, h)

        return final_img, new_shapes

    def _normalize_points(self, shape_type, points):
        if shape_type == "rectangle" and len(points) == 2:
            p1, p2 = points[0], points[1]
            return [
                [min(p1[0], p2[0]), min(p1[1], p2[1])],
                [max(p1[0], p2[0]), max(p1[1], p2[1])],
            ]
        return points

    def _clip_and_normalize(self, shape_type, points, w, h):
        clipped = []
        for p in points:
            cx = max(0.0, min(float(w), float(p[0])))
            cy = max(0.0, min(float(h), float(p[1])))
            clipped.append([cx, cy])

        return self._normalize_points(shape_type, clipped)

    def _is_valid_shape(self, shape, w, h):
        points = shape.get("points", [])
        stype = shape.get("shape_type", "polygon")

        if not points:
            return False

        if stype == "rectangle":
            if len(points) < 2:
                return False
            dx = abs(points[1][0] - points[0][0])
            dy = abs(points[1][1] - points[0][1])
            return (dx * dy) >= 4.0

        elif stype == "polygon":
            if len(points) < 3:
                return False
            # Approximate polygon area using Shoelace formula
            area = 0.0
            n = len(points)
            for i in range(n):
                j = (i + 1) % n
                area += points[i][0] * points[j][1]
                area -= points[j][0] * points[i][1]
            return (abs(area) * 0.5) >= 10.0

        return True
