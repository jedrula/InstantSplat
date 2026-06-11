import json
import os
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import tqdm as _tqdm_module
from PIL import Image
try:
    from pycolmap import SceneManager
except (ImportError, AttributeError):
    # pycolmap >=4.0 dropped SceneManager; provide a compatibility shim
    import pycolmap as _pycolmap
    import numpy as _np

    class _ImgWrap:
        def __init__(self, img):
            self._img = img
        def R(self):
            return self._img.cam_from_world().rotation.matrix()
        @property
        def tvec(self):
            return self._img.cam_from_world().translation
        def __getattr__(self, name):
            if name == '_img':
                raise AttributeError(name)
            return getattr(self._img, name)

    class _CamWrap:
        _LAYOUT = {
            0: [('fx',0),('fy',0),('cx',1),('cy',2)],
            1: [('fx',0),('fy',1),('cx',2),('cy',3)],
            2: [('fx',0),('fy',0),('cx',1),('cy',2),('k1',3)],
            3: [('fx',0),('fy',0),('cx',1),('cy',2),('k1',3),('k2',4)],
            4: [('fx',0),('fy',1),('cx',2),('cy',3),('k1',4),('k2',5),('p1',6),('p2',7)],
            5: [('fx',0),('fy',1),('cx',2),('cy',3),('k1',4),('k2',5),('k3',6),('k4',7)],
        }
        def __init__(self, cam):
            self._cam = cam
            m = cam.model
            model_id = m.value if hasattr(m, 'value') else int(m)
            p = cam.params
            d = {name: float(p[idx]) for name, idx in self._LAYOUT.get(model_id, [])}
            for attr in ('k1', 'k2', 'k3', 'k4', 'p1', 'p2'):
                d.setdefault(attr, 0.0)
            self.__dict__.update(d)
            self.camera_type = model_id
        def __getattr__(self, name):
            if name == '_cam':
                raise AttributeError(name)
            return getattr(self._cam, name)

    class SceneManager:
        def __init__(self, path):
            self._r = _pycolmap.Reconstruction(path)
        def load_cameras(self): pass
        def load_images(self): pass
        def load_points3D(self): pass
        @property
        def cameras(self):
            return {cid: _CamWrap(c) for cid, c in self._r.cameras.items()}
        @property
        def images(self):
            return {iid: _ImgWrap(img) for iid, img in self._r.images.items()}
        @property
        def points3D(self):
            pts = list(self._r.points3D.values())
            return _np.array([p.xyz for p in pts], dtype=_np.float64) if pts else _np.zeros((0, 3))
        @property
        def point3D_colors(self):
            pts = list(self._r.points3D.values())
            return _np.array([p.color[:3] for p in pts], dtype=_np.uint8) if pts else _np.zeros((0, 3), dtype=_np.uint8)
        @property
        def point3D_errors(self):
            pts = list(self._r.points3D.values())
            return _np.array([p.error for p in pts], dtype=_np.float64) if pts else _np.zeros(0)
        @property
        def name_to_image_id(self):
            return {img.name: iid for iid, img in self._r.images.items()}
        @property
        def point3D_id_to_images(self):
            return {pid: [(el.image_id, el.point2D_idx) for el in pt.track.elements]
                    for pid, pt in self._r.points3D.items()}
        @property
        def point3D_id_to_point3D_idx(self):
            return {pid: idx for idx, pid in enumerate(self._r.points3D.keys())}
from tqdm import tqdm
from typing_extensions import assert_never

from .normalize import (
    align_principal_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)


def _qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    """Convert COLMAP qvec [qw, qx, qy, qz] to rotation matrix."""
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


class _FastCamera:
    """Minimal camera object compatible with parser expectations."""

    def __init__(self, camera_id: int, model_id: int, width: int, height: int, params: np.ndarray):
        self.camera_id = camera_id
        self.camera_type = model_id
        self.width = width
        self.height = height

        # Defaults for non-distorted and unknown models.
        self.fx = self.fy = float(params[0]) if len(params) > 0 else 1.0
        self.cx = float(params[1]) if len(params) > 1 else width / 2.0
        self.cy = float(params[2]) if len(params) > 2 else height / 2.0
        self.k1 = self.k2 = self.k3 = self.k4 = self.p1 = self.p2 = 0.0

        # Match the camera model handling used in this parser.
        if model_id in (0, 2, 3):  # SIMPLE_PINHOLE, SIMPLE_RADIAL, RADIAL
            self.fx = self.fy = float(params[0])
            self.cx = float(params[1])
            self.cy = float(params[2])
            if len(params) > 3:
                self.k1 = float(params[3])
            if len(params) > 4:
                self.k2 = float(params[4])
        elif model_id in (1, 4, 5):  # PINHOLE, OPENCV, OPENCV_FISHEYE
            self.fx = float(params[0])
            self.fy = float(params[1])
            self.cx = float(params[2])
            self.cy = float(params[3])
            if len(params) > 4:
                self.k1 = float(params[4])
            if len(params) > 5:
                self.k2 = float(params[5])
            if model_id == 4 and len(params) > 7:
                self.p1 = float(params[6])
                self.p2 = float(params[7])
            if model_id == 5:
                if len(params) > 6:
                    self.k3 = float(params[6])
                if len(params) > 7:
                    self.k4 = float(params[7])


class _FastImage:
    """Minimal image object compatible with parser expectations."""

    def __init__(self, image_id: int, qvec: np.ndarray, tvec: np.ndarray, camera_id: int, name: str):
        self.image_id = image_id
        self.qvec = qvec
        self.tvec = tvec
        self.camera_id = camera_id
        self.name = name

    def R(self):
        return _qvec_to_rotmat(self.qvec)


def _load_colmap_cameras_images_fast(colmap_dir: str):
    """Load only cameras and images from COLMAP binaries, skipping point tracks."""
    cameras_bin = os.path.join(colmap_dir, "cameras.bin")
    images_bin = os.path.join(colmap_dir, "images.bin")
    if not os.path.exists(cameras_bin) or not os.path.exists(images_bin):
        raise FileNotFoundError("fast_init requires cameras.bin and images.bin")

    camera_model_num_params = {
        0: 3,  # SIMPLE_PINHOLE
        1: 4,  # PINHOLE
        2: 4,  # SIMPLE_RADIAL
        3: 5,  # RADIAL
        4: 8,  # OPENCV
        5: 8,  # OPENCV_FISHEYE
        6: 12, 7: 5, 8: 4, 9: 5, 10: 12, 11: 12,
    }

    cameras = {}
    with open(cameras_bin, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            camera_id = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]
            n_params = camera_model_num_params.get(model_id, 4)
            params = np.array(struct.unpack(f"<{n_params}d", f.read(8 * n_params)))
            cameras[camera_id] = _FastCamera(camera_id, model_id, width, height, params)

    images = {}
    with open(images_bin, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<I", f.read(4))[0]
            qvec = np.array(struct.unpack("<4d", f.read(32)), dtype=np.float64)
            tvec = np.array(struct.unpack("<3d", f.read(24)), dtype=np.float64)
            camera_id = struct.unpack("<I", f.read(4))[0]

            name_bytes = bytearray()
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_bytes.extend(c)
            name = name_bytes.decode("utf-8", errors="replace")

            n_points2d = struct.unpack("<Q", f.read(8))[0]
            # Skip x(double), y(double), point3D_id(int64) per point.
            f.seek(n_points2d * 24, os.SEEK_CUR)

            images[image_id] = _FastImage(image_id, qvec, tvec, camera_id, name)

    return cameras, images


def _load_colmap_points_fast(colmap_dir: str):
    """Load COLMAP sparse points from points3D.bin while skipping tracks."""
    points_bin = os.path.join(colmap_dir, "points3D.bin")
    if not os.path.exists(points_bin):
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, 3), dtype=np.uint8),
        )

    points = []
    points_err = []
    points_rgb = []
    with open(points_bin, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_points):
            # point3D_id
            _ = struct.unpack("<Q", f.read(8))[0]
            xyz = struct.unpack("<3d", f.read(24))
            rgb = struct.unpack("<3B", f.read(3))
            error = struct.unpack("<d", f.read(8))[0]
            track_len = struct.unpack("<Q", f.read(8))[0]
            # Skip track elements: (image_id:int32, point2D_idx:int32)
            f.seek(track_len * 8, os.SEEK_CUR)

            points.append(xyz)
            points_rgb.append(rgb)
            points_err.append(error)

    if not points:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, 3), dtype=np.uint8),
        )

    return (
        np.asarray(points, dtype=np.float32),
        np.asarray(points_err, dtype=np.float32),
        np.asarray(points_rgb, dtype=np.uint8),
    )


def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths


def _resize_image_folder(image_dir: str, resized_dir: str, factor: int) -> str:
    """Resize image folder."""
    print(f"Downscaling images by {factor}x from {image_dir} to {resized_dir}.")
    os.makedirs(resized_dir, exist_ok=True)

    image_files = _get_rel_paths(image_dir)
    for image_file in tqdm(image_files):
        image_path = os.path.join(image_dir, image_file)
        resized_path = os.path.join(
            resized_dir, os.path.splitext(image_file)[0] + ".png"
        )
        if os.path.isfile(resized_path):
            continue
        image = imageio.imread(image_path)[..., :3]
        resized_size = (
            int(round(image.shape[1] / factor)),
            int(round(image.shape[0] / factor)),
        )
        resized_image = np.array(
            Image.fromarray(image).resize(resized_size, Image.BICUBIC)
        )
        imageio.imwrite(resized_path, resized_image)
    return resized_dir


class Parser:
    """COLMAP parser."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
        fast_init: bool = False,
        mask_dir: Optional[str] = None,
        load_exposure: bool = False,
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every
        self.mask_dir = mask_dir

        colmap_dir = os.path.join(data_dir, "sparse/0/")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(data_dir, "sparse")
        assert os.path.exists(
            colmap_dir
        ), f"COLMAP directory {colmap_dir} does not exist."

        if fast_init:
            cameras_fast, images_fast = _load_colmap_cameras_images_fast(colmap_dir)
            imdata = images_fast
        else:
            manager = SceneManager(colmap_dir)
            manager.load_cameras()
            manager.load_images()
            manager.load_points3D()
            imdata = manager.images

        # Extract extrinsic matrices in world-to-camera format.
        w2c_mats = []
        camera_ids = []
        Ks_dict = dict()
        params_dict = dict()
        imsize_dict = dict()  # width, height
        mask_dict = dict()
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
        for k in imdata:
            im = imdata[k]
            rot = im.R()
            trans = im.tvec.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)

            # support different camera intrinsics
            camera_id = im.camera_id
            camera_ids.append(camera_id)

            # camera intrinsics
            cam = cameras_fast[camera_id] if fast_init else manager.cameras[camera_id]
            fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            K[:2, :] /= factor
            Ks_dict[camera_id] = K

            # Get distortion parameters.
            type_ = cam.camera_type
            if type_ == 0 or type_ == "SIMPLE_PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            elif type_ == 1 or type_ == "PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            if type_ == 2 or type_ == "SIMPLE_RADIAL":
                params = np.array([cam.k1, 0.0, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 3 or type_ == "RADIAL":
                params = np.array([cam.k1, cam.k2, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 4 or type_ == "OPENCV":
                params = np.array([cam.k1, cam.k2, cam.p1, cam.p2], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 5 or type_ == "OPENCV_FISHEYE":
                params = np.array([cam.k1, cam.k2, cam.k3, cam.k4], dtype=np.float32)
                camtype = "fisheye"
            assert (
                camtype == "perspective" or camtype == "fisheye"
            ), f"Only perspective and fisheye cameras are supported, got {type_}"

            params_dict[camera_id] = params
            imsize_dict[camera_id] = (cam.width // factor, cam.height // factor)
            mask_dict[camera_id] = None
        print(
            f"[Parser] {len(imdata)} images, taken by {len(set(camera_ids))} cameras."
        )

        if len(imdata) == 0:
            raise ValueError("No images found in COLMAP.")
        if not (type_ == 0 or type_ == 1):
            print("Warning: COLMAP Camera is not PINHOLE. Images have distortion.")

        w2c_mats = np.stack(w2c_mats, axis=0)

        # Convert extrinsics to camera-to-world.
        camtoworlds = np.linalg.inv(w2c_mats)

        # Image names from COLMAP. No need for permuting the poses according to
        # image names anymore.
        image_names = [imdata[k].name for k in imdata]

        # Previous Nerf results were generated with images sorted by filename,
        # ensure metrics are reported on the same test set.
        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        camtoworlds = camtoworlds[inds]
        camera_ids = [camera_ids[i] for i in inds]

        # Load extended metadata. Used by Bilarf dataset.
        self.extconf = {
            "spiral_radius_scale": 1.0,
            "no_factor_suffix": False,
        }
        extconf_file = os.path.join(data_dir, "ext_metadata.json")
        if os.path.exists(extconf_file):
            with open(extconf_file) as f:
                self.extconf.update(json.load(f))

        # Load bounds if possible (only used in forward facing scenes).
        self.bounds = np.array([0.01, 1.0])
        posefile = os.path.join(data_dir, "poses_bounds.npy")
        if os.path.exists(posefile):
            self.bounds = np.load(posefile)[:, -2:]

        # Load images.
        if factor > 1 and not self.extconf["no_factor_suffix"]:
            image_dir_suffix = f"_{factor}"
        else:
            image_dir_suffix = ""
        colmap_image_dir = os.path.join(data_dir, "images")
        image_dir = os.path.join(data_dir, "images" + image_dir_suffix)
        for d in [image_dir, colmap_image_dir]:
            if not os.path.exists(d):
                raise ValueError(f"Image folder {d} does not exist.")

        # Downsampled images may have different names vs images used for COLMAP,
        # so we need to map between the two sorted lists of files.
        colmap_files = sorted(_get_rel_paths(colmap_image_dir))
        image_files = sorted(_get_rel_paths(image_dir))
        if factor > 1 and os.path.splitext(image_files[0])[1].lower() == ".jpg":
            image_dir = _resize_image_folder(
                colmap_image_dir, image_dir + "_png", factor=factor
            )
            image_files = sorted(_get_rel_paths(image_dir))
        colmap_to_image = dict(zip(colmap_files, image_files))
        image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]

        # 3D points and {image_name -> [point_idx]}
        if fast_init:
            points, points_err, points_rgb = _load_colmap_points_fast(colmap_dir)
            point_indices = {}
        else:
            points = manager.points3D.astype(np.float32)
            points_err = manager.point3D_errors.astype(np.float32)
            points_rgb = manager.point3D_colors.astype(np.uint8)
            point_indices = dict()

            image_id_to_name = {v: k for k, v in manager.name_to_image_id.items()}
            for point_id, data in manager.point3D_id_to_images.items():
                for image_id, _ in data:
                    image_name = image_id_to_name[image_id]
                    point_idx = manager.point3D_id_to_point3D_idx[point_id]
                    point_indices.setdefault(image_name, []).append(point_idx)
            point_indices = {
                k: np.array(v).astype(np.int32) for k, v in point_indices.items()
            }

        # Normalize the world space.
        if normalize:
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            points = transform_points(T1, points)

            # Fast smoke mode can skip COLMAP points entirely. In that case,
            # keep camera-based normalization only and avoid point-based PCA.
            if points.shape[0] >= 3:
                T2 = align_principal_axes(points)
                camtoworlds = transform_cameras(T2, camtoworlds)
                points = transform_points(T2, points)
            else:
                T2 = np.eye(4)

            transform = T2 @ T1

            # Fix for up side down. We assume more points towards
            # the bottom of the scene which is true when ground floor is
            # present in the images.
            if points.shape[0] > 0 and np.median(points[:, 2]) > np.mean(points[:, 2]):
                # rotate 180 degrees around x axis such that z is flipped
                T3 = np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
                camtoworlds = transform_cameras(T3, camtoworlds)
                points = transform_points(T3, points)
                transform = T3 @ transform
        else:
            transform = np.eye(4)

        self.image_names = image_names  # List[str], (num_images,)
        self.image_paths = image_paths  # List[str], (num_images,)
        self.camtoworlds = camtoworlds  # np.ndarray, (num_images, 4, 4)
        self.camera_ids = camera_ids  # List[int], (num_images,)
        self.Ks_dict = Ks_dict  # Dict of camera_id -> K
        self.params_dict = params_dict  # Dict of camera_id -> params
        self.imsize_dict = imsize_dict  # Dict of camera_id -> (width, height)
        self.mask_dict = mask_dict  # Dict of camera_id -> mask (fisheye ROI only)

        # 0-based contiguous camera indices — needed by PPISP and other per-camera modules.
        unique_camera_ids = sorted(set(camera_ids))
        self.camera_id_to_idx = {cid: idx for idx, cid in enumerate(unique_camera_ids)}
        self.camera_indices = [self.camera_id_to_idx[cid] for cid in camera_ids]
        self.num_cameras = len(unique_camera_ids)

        # Per-image masks loaded from mask_dir (e.g. DA3-refined wall masks).
        if mask_dir is not None:
            self.image_masks = []
            for name in image_names:
                p = os.path.join(mask_dir, name)
                if os.path.exists(p):
                    self.image_masks.append(
                        imageio.imread(p).astype(bool)
                    )
                else:
                    self.image_masks.append(None)
            n_loaded = sum(m is not None for m in self.image_masks)
            print(f"[Parser] Loaded {n_loaded}/{len(image_names)} masks from {mask_dir}")
        else:
            self.image_masks = [None] * len(image_names)
        self.points = points  # np.ndarray, (num_points, 3)
        self.points_err = points_err  # np.ndarray, (num_points,)
        self.points_rgb = points_rgb  # np.ndarray, (num_points, 3)
        self.point_indices = point_indices  # Dict[str, np.ndarray], image_name -> [M,]
        self.transform = transform  # np.ndarray, (4, 4)

        # Load EXIF exposure data if requested.
        self.load_exposure = load_exposure
        if load_exposure:
            from exif import compute_exposure_from_exif  # gsplat examples/exif.py
            exposure_values: List[Optional[float]] = []
            for image_name in _tqdm_module.tqdm(image_names, desc="Loading EXIF exposure"):
                original_path = Path(colmap_image_dir) / image_name
                exposure_values.append(compute_exposure_from_exif(original_path))
            valid = [e for e in exposure_values if e is not None]
            if valid:
                exposure_mean = sum(valid) / len(valid)
                self.exposure_values = [
                    (e - exposure_mean) if e is not None else None
                    for e in exposure_values
                ]
            else:
                self.exposure_values = [None] * len(image_names)
        else:
            self.exposure_values = [None] * len(image_names)

        # load one image to check the size. In the case of tanksandtemples dataset, the
        # intrinsics stored in COLMAP corresponds to 2x upsampled images.
        actual_image = imageio.imread(self.image_paths[0])[..., :3]
        actual_height, actual_width = actual_image.shape[:2]
        colmap_width, colmap_height = self.imsize_dict[self.camera_ids[0]]
        s_height, s_width = actual_height / colmap_height, actual_width / colmap_width
        for camera_id, K in self.Ks_dict.items():
            K[0, :] *= s_width
            K[1, :] *= s_height
            self.Ks_dict[camera_id] = K
            width, height = self.imsize_dict[camera_id]
            self.imsize_dict[camera_id] = (int(width * s_width), int(height * s_height))

        # undistortion
        self.mapx_dict = dict()
        self.mapy_dict = dict()
        self.roi_undist_dict = dict()
        for camera_id in self.params_dict.keys():
            params = self.params_dict[camera_id]
            if len(params) == 0:
                continue  # no distortion
            assert camera_id in self.Ks_dict, f"Missing K for camera {camera_id}"
            assert (
                camera_id in self.params_dict
            ), f"Missing params for camera {camera_id}"
            K = self.Ks_dict[camera_id]
            width, height = self.imsize_dict[camera_id]

            if camtype == "perspective":
                K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                    K, params, (width, height), 0
                )
                mapx, mapy = cv2.initUndistortRectifyMap(
                    K, params, None, K_undist, (width, height), cv2.CV_32FC1
                )
                mask = None
            elif camtype == "fisheye":
                fx = K[0, 0]
                fy = K[1, 1]
                cx = K[0, 2]
                cy = K[1, 2]
                grid_x, grid_y = np.meshgrid(
                    np.arange(width, dtype=np.float32),
                    np.arange(height, dtype=np.float32),
                    indexing="xy",
                )
                x1 = (grid_x - cx) / fx
                y1 = (grid_y - cy) / fy
                theta = np.sqrt(x1**2 + y1**2)
                r = (
                    1.0
                    + params[0] * theta**2
                    + params[1] * theta**4
                    + params[2] * theta**6
                    + params[3] * theta**8
                )
                mapx = (fx * x1 * r + width // 2).astype(np.float32)
                mapy = (fy * y1 * r + height // 2).astype(np.float32)

                # Use mask to define ROI
                mask = np.logical_and(
                    np.logical_and(mapx > 0, mapy > 0),
                    np.logical_and(mapx < width - 1, mapy < height - 1),
                )
                y_indices, x_indices = np.nonzero(mask)
                y_min, y_max = y_indices.min(), y_indices.max() + 1
                x_min, x_max = x_indices.min(), x_indices.max() + 1
                mask = mask[y_min:y_max, x_min:x_max]
                K_undist = K.copy()
                K_undist[0, 2] -= x_min
                K_undist[1, 2] -= y_min
                roi_undist = [x_min, y_min, x_max - x_min, y_max - y_min]
            else:
                assert_never(camtype)

            self.mapx_dict[camera_id] = mapx
            self.mapy_dict[camera_id] = mapy
            self.Ks_dict[camera_id] = K_undist
            self.roi_undist_dict[camera_id] = roi_undist
            self.imsize_dict[camera_id] = (roi_undist[2], roi_undist[3])
            self.mask_dict[camera_id] = mask

        # size of the scene measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)


class Dataset:
    """A simple dataset class."""

    def __init__(
        self,
        parser: Parser,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        indices = np.arange(len(self.parser.image_names))
        if split == "train":
            self.indices = indices[indices % self.parser.test_every != 0]
        else:
            self.indices = indices[indices % self.parser.test_every == 0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        image = imageio.imread(self.parser.image_paths[index])[..., :3]
        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()  # undistorted K
        params = self.parser.params_dict[camera_id]
        camtoworlds = self.parser.camtoworlds[index]
        mask = self.parser.mask_dict[camera_id]

        image_mask = self.parser.image_masks[index]

        if len(params) > 0:
            # Images are distorted. Undistort them.
            mapx, mapy = (
                self.parser.mapx_dict[camera_id],
                self.parser.mapy_dict[camera_id],
            )
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            x, y, w, h = self.parser.roi_undist_dict[camera_id]
            image = image[y : y + h, x : x + w]
            # Apply the same remap+crop to the per-image mask so shapes match.
            if image_mask is not None:
                image_mask = cv2.remap(
                    image_mask.astype(np.uint8), mapx, mapy, cv2.INTER_NEAREST
                ).astype(bool)
                image_mask = image_mask[y : y + h, x : x + w]

        if self.patch_size is not None:
            # Random crop.
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            K[0, 2] -= x
            K[1, 2] -= y

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,  # the index of the image in the dataset
            "camera_idx": self.parser.camera_indices[index],  # 0-based contiguous camera index
        }
        # Per-image mask (DA3 or similar) takes priority; fall back to camera-level ROI mask.
        if image_mask is not None:
            data["mask"] = torch.from_numpy(image_mask).bool()
        elif mask is not None:
            data["mask"] = torch.from_numpy(mask).bool()

        exposure = self.parser.exposure_values[index]
        if exposure is not None:
            data["exposure"] = torch.tensor(exposure, dtype=torch.float32)

        if self.load_depths:
            # projected points to image plane to get depths
            worldtocams = np.linalg.inv(camtoworlds)
            image_name = self.parser.image_names[index]
            point_indices = self.parser.point_indices[image_name]
            points_world = self.parser.points[point_indices]
            points_cam = (worldtocams[:3, :3] @ points_world.T + worldtocams[:3, 3:4]).T
            points_proj = (K @ points_cam.T).T
            points = points_proj[:, :2] / points_proj[:, 2:3]  # (M, 2)
            depths = points_cam[:, 2]  # (M,)
            # filter out points outside the image
            selector = (
                (points[:, 0] >= 0)
                & (points[:, 0] < image.shape[1])
                & (points[:, 1] >= 0)
                & (points[:, 1] < image.shape[0])
                & (depths > 0)
            )
            points = points[selector]
            depths = depths[selector]
            data["points"] = torch.from_numpy(points).float()
            data["depths"] = torch.from_numpy(depths).float()

        return data


if __name__ == "__main__":
    import argparse

    import imageio.v2 as imageio

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/360_v2/garden")
    parser.add_argument("--factor", type=int, default=4)
    args = parser.parse_args()

    # Parse COLMAP data.
    parser = Parser(
        data_dir=args.data_dir, factor=args.factor, normalize=True, test_every=8
    )
    dataset = Dataset(parser, split="train", load_depths=True)
    print(f"Dataset: {len(dataset)} images.")

    writer = imageio.get_writer("results/points.mp4", fps=30)
    for data in tqdm(dataset, desc="Plotting points"):
        image = data["image"].numpy().astype(np.uint8)
        points = data["points"].numpy()
        depths = data["depths"].numpy()
        for x, y in points:
            cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0), -1)
        writer.append_data(image)
    writer.close()
