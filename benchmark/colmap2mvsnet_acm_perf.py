#!/usr/bin/env python
"""
Copyright 2019, Jingyang Zhang and Yao Yao, HKUST. Model reading is provided by COLMAP.
Preprocess script.
View selection is modified according to COLMAP's strategy, Qingshan Xu
Runtime performance optimized to support large datasets, Gabor Pelesz
"""

import os
import collections
import struct
import argparse
import shutil
import multiprocessing as mp
from functools import partial
from multiprocessing.pool import ThreadPool
import time

import cv2
import tqdm
import numpy as np
from pyquaternion import Quaternion

#============================ read_model.py ============================#
CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model) \
                         for camera_model in CAMERA_MODELS])


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_cameras_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras


def read_cameras_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for camera_line_index in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras


def read_images_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images


def read_images_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for image_index in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images


def read_points3D_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    points3D = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                point3D_id = int(elems[0])
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = float(elems[7])
                image_ids = np.array(tuple(map(int, elems[8::2])))
                point2D_idxs = np.array(tuple(map(int, elems[9::2])))
                points3D[point3D_id] = Point3D(id=point3D_id, xyz=xyz, rgb=rgb,
                                               error=error, image_ids=image_ids,
                                               point2D_idxs=point2D_idxs)
    return points3D


def read_points3d_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for point_line_index in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length,
                format_char_sequence="ii"*track_length)
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(
                id=point3D_id, xyz=xyz, rgb=rgb,
                error=error, image_ids=image_ids,
                point2D_idxs=point2D_idxs)
    return points3D


def read_model(path, ext):
    if ext == ".txt":
        cameras = read_cameras_text(os.path.join(path, "cameras" + ext))
        images = read_images_text(os.path.join(path, "images" + ext))
        points3D = read_points3D_text(os.path.join(path, "points3D") + ext)
    else:
        cameras = read_cameras_binary(os.path.join(path, "cameras" + ext))
        images = read_images_binary(os.path.join(path, "images" + ext))
        points3D = read_points3d_binary(os.path.join(path, "points3D") + ext)
    return cameras, images, points3D

distortion_param_type = {
    'SIMPLE_PINHOLE': ['f', 'cx', 'cy'],
    'PINHOLE': ['fx', 'fy', 'cx', 'cy'],
    'SIMPLE_RADIAL': ['f', 'cx', 'cy', 'k'],
    'SIMPLE_RADIAL_FISHEYE': ['f', 'cx', 'cy', 'k'],
    'RADIAL': ['f', 'cx', 'cy', 'k1', 'k2'],
    'RADIAL_FISHEYE': ['f', 'cx', 'cy', 'k1', 'k2'],
    'OPENCV': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2'],
    'OPENCV_FISHEYE': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'k3', 'k4'],
    'FULL_OPENCV': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6'],
    'FOV': ['fx', 'fy', 'cx', 'cy', 'omega'],
    'THIN_PRISM_FISHEYE': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'sx1', 'sy1']
}

def to_homogeneous(points):
    # points.shape = [N, D]    -> list of row vectors without explicit column component
    if len(points.shape) == 2:
        ones = np.ones((len(points), 1), dtype=np.float64)
        return np.hstack((points, ones))

    # points.shape = [N, D, 1] -> list of row vectors
    elif len(points.shape) == 3 and points.shape[-1] == 1:
        ones = np.ones((len(points), 1, 1), dtype=np.float64)
        return np.hstack((points, ones))

    # points.shape = [N, 1, D] -> list of col vectors
    elif len(points.shape) == 3 and points.shape[-2] == 1:
        ones = np.ones((len(points), 1, 1), dtype=np.float64)
        return np.hstack((points.transpose([0, 2, 1]), ones)).transpose([0, 2, 1])

    raise Exception(f"Unexpected shape for points: {points.shape}")

def qvec2rotmat(qvec):
    return Quaternion(qvec).rotation_matrix
def rotmat2qvec(R):
    return Quaternion(matrix=R).elements

def calc_score(inputs, images, points3d, extrinsic, args):
    i, j = inputs
    id_i = images[i].point3D_ids
    id_j = images[j].point3D_ids
    id_intersect = [it for it in id_i if it in id_j]

    cam_center_i = -np.matmul(extrinsic[i][:3, :3].transpose(), extrinsic[i][:3, 3:4])[:, 0]
    cam_center_j = -np.matmul(extrinsic[j][:3, :3].transpose(), extrinsic[j][:3, 3:4])[:, 0]

    score = 0
    angles = []
    for pid in id_intersect:
        if pid == -1:
            continue
        p = points3d[pid].xyz
        theta = (180 / np.pi) * np.arccos(np.dot(cam_center_i - p, cam_center_j - p) / np.linalg.norm(cam_center_i - p) / np.linalg.norm(cam_center_j - p)) # triangulation angle
        angles.append(theta)
        score += 1
    if len(angles) > 0:
        angles_sorted = sorted(angles)
        triangulationangle = angles_sorted[int(len(angles_sorted) * 0.75)]
        if triangulationangle < 1:
            score = 0.0
    return i, j, score


def calc_score_fast(queue_item, point3D_ids, points3d_xyz, cam_centers):
    i, j = queue_item
    id_i = point3D_ids[i]
    id_j = point3D_ids[j]

    # NOTE: Potential bug found in original approach.
    # In previous version id_i and id_j were not unique.
    # Also their intersection was not unique! (some IDs were found multiple times)
    # If this was the intended behavior then try to call
    # np.intersect1d with returning indices instead,
    # then iterate over one of the array's indices to get
    # the same result. We won't use this because we are assuming
    # that the intersection was meant to be unique.
    # id_intersect = [it for it in id_i if it in id_j]
    # The final score will be different because essentially: score = len(id_intersect)
    id_intersect = np.intersect1d(
        id_i,
        id_j,
        # we can assume unique because id_i & id_j
        # was produced using np.unique
        assume_unique=True,
        return_indices=False,
    )

    if len(id_intersect) == 0:
        return i, j, 0

    cam_center_i = cam_centers[i]
    cam_center_j = cam_centers[j]

    score = len(id_intersect)

    intersect3D = points3d_xyz[id_intersect]
    cam_center_i_p = cam_center_i - intersect3D
    cam_center_j_p = cam_center_j - intersect3D
    dot_on_axis0 = np.einsum("nk,nk->n", cam_center_i_p, cam_center_j_p)
    angles = (180.0 / np.pi) * np.arccos(dot_on_axis0 / np.linalg.norm(cam_center_i_p, axis=1) / np.linalg.norm(cam_center_j_p, axis=1))

    angles.sort()
    triangulationangle = angles[int(len(angles) * 0.75)]
    if triangulationangle < 1:
        score = 0.0

    return i, j, score

def calc_score_fast_chunked(queue_chunk, point3D_ids, points3d_xyz, cam_centers):
    return [calc_score_fast(item, point3D_ids, points3d_xyz, cam_centers) for item in queue_chunk]

def calc_score_old_chunked(queue_chunk, *args, **kargs):
    return [calc_score(item, *args, **kargs) for item in queue_chunk]


def processing_single_scene(dense_folder, save_images_dir, save_cams_dir, interval_scale=1, max_d=192, multiprocessing=True):
    t0 = time.perf_counter()
    print("Reading model...")
    sparse_dir = os.path.join(dense_folder, 'dslr_calibration_undistorted')
    cameras, images, points3d = read_model(sparse_dir, ".txt")
    num_images = len(list(images.items()))
    t1 = time.perf_counter()
    print(f"Finished. Num images: {num_images} ({(t1-t0):.3f} sec)")

    print("Remap IDs...")
    images = [images[image_id] for image_id in sorted(images.keys())]

    # intrinsic
    print("start intrinsic...")
    intrinsic = {}
    for camera_id, cam in cameras.items():
        params_dict = {key: value for key, value in zip(distortion_param_type[cam.model], cam.params)}
        if 'f' in distortion_param_type[cam.model]:
            params_dict['fx'] = params_dict['f']
            params_dict['fy'] = params_dict['f']
        i = np.array([
            [params_dict['fx'], 0, params_dict['cx']],
            [0, params_dict['fy'], params_dict['cy']],
            [0, 0, 1]
        ])
        intrinsic[camera_id] = i
    print('intrinsic finished!')

    # extrinsic
    print("start extrinsic...")
    t0 = time.perf_counter()
    extrinsic = []
    for image in images:
        e = np.eye(4)
        e[:3, :3] = qvec2rotmat(image.qvec)
        e[:3, 3] = image.tvec
        extrinsic.append(e)
    t1 = time.perf_counter()
    print(f"Extrinsic finished! ({(t1-t0):.3f} sec)")

    # view selection
    print("Initializing view selection queue...")
    t0 = time.perf_counter()
    queue = np.stack(np.triu_indices(n=len(images), m=len(images), k=1), axis=1)
    t1 = time.perf_counter()
    print(f"View selection queue initialized! ({(t1-t0):.3f} sec)")

    
    # Remap points3D_ids to array indices for convenient and fast score calc
    print(f"Start Point3D index remap.")
    t0 = time.perf_counter()
    points3d_id_to_idx_LUT = np.zeros(max(points3d.keys())+1, np.int32)
    points3d_xyz = []
    for i, p3D_ID in enumerate(sorted(points3d.keys())):
        points3d_id_to_idx_LUT[p3D_ID] = i
        points3d_xyz.append(points3d[p3D_ID].xyz.tolist())
    points3d_xyz = np.array(points3d_xyz, dtype=np.float64)
    point3D_ids = [points3d_id_to_idx_LUT[np.unique(image.point3D_ids[image.point3D_ids != -1])] for image in images]

    cam_centers = [np.linalg.inv(e)[:3, 3] for e in extrinsic]
    t1 = time.perf_counter()
    print(f"Point3D index remap finished! ({(t1-t0):.3f} sec)")

    print(f"Start map... ({len(images)} -> {len(queue)})")
    t0 = time.perf_counter()
    if multiprocessing:
        print(f"Multiprocessing enabled with {mp.cpu_count()} cores")
        # func = partial(calc_score_old_chunked, images=images, points3d=points3d, args=args, extrinsic=extrinsic)
        func = partial(calc_score_fast_chunked, point3D_ids=point3D_ids, points3d_xyz=points3d_xyz, cam_centers=cam_centers)

        # Dynamic chunk size estimation
        # chunk size really depends on the number of point3D indices, thus larger project might benefit
        # from larger chunksizes.
        import random
        chunk_estimation_t0 = time.perf_counter()
        runtime_samples = 30
        [func([queue[i]]) for i in [random.randrange(0, len(queue)) for _ in range(runtime_samples)]]
        chunk_estimation_t1 = time.perf_counter()
        it_per_sec = 1.0 / ((chunk_estimation_t1 - chunk_estimation_t0) / runtime_samples)
        print(f"Estimated speed: {it_per_sec:.3f} iter/sec")

        total = len(queue)
        chunksize = int(it_per_sec * 5) # we have it/s, let's run for 5 sec at least per processing unit
        print(f"Selected chunksize for multiprocessing: {chunksize}")

        if chunksize > total:
            # do it without multiprocessing
            func = partial(calc_score_fast, point3D_ids=point3D_ids, points3d_xyz=points3d_xyz, cam_centers=cam_centers)
            # func = partial(calc_score, images=images, points3d=points3d, args=args, extrinsic=extrinsic)
            result = [func(q) for q in tqdm.tqdm(queue)]
        else:
            queue_chunk = [queue[chunk:chunk+chunksize]
                        for chunk in range(0, total, chunksize)]
            result = []
            with mp.Pool(processes=mp.cpu_count()) as pool:
                with tqdm.tqdm(total=total, disable=False) as pbar:
                    for result_chunk in pool.imap_unordered(func, queue_chunk):
                        result += result_chunk
                        pbar.update(len(result_chunk))
    else:
        # func = partial(calc_score, images=images, points3d=points3d, args=args, extrinsic=extrinsic)
        func = partial(calc_score_fast, point3D_ids=point3D_ids, points3d_xyz=points3d_xyz, cam_centers=cam_centers)
        result = [func(q) for q in tqdm.tqdm(queue)]
    t1 = time.perf_counter()
    print(f"Finished map. {(t1-t0):.3f} sec")

    print("Start view selection...")
    t0 = time.perf_counter()
    score = np.zeros((len(images), len(images)))
    for i, j, s in result:
        score[i, j] = s
        score[j, i] = s
    view_sel = []
    num_view = min(20, len(images) - 1)
    for i in range(len(images)):
        sorted_score = np.argsort(score[i])[::-1]
        view_sel.append([(k, score[i, k]) for k in sorted_score[:num_view]])
    t1 = time.perf_counter()
    print(f"View selection finished! {(t1-t0):.3f} sec")

    # depth range and interval
    print("start depth_ranges...")
    t0 = time.perf_counter()
    depth_ranges = []
    for i in range(len(images)):        
        # look up corresponding 3D points for point IDs
        # points3d_xyz_valid = points3d_xyz[point3D_ids[i]] # NOTE: could use this however this contains unique IDs!
        points3d_xyz_valid = [points3d[p3D_id].xyz for p3D_id in images[i].point3D_ids[images[i].point3D_ids != -1]]
        if len(points3d_xyz_valid) != 0:
            points3d_xyz_valid = np.array(points3d_xyz_valid)
            transformed = extrinsic[i] @ to_homogeneous(points3d_xyz_valid[..., np.newaxis])
            zs = transformed[:, 2, 0]
            zs.sort()
            depth_min = zs[int(len(zs) * .01)] * 0.75
            depth_max = zs[int(len(zs) * .99)] * 1.25
        else:
            depth_min = 1 * 0.75
            depth_max = 2 * 1.25

        if max_d != 0:
            depth_num = max_d
        else:
            raise NotImplementedError("Determining the depth number by inverse "
                                    "depth setting is not implemented in the refactored version")

        depth_interval = (depth_max - depth_min) / (depth_num - 1) / interval_scale
        depth_ranges.append((depth_min, depth_interval, depth_num, depth_max))
    t1 = time.perf_counter()
    print(f"Finished depth_ranges. {(t1-t0):.3f} sec")

    # write
    print("Start writing .txt files.")
    t0 = time.perf_counter()
    for i in range(len(images)):
        with open(os.path.join(save_cams_dir, '%08d_cam.txt' % i), 'w') as f:
            f.write('extrinsic\n')
            for j in range(4):
                for k in range(4):
                    f.write(str(extrinsic[i][j, k]) + ' ')
                f.write('\n')
            f.write('\nintrinsic\n')
            for j in range(3):
                for k in range(3):
                    f.write(str(intrinsic[images[i].camera_id][j, k]) + ' ')
                f.write('\n')
            f.write('\n%f %f %f %f\n' % (depth_ranges[i][0], depth_ranges[i][1], depth_ranges[i][2], depth_ranges[i][3]))
            # write image name
            f.write('%s\n' % images[i].name)
    with open(os.path.join(args.save_folder, 'pair.txt'), 'w') as f:
        f.write('%d\n' % len(images))
        for i, sorted_score in enumerate(view_sel):
            f.write('%d\n%d ' % (i, len(sorted_score)))
            for image_id, s in sorted_score:
                f.write('%d %d ' % (image_id, s))
            f.write('\n')
    t1 = time.perf_counter()
    print(f"Finished writing .txt files. {(t1-t0):.3f} sec")


    #convert to jpg
    print("Start copying images...")
    t0 = time.perf_counter()
    image_dir = os.path.join(args.dense_folder, 'images')

    def copy_to_jpg(image):
        img_path = os.path.join(image_dir, image.name)
        out_path = os.path.join(save_images_dir, '%08d.jpg' % i)
        if not img_path.endswith(".jpg"):
            cv2.imwrite(out_path, cv2.imread(img_path))
        else:
            shutil.copyfile(img_path, out_path)

    if multiprocessing:
        with ThreadPool(processes=os.cpu_count()) as pool:
            i = 0
            for _ in pool.imap_unordered(copy_to_jpg, images):
                print(f"Copying image: {i+1} / {len(images)}")
                i+=1
    else:
        for i, image in enumerate(images):
            copy_to_jpg(image)
            print(f"Copying image: {i+1} / {len(images)}")

    t1 = time.perf_counter()
    print(f"Copying images finished: {(t1-t0):.3f} sec")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert colmap camera')

    parser.add_argument('--dense_folder', required=True, type=str)
    parser.add_argument('--save_folder', required=True, type=str)
    parser.add_argument('--max_d', type=int, default=192)
    parser.add_argument('--interval_scale', type=float, default=1)
    parser.add_argument('--single-thread', type=bool, help="Turns off multiprocessing and multithreading.")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.save_folder), exist_ok=True)

    save_cams_dir = os.path.join(args.save_folder, 'cams')
    save_images_dir = os.path.join(args.save_folder, 'images')

    if os.path.exists(save_images_dir):
        print("remove:{}".format(save_images_dir))
        shutil.rmtree(save_images_dir)
    os.makedirs(save_images_dir)
    if os.path.exists(save_cams_dir):
        print("remove:{}".format(save_cams_dir))
        shutil.rmtree(save_cams_dir)
    os.makedirs(save_cams_dir)

    processing_single_scene(args.dense_folder, save_images_dir, save_cams_dir, args.interval_scale, args.max_d, (not args.single_thread))