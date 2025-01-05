# BSD 2-Clause License
#
# Copyright (c) 2024, Christoph Neuhauser
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# The synthetic datasets from the original NeRF paper (https://www.matthewtancik.com/nerf)
# do not contain SfM data by COLMAP. While we usually use the sparse point cloud data to estimate the scene bounds,
# this is thus not possible for this data. I see three possibilities:
# - Run COLMAP to only generate sparse 3D points.
# - Use voxel carving (disadvantage: not exact, and we also need bounds for the voxel grid...).
# - Use the depth maps and the formula from: https://github.com/bmild/nerf/issues/77#issuecomment-1859165869
# This file is for testing these options.

import math
import os
import sys
import glob
import json
import argparse
import pathlib
from pathlib import Path
import subprocess
import struct
import numpy as np
from scipy.spatial.transform import Rotation
from numba import njit
from PIL import Image
import torch
import difftetvr as d
import pycolmap  # pip install pycolmap
import open3d as o3d


def run_command(command):
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = proc.communicate()
    proc_status = proc.wait()
    if proc_status != 0:
        if os.name == 'nt':
            stderr_string = err.decode('latin-1')
            stdout_string = output.decode('latin-1')
        else:
            stderr_string = err.decode('utf-8')
            stdout_string = output.decode('utf-8')
        print(f"Command '{' '.join(command)}' exited with code {proc_status}.")
        print('--- Output from stdout ---')
        print(stdout_string.rstrip('\n'))
        print('---\n')
        print('--- Output from stderr ---', file=sys.stderr)
        print(stderr_string.rstrip('\n'), file=sys.stderr)
        print('---', file=sys.stderr)
        sys.exit(1)


def convert_fov_to_focal_length(fov, pixel_size):
    return pixel_size / (2.0 * math.tan(0.5 * fov))


def read_u64(file):
    return struct.unpack('<Q', file.read(8))[0]


def read_double_vec(file, vec_size):
    return np.array(struct.unpack('<' + ('d' * vec_size), file.read(8 * vec_size)))


# Creates a sparse 3D point cloud using COLMAP and computes its AABB. Currently unfortunately broken...
def compute_aabb_colmap(colmap_command, dataset_path):
    # Retrieve the field of view and image resolution from image 0.
    with open(os.path.join(dataset_path, 'transforms_train.json')) as json_file:
        transforms_json = json.load(json_file)
        fovx = transforms_json['camera_angle_x']
        frames = transforms_json['frames']
        frame_0 = frames[0]
        frame_0_file_path = frame_0['file_path'] + '.png'
        image_0_path = os.path.join(dataset_path, frame_0_file_path)
        image_0 = Image.open(image_0_path)
        image_0 = np.array(image_0).astype(np.float32) / 255.0
        img_width = image_0.shape[1]
        img_height = image_0.shape[0]
        focal_length_x = convert_fov_to_focal_length(fovx, img_width)

    # Whether to use COLMAP from the command line or via pycolmap.
    use_commandline = False

    database_path = os.path.join(dataset_path, 'database.db')
    images_path = os.path.join(dataset_path, 'train')
    sparse_path = os.path.join(dataset_path, 'sparse', '0')
    input_init_path = os.path.join(dataset_path, 'input_init')
    if not os.path.isdir(sparse_path) and use_commandline:
        # https://colmap.github.io/faq.html#reconstruct-sparse-dense-model-from-known-camera-poses
        # https://github.com/colmap/colmap/issues/2888
        Path(sparse_path).mkdir(parents=True, exist_ok=True)
        Path(input_init_path).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(input_init_path, 'cameras.txt'), 'w') as cameras_txt_file:
            cameras_txt_file.write('# Camera list with one line of data per camera:\n')
            cameras_txt_file.write('#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n')
            cameras_txt_file.write('# Number of cameras: 1\n')
            cameras_txt_file.write(
                f'1 SIMPLE_PINHOLE {img_width} {img_height} {focal_length_x} {img_width / 2} {img_height / 2}\n')
        with open(os.path.join(input_init_path, 'images.txt'), 'w') as cameras_txt_file:
            cameras_txt_file.write('# Image list with two lines of data per image:\n')
            cameras_txt_file.write('#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n')
            cameras_txt_file.write('#   POINTS2D[] as (X, Y, POINT3D_ID)\n')
            for idx, frame in enumerate(frames):
                f = os.path.basename(frame['file_path']) + '.png'
                transform_matrix = frame['transform_matrix']
                t = np.array([transform_matrix[0][3], transform_matrix[1][3], transform_matrix[2][3]])
                rotation_matrix = np.array(transform_matrix)[0:3, 0:3]
                q = Rotation.from_matrix(rotation_matrix).as_quat()
                q = [q[3], q[0], q[1], q[2]]
                cameras_txt_file.write(f'{idx + 1} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} 1 {f}\n\n')
        with open(os.path.join(input_init_path, 'points3D.txt'), 'w') as cameras_txt_file:
            pass
        run_command([
            colmap_command, 'feature_extractor',
            '--database_path', database_path,
            '--image_path', images_path,
        ])
        run_command([
            colmap_command, 'exhaustive_matcher',
            '--database_path', database_path,
        ])
        run_command([
            colmap_command, 'point_triangulator',
            '--database_path', database_path,
            '--image_path', images_path,
            '--input_path', input_init_path,
            '--output_path', sparse_path,
        ])

    if not os.path.isdir(sparse_path) and not use_commandline:
        # Further reading:
        # https://github.com/colmap/pycolmap/issues/289
        # https://github.com/colmap/colmap/issues/2888
        # https://github.com/colmap/pycolmap/issues/272
        # https://github.com/colmap/colmap/blob/main/scripts/python/database.py
        # https://colmap.github.io/faq.html#reconstruct-sparse-dense-model-from-known-camera-poses
        # https://github.com/NVlabs/instant-ngp/blob/master/docs/nerf_dataset_tips.md

        colmap_camera = pycolmap.Camera(
            camera_id=1,
            model='SIMPLE_PINHOLE',
            width=img_width,
            height=img_height,
            params=[focal_length_x, img_width / 2, img_height / 2]
        )
        colmap_images = []

        # Manually copy intrinsics, cf. https://github.com/colmap/colmap/blob/main/scripts/python/database.py
        database = pycolmap.Database(database_path)
        database.write_camera(use_camera_id=True, camera=colmap_camera)
        for idx, frame in enumerate(frames):
            img_name = os.path.basename(frame['file_path']) + '.png'
            transform_matrix = frame['transform_matrix']
            camera_position = np.array([transform_matrix[0][3], transform_matrix[1][3], transform_matrix[2][3]])
            rotation_matrix = np.array(transform_matrix)[0:3, 0:3]
            prior_pose = pycolmap.PosePrior(camera_position, rotation_matrix)
            transform_matrix = np.array(transform_matrix)[0:3, 0:4]
            cam_from_world = pycolmap.Rigid3d(transform_matrix)
            colmap_image = pycolmap.Image(
                name=img_name, cam_from_world=cam_from_world, camera_id=1, id=(idx + 1))
            colmap_images.append(colmap_image)
            database.write_image(use_image_id=True, image=colmap_image)
            database.write_pose_prior(image_id=(idx + 1), pose_prior=prior_pose)
        database.close()

        Path(sparse_path).mkdir(parents=True, exist_ok=True)
        pycolmap.extract_features(database_path, images_path, os.listdir(images_path))

        pycolmap.match_exhaustive(database_path)
        reconstruction = pycolmap.Reconstruction()
        reconstruction.add_camera(colmap_camera)
        for idx, frame in enumerate(frames):
            colmap_image = colmap_images[idx]
            reconstruction.add_image(colmap_image)
            reconstruction.register_image(idx + 1)
        pycolmap.triangulate_points(
            reconstruction, database_path, images_path, sparse_path, refine_intrinsics=False, clear_points=False)
        reconstruction.check()
        reconstruction.write(output_dir=sparse_path)

    # Finally, open the sparse 3D point cloud and compute the bounds.
    points_bin_path = os.path.join(sparse_path, 'points3D.bin')
    with open(points_bin_path, 'rb') as points_bin_file:
        num_points = read_u64(points_bin_file)
        points_3d = np.empty((num_points, 3))
        for i in range(num_points):
            point_id = read_u64(points_bin_file)
            point_position = read_double_vec(points_bin_file, 3)
            points_3d[i, :] = point_position
            points_bin_file.read(3 + 8)  # 3x u8 rgb + 1x f64 error
            track_length = read_u64(points_bin_file)
            points_bin_file.read(8 * track_length)  # track elements
    #pcd = o3d.geometry.PointCloud()
    #pcd.points = o3d.utility.Vector3dVector(points_3d)
    #o3d.visualization.draw_geometries([pcd])
    min_vec = np.min(points_3d, axis=0)
    max_vec = np.max(points_3d, axis=0)
    aabb = d.AABB3(d.vec3(min_vec[0], min_vec[1], min_vec[2]), d.vec3(max_vec[0], max_vec[1], max_vec[2]))
    print('AABB using COLMAP:')
    print(f'{aabb.min.x}, {aabb.min.y}, {aabb.min.z}')
    print(f'{aabb.max.x}, {aabb.max.y}, {aabb.max.z}')
    return aabb


# Used by @see compute_aabb_from_depth_images to get the min/max world position for a depth map.
@njit
def get_min_max_world_range(depth_image, valid_mask, cam_to_world, fovx, min_vec, max_vec):
    width = depth_image.shape[1]
    height = depth_image.shape[0]
    x0_max = math.tan(0.5 * fovx)
    y0_max = x0_max * height / width
    point = np.ones(4)
    point_list_a = []
    point_list_b = []
    for y in range(height):
        y_coord = 2.0 * (y + 0.5) / height - 1.0
        for x in range(width):
            if valid_mask[y, x]:
                x_coord = 2.0 * (x + 0.5) / width - 1.0
                depth = depth_image[y, x]
                point[0] = x0_max * x_coord * depth
                point[1] = y0_max * y_coord * depth
                point[2] = -depth
                point_world = np.dot(cam_to_world, point)
                point_list_a.append(point[0:3].copy())
                point_list_b.append(point_world)
                for i in range(3):
                    min_vec[i] = min(min_vec[i], point_world[i])
                    max_vec[i] = max(max_vec[i], point_world[i])

    # Debug code. For some reason, the point clouds don't perfectly align.
    #pcd_a = o3d.geometry.PointCloud()
    #pcd_a.points = o3d.utility.Vector3dVector(np.array(point_list_a))
    #o3d.io.write_point_cloud('pcd_a.ply', pcd_a)
    #pcd_b = o3d.geometry.PointCloud()
    #pcd_b.points = o3d.utility.Vector3dVector(np.array(point_list_b))
    #o3d.io.write_point_cloud('pcd_b.ply', pcd_b)

    return min_vec, max_vec


# Computes the AABB by using the depth images provided in the test folder.
def compute_aabb_from_depth_images(dataset_path):
    image_path = os.path.join(dataset_path, 'test')
    with open(os.path.join(dataset_path, 'transforms_test.json')) as f:
        transforms_json = json.load(f)
        fovx = transforms_json['camera_angle_x']
        frames = transforms_json['frames']
        all_filenames = os.listdir(image_path)
        min_vec = np.ones(3) * 1e10
        max_vec = np.ones(3) * -1e10
        for frame in frames:
            frame_file_path = frame['file_path']
            frame_filename = os.path.basename(frame_file_path) + '_depth'
            depth_image_path = None
            for filename in all_filenames:
                if filename.startswith(frame_filename):
                    depth_image_path = os.path.join(dataset_path, 'test', filename)
                    break
            depth_image = Image.open(depth_image_path)
            depth_image = np.array(depth_image).astype(np.float32)[:, :, 0] / 255.0
            valid_mask = depth_image != 0.0
            # https://github.com/bmild/nerf/issues/77
            depth_image_shift = 8.0 * (1.0 - depth_image)
            depth_image[valid_mask] = depth_image_shift[valid_mask]
            transform_matrix = frame['transform_matrix']
            camera_position = np.array([transform_matrix[0][3], transform_matrix[1][3], transform_matrix[2][3]])
            transform_matrix = np.array(transform_matrix)
            min_vec, max_vec = get_min_max_world_range(
                depth_image, valid_mask, transform_matrix[0:3, :], fovx, min_vec, max_vec)
    print('AABB from depth:')
    print(min_vec)
    print(max_vec)
    return d.AABB3(d.vec3(min_vec[0], min_vec[1], min_vec[2]), d.vec3(max_vec[0], max_vec[1], max_vec[2]))


# Computes the AABB by using voxel carving. Currently unfortunately broken...
def compute_aabb_voxel_carving(dataset_path):
    with open(os.path.join(dataset_path, 'transforms_train.json')) as f:
        transforms_json = json.load(f)
        fovx = transforms_json['camera_angle_x']
        frames = transforms_json['frames']

        # First, compute the bounds for the camera poses. We assume the object is encompassed by them.
        min_vec_cams = np.ones(3) * 1e10
        max_vec_cams = np.ones(3) * -1e10
        for frame in frames:
            inv_view_matrix = frame['transform_matrix']
            camera_position = np.array([inv_view_matrix[0][3], inv_view_matrix[1][3], inv_view_matrix[2][3]])
            for i in range(3):
                min_vec_cams[i] = min(min_vec_cams[i], camera_position[i])
                max_vec_cams[i] = max(max_vec_cams[i], camera_position[i])

        # Next, apply voxel carving.
        print('Starting voxel carving...')
        camera_settings = d.CameraSettings()
        grid_bounding_box = d.AABB3(
            d.vec3(min_vec_cams[0], min_vec_cams[1], min_vec_cams[2]),
            d.vec3(max_vec_cams[0], max_vec_cams[1], max_vec_cams[2]))
        # Issue: The bounds of the camera poses seems to be not useful, as they may not surround the whole object...
        grid_bounding_box = d.AABB3(d.vec3(-4.0, -4.0, -4.0), d.vec3(4.0, 4.0, 4.0))
        grid_resolution = d.uvec3(256, 256, 256)
        voxel_carving = d.VoxelCarving(grid_bounding_box, grid_resolution)
        for frame in frames:
            frame_file_path = frame['file_path'] + '.png'
            image_path = os.path.join(dataset_path, frame_file_path)
            image = Image.open(image_path)
            image = np.array(image).astype(np.float32) / 255.0
            img_width = image.shape[1]
            img_height = image.shape[0]
            image = torch.from_numpy(image)
            fovy = fovx * img_height / img_width
            inv_view_matrix = np.array(frame['transform_matrix'])
            view_matrix = np.linalg.inv(inv_view_matrix)
            view_matrix_array = np.empty(16)
            for k in range(4):
                for j in range(4):
                    view_matrix_array[k * 4 + j] = view_matrix[j, k]
            camera_settings.set_intrinsics(img_width, img_height, fovy, near=1e-2, far=100.0)
            camera_settings.set_view_matrix(view_matrix_array)
            voxel_carving.process_next_frame(image, camera_settings)

    aabb = voxel_carving.compute_non_empty_bounding_box()
    print('AABB using voxel carving:')
    print(f'{aabb.min.x}, {aabb.min.y}, {aabb.min.z}')
    print(f'{aabb.max.x}, {aabb.max.y}, {aabb.max.z}')
    return aabb


def main():
    parser = argparse.ArgumentParser(
        prog='difftetvr/utils/synthetic_compute_aabb.py',
        description='Computes the AABB (axis-aligned bounding box) for the synthetic NeRF datasets.')
    parser.add_argument('--datasets_path', type=str)
    parser.add_argument('--colmap_path', type=str)
    args = parser.parse_args()
    default_colmap_search_path = os.path.join(pathlib.Path.home(), 'Software/COLMAP/bin/colmap')
    if args.colmap_path is None and os.path.isfile(default_colmap_search_path):
        args.colmap_path = default_colmap_search_path
    colmap_command = args.colmap_path if args.colmap_path is not None else "colmap"

    datasets_path = args.datasets_path
    datasets_folder_default = '/media/christoph/Elements16C/Datasets/NeRF/nerf_synthetic'
    if datasets_path is None and os.path.isdir(datasets_folder_default):
        datasets_path = datasets_folder_default
    if datasets_path is None:
        raise RuntimeError('Dataset path needs to be specified using \'--datasets_path\'.')

    dataset_list = os.listdir(datasets_path)
    dataset_list = ['lego']
    for dataset_name in dataset_list:
        dataset_path = os.path.join(datasets_path, dataset_name)
        if os.path.isdir(dataset_path):
            aabb_colmap = compute_aabb_colmap(colmap_command, dataset_path)
            aabb_depth = compute_aabb_from_depth_images(dataset_path)
            aabb_voxel_carving = compute_aabb_voxel_carving(dataset_path)

            # Currently, only aabb_depth seems to work more or less reliably...
            aabb = aabb_depth
            with open(os.path.join(dataset_path, 'aabb.json'), 'w') as aabb_file:
                aabb_file.write(f'[{aabb.min.x}, {aabb.min.y}, {aabb.min.z}, {aabb.max.x}, {aabb.max.y}, {aabb.max.z}]')


if __name__ == '__main__':
    main()
