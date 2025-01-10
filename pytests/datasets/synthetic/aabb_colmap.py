# BSD 2-Clause License
#
# Copyright (c) 2024-2025, Christoph Neuhauser
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

import math
import os
import sys
import struct
import json
from pathlib import Path
import subprocess
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image
import difftetvr as d
import pycolmap  # pip install pycolmap
import open3d as o3d  # pip install open3d


def convert_fov_to_focal_length(fov, pixel_size):
    return pixel_size / (2.0 * math.tan(0.5 * fov))


def read_u64(file):
    return struct.unpack('<Q', file.read(8))[0]


def read_double_vec(file, vec_size):
    return np.array(struct.unpack('<' + ('d' * vec_size), file.read(8 * vec_size)))


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


def visualize_outliers(point_cloud: o3d.geometry.PointCloud, inlier_indices):
    inlier_point_cloud = point_cloud.select_by_index(inlier_indices)
    outlier_point_cloud = point_cloud.select_by_index(inlier_indices, invert=True)
    inlier_point_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    outlier_point_cloud.paint_uniform_color([1.0, 0.0, 0.0])
    o3d.visualization.draw_geometries([inlier_point_cloud, outlier_point_cloud])


def compute_aabb_colmap(colmap_command, dataset_path, use_commandline=True):
    """
    Creates a sparse 3D point cloud using COLMAP and computes its AABB.
    :param colmap_command: The COLMAP command to use when using the commandline.
    :param dataset_path: The path to the dataset for which to compute the AABB.
    :param use_commandline: Whether to use COLMAP from the command line or via pycolmap.
    :return: The AABB estimated from the reconstructed sparse point cloud.
    """
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

    cmdline_str = 'cmdline' if use_commandline else 'pycolmap'
    database_path = os.path.join(dataset_path, f'database_{cmdline_str}.db')
    images_path = os.path.join(dataset_path, 'train')
    sparse_path = os.path.join(dataset_path, 'sparse', cmdline_str)
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
                # We need world to camera space according to https://colmap.github.io/format.html#images-txt
                # - The "pose of an image is specified as the projection from world to the camera coordinate system".
                # - "The local camera coordinate system of an image is defined in a way that the X axis points to the
                #   right, the Y axis to the bottom, and the Z axis to the front as seen from the image."
                # By default, the coordinate system used by the synthetic NeRF datasets has the following convention:
                # - x right, y up, z back
                transform_matrix = np.array(transform_matrix)
                # y -> -z, z -> -z
                transform_matrix[0:3, 1:3] = - transform_matrix[0:3, 1:3]
                view_matrix = np.linalg.inv(transform_matrix)
                t = np.array([view_matrix[0][3], view_matrix[1][3], view_matrix[2][3]])
                rotation_matrix = view_matrix[0:3, 0:3]
                q = Rotation.from_matrix(rotation_matrix).as_quat()
                cameras_txt_file.write(f'{idx + 1} {q[3]} {q[0]} {q[1]} {q[2]} {t[0]} {t[1]} {t[2]} 1 {f}\n\n')
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
            model=pycolmap.CameraModelId.SIMPLE_PINHOLE,
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
            transform_matrix = np.array(frame['transform_matrix'])
            # See use_commandline case above: y -> -z, z -> -z
            transform_matrix[0:3, 1:3] = -transform_matrix[0:3, 1:3]
            view_matrix = np.linalg.inv(transform_matrix)
            t = np.array([view_matrix[0][3], view_matrix[1][3], view_matrix[2][3]])
            rotation_matrix = view_matrix[0:3, 0:3]
            # camera_position = np.array([transform_matrix[0][3], transform_matrix[1][3], transform_matrix[2][3]])
            # rotation_matrix = np.array(transform_matrix)[0:3, 0:3]
            # prior_pose = pycolmap.PosePrior(camera_position, rotation_matrix)
            prior_pose = pycolmap.PosePrior(t, rotation_matrix)
            cam_from_world = pycolmap.Rigid3d(view_matrix[0:3, 0:4])
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

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd_filtered, filtered_indices = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    # radius = np.linalg.norm(max_vec - min_vec) * 5e-3
    # pcd_filtered, filtered_indices = pcd.remove_radius_outlier(nb_points=5, radius=radius)
    # o3d.io.write_point_cloud('0_pts_filt.ply', pcd_filtered)
    # o3d.visualization.draw_geometries([pcd_filtered])
    visualize_outliers(pcd, filtered_indices)
    indices_arr = np.array(filtered_indices, dtype=int)
    points_3d_sel = points_3d[indices_arr]
    min_vec = np.min(points_3d_sel, axis=0)
    max_vec = np.max(points_3d_sel, axis=0)
    aabb = d.AABB3(d.vec3(min_vec[0], min_vec[1], min_vec[2]), d.vec3(max_vec[0], max_vec[1], max_vec[2]))
    return aabb
