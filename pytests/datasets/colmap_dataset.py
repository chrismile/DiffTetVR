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

import os
import struct
import numpy as np
import torch
import torch.utils.data
import difftetvr as d
try:
    import open3d as o3d  # pip install open3d
    can_use_o3d = True
except ImportError:
    o3d = None
    can_use_o3d = False
from .dataset import Dataset3D
from .imgutils import load_image_array
from .sample_view import matrix_translation, matrix_quaternion, convert_focal_length_to_fov, get_scale_factor, \
    apply_scale_factor_aabb
from .colmap.colmap_align_axes import align_axes_point_cloud


def read_u64(file):
    return struct.unpack('<Q', file.read(8))[0]


def read_i32(file):
    return struct.unpack('<i', file.read(4))[0]


def read_double_vec(file, vec_size):
    return np.array(struct.unpack('<' + ('d' * vec_size), file.read(8 * vec_size)))


def visualize_outliers(point_cloud: o3d.geometry.PointCloud, inlier_indices):
    inlier_point_cloud = point_cloud.select_by_index(inlier_indices)
    outlier_point_cloud = point_cloud.select_by_index(inlier_indices, invert=True)
    inlier_point_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    outlier_point_cloud.paint_uniform_color([1.0, 0.0, 0.0])
    o3d.visualization.draw_geometries([inlier_point_cloud, outlier_point_cloud])


class ColmapDataset(torch.utils.data.Dataset, Dataset3D):
    """
    Loads COLMAP data sets. For more information see:
    - https://github.com/colmap/colmap/blob/main/scripts/python/read_write_model.py
    - https://github.com/colmap/colmap/blob/main/src/colmap/scene/reconstruction_io.cc
    """

    def __init__(self, data_dir, resolution=1, images_dir_name='images', sparse_dirname='sparse/0'):
        super().__init__()

        if resolution > 1 and '_' not in images_dir_name:
            images_dir_name += f'_{resolution}'
        self.images_dir_name = images_dir_name
        self.images_dir = os.path.join(data_dir, images_dir_name)
        sparse_0_dir = os.path.join(data_dir, sparse_dirname)
        self.cameras_bin_path = os.path.join(sparse_0_dir, 'cameras.bin')
        self.images_bin_path = os.path.join(sparse_0_dir, 'images.bin')
        self.points_bin_path = os.path.join(sparse_0_dir, 'points3D.bin')
        if not os.path.isdir(self.images_dir):
            raise RuntimeError(f'Error: Images directory "{self.images_dir}" does not exist.')
        if not os.path.isfile(self.cameras_bin_path):
            raise RuntimeError(f'Error: Cameras file "{self.cameras_bin_path}" does not exist.')
        if not os.path.isfile(self.cameras_bin_path):
            raise RuntimeError(f'Error: Images file "{self.images_bin_path}" does not exist.')
        if not os.path.isfile(self.cameras_bin_path):
            raise RuntimeError(f'Error: Points file "{self.points_bin_path}" does not exist.')

        self.img_width = 0
        self.img_height = 0
        self.aabb = None
        self.scale_factor = 1.0
        self.camera_model_id = 0
        self.camera_params = []
        self.camera_settings = []

        # For models see: https://github.com/colmap/colmap/blob/main/scripts/python/read_write_model.py#L57
        self.camera_models_params = [3, 4, 4, 5, 8, 8, 12, 5, 4, 5, 12]

        self.load_cameras_bin()
        self.load_points_bin()
        self.load_images_bin()

    def compute_fov(self):
        if self.camera_model_id == 0:
            # SIMPLE_PINHOLE model, 3 parameters
            focal_length_x = self.camera_params[0]
            # fovx = convert_focal_length_to_fov(focal_length_x, self.img_width)
            fovy = convert_focal_length_to_fov(focal_length_x, self.img_height)
            self.fovy = fovy
        elif self.camera_model_id == 1:
            # PINHOLE, 4 parameters
            focal_length_x = self.camera_params[0]
            focal_length_y = self.camera_params[1]
            if abs(focal_length_x - focal_length_y) > 1e-3:
                raise RuntimeError(
                    f'Mismatch in focal length for COLMAP data set ({focal_length_x}, {focal_length_y}).')
            # fovx = convert_focal_length_to_fov(focal_length_x, self.img_width)
            fovy = convert_focal_length_to_fov(focal_length_y, self.img_height)
            self.fovy = fovy
        else:
            raise RuntimeError(f'Unsupported COLMAP camera model with ID {self.camera_model_id}.')

    def load_cameras_bin(self):
        with open(self.cameras_bin_path, 'rb') as cameras_bin_file:
            num_cameras = read_u64(cameras_bin_file)
            if num_cameras != 1:
                raise RuntimeError(
                    'More than one camera configuration detected in the COLMAP data set, which is currently '
                    'unsupported.')
            for i in range(num_cameras):
                camera_id = read_i32(cameras_bin_file)
                self.camera_model_id = read_i32(cameras_bin_file)
                img_width = read_u64(cameras_bin_file)
                img_height = read_u64(cameras_bin_file)
                if self.img_width == 0 and self.img_height == 0:
                    self.img_width = img_width
                    self.img_height = img_height
                elif self.img_width != img_width or self.img_height != img_height:
                    raise RuntimeError('Mismatch in image resolution detected in the COLMAP data set.')
                num_camera_params = self.camera_models_params[self.camera_model_id]
                self.camera_params = struct.unpack(
                    '<' + ('d' * num_camera_params), cameras_bin_file.read(8 * num_camera_params))
                self.compute_fov()

    def load_images_bin(self):
        with open(self.images_bin_path, 'rb') as images_bin_file:
            num_images = read_u64(images_bin_file)
            for i in range(num_images):
                image_id = read_i32(images_bin_file)
                orientation = read_double_vec(images_bin_file, 4)
                translation = read_double_vec(images_bin_file, 3)
                translation = np.array(translation) * self.scale_factor
                rotation_matrix = matrix_quaternion(orientation)
                # We get world to camera space according to https://colmap.github.io/format.html#images-txt
                # - The "pose of an image is specified as the projection from world to the camera coordinate system".
                # - "The local camera coordinate system of an image is defined in a way that the X axis points to the
                #   right, the Y axis to the bottom, and the Z axis to the front as seen from the image."
                # By default, the coordinate system used by the synthetic NeRF datasets has the following convention:
                # - x right, y up, z back (=> y -> -z, z -> -z)
                view_matrix = matrix_translation(translation).dot(rotation_matrix)
                inverse_view_matrix = np.linalg.inv(view_matrix)
                inverse_view_matrix[0:3, 1:3] = -inverse_view_matrix[0:3, 1:3]
                view_matrix = np.linalg.inv(inverse_view_matrix)
                camera_id = read_i32(images_bin_file)  # We only support a single camera ID currently, so we skip it.
                img_name = b''
                while True:
                    next_char = struct.unpack('c', images_bin_file.read(1))[0]
                    if next_char == b'\0':
                        break
                    img_name += next_char
                img_name = img_name.decode('utf-8')

                if i == 0 and '_' in self.images_dir_name:
                    # We are likely using downscaled images ('images_<res>').
                    image_path = os.path.join(self.images_dir, img_name)
                    image_0 = load_image_array(image_path)
                    self.img_width = image_0.shape[1]
                    self.img_height = image_0.shape[0]
                    self.compute_fov()
                    del image_0

                self.camera_settings.append({'vm': view_matrix, 'img_name': img_name, 'fovy': self.fovy})
                num_points_2d = read_u64(images_bin_file)
                images_bin_file.read(24 * num_points_2d)

    def load_points_bin(self):
        with open(self.points_bin_path, 'rb') as points_bin_file:
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
        pcd_filtered, filtered_indices = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=3.0)
        # radius = np.linalg.norm(max_vec - min_vec) * 5e-3
        # pcd_filtered, filtered_indices = pcd.remove_radius_outlier(nb_points=5, radius=radius)
        # o3d.io.write_point_cloud('0_pts_filt.ply', pcd_filtered)
        o3d.visualization.draw_geometries([pcd_filtered])
        # visualize_outliers(pcd, filtered_indices)
        indices_arr = np.array(filtered_indices, dtype=int)
        points_3d_sel = points_3d[indices_arr]

        points_aligned, transform = align_axes_point_cloud(pcd_filtered, points_3d_sel)
        pcd_aligned = o3d.geometry.PointCloud()
        pcd_aligned.points = o3d.utility.Vector3dVector(points_aligned)
        o3d.visualization.draw_geometries([pcd_aligned])

        min_vec = np.min(points_aligned, axis=0)
        max_vec = np.max(points_aligned, axis=0)
        self.aabb = d.AABB3(d.vec3(min_vec[0], min_vec[1], min_vec[2]), d.vec3(max_vec[0], max_vec[1], max_vec[2]))
        self.scale_factor = get_scale_factor(self.aabb)
        self.aabb = apply_scale_factor_aabb(self.aabb, self.scale_factor)

    def get_view_matrix_array(self, idx):
        camera_dict = self.camera_settings[idx]
        vm = camera_dict['vm']
        view_matrix_array = np.empty(16)
        for k in range(4):
            for j in range(4):
                view_matrix_array[k * 4 + j] = vm[j, k]
        return view_matrix_array

    def __len__(self):
        return len(self.camera_settings)

    def __getitem__(self, idx):
        camera_dict = self.camera_settings[idx]
        image_path = os.path.join(self.images_dir, camera_dict['img_name'])
        image = torch.from_numpy(load_image_array(image_path)).cuda()
        return image, self.get_view_matrix_array(idx)

    def get_fovy(self) -> float:
        return self.fovy

    def get_aabb(self) -> d.AABB3:
        return self.aabb

    def get_img_width(self) -> int:
        return self.img_width

    def get_img_height(self) -> int:
        return self.img_height
