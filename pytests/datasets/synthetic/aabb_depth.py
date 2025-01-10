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

import os
import json
import math
import numpy as np
from numba import njit
from PIL import Image
import difftetvr as d


# Used by @see compute_aabb_from_depth_images to get the min/max world position for a depth map.
@njit
def get_min_max_world_range(depth_image, valid_mask, cam_to_world, fovx, min_vec, max_vec):
    width = depth_image.shape[1]
    height = depth_image.shape[0]
    x0_max = math.tan(0.5 * fovx)
    y0_max = x0_max * height / width
    point = np.ones(4)
    for y in range(height):
        # y in the images points down, so use negative y coordinate.
        y_coord = -2.0 * (y + 0.5) / height + 1.0
        for x in range(width):
            if valid_mask[y, x]:
                x_coord = 2.0 * (x + 0.5) / width - 1.0
                depth = depth_image[y, x]
                point[0] = x0_max * x_coord * depth
                point[1] = y0_max * y_coord * depth
                point[2] = -depth
                point_world = np.dot(cam_to_world, point)
                for i in range(3):
                    min_vec[i] = min(min_vec[i], point_world[i])
                    max_vec[i] = max(max_vec[i], point_world[i])
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
    return d.AABB3(d.vec3(min_vec[0], min_vec[1], min_vec[2]), d.vec3(max_vec[0], max_vec[1], max_vec[2]))
