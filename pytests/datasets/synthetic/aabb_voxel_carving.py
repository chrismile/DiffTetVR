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
import numpy as np
from PIL import Image
import torch
import difftetvr as d


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
    return aabb
