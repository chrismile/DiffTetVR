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
import json
import numpy as np
import torch
import torch.utils.data
import difftetvr as d
from .dataset import Dataset3D
from .imgutils import load_image_array
from .sample_view import matrix_translation, matrix_quaternion, convert_focal_length_to_fov, get_scale_factor, \
    apply_scale_factor_aabb


class NeRFSyntheticDataset(torch.utils.data.Dataset, Dataset3D):
    """
    Loads the synthetic data sets from: https://www.matthewtancik.com/nerf
    NOTE: synthetic_compute_aabb.py needs to be run in advance to generate AABB information.
    """

    def __init__(self, data_dir, images_dir_name='train'):
        super().__init__()

        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, images_dir_name)

        with open(os.path.join(data_dir, f'transforms_{images_dir_name}.json')) as json_file:
            transforms_json = json.load(json_file)
            fovx = transforms_json['camera_angle_x']
            self.frames = transforms_json['frames']
            frame_0 = self.frames[0]
            frame_0_file_path = frame_0['file_path'] + '.png'
            image_0_path = os.path.join(data_dir, frame_0_file_path)
            image_0 = load_image_array(image_0_path)
            self.img_width = image_0.shape[1]
            self.img_height = image_0.shape[0]
            self.fovy = fovx * self.img_height / self.img_width
            del image_0

        aabb_file_path = os.path.join(data_dir, f'aabb.json')
        if os.path.isfile(aabb_file_path):
            with open(aabb_file_path) as aabb_file:
                aabb_json = json.load(aabb_file)
                self.aabb = d.AABB3(
                    d.vec3(aabb_json[0], aabb_json[1], aabb_json[2]),
                    d.vec3(aabb_json[3], aabb_json[4], aabb_json[5]))
        else:
            # TODO: Add this step automatically.
            raise RuntimeError('Run utils/synthetic_compute_aabb.py first to pre-create AABB information.')
        self.scale_factor = get_scale_factor(self.aabb)
        self.aabb = apply_scale_factor_aabb(self.aabb, self.scale_factor)

    def get_view_matrix_array(self, idx):
        frame = self.frames[idx]
        inv_view_matrix = np.array(frame['transform_matrix'])
        inv_view_matrix[0:3, 3] = self.scale_factor * inv_view_matrix[0:3, 3]
        view_matrix = np.linalg.inv(inv_view_matrix)
        view_matrix_array = np.empty(16)
        for k in range(4):
            for j in range(4):
                view_matrix_array[k * 4 + j] = view_matrix[j, k]
        return view_matrix_array

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        image_path = os.path.join(self.data_dir, frame['file_path'] + '.png')
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
