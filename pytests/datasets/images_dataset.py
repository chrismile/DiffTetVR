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
import numpy as np
import json
import torch
import torch.utils.data
import difftetvr as d
from .dataset import Dataset3D
from .imgutils import load_image_array


class ImagesDataset(torch.utils.data.Dataset, Dataset3D):
    def __init__(self, img_dir):
        super().__init__()

        images_dir = os.path.join(img_dir, 'images')
        cameras_path = os.path.join(img_dir, 'cameras.json')
        if not os.path.isdir(images_dir):
            raise RuntimeError(f'Error: Images directory "{images_dir}" does not exist.')
        if not os.path.isfile(cameras_path):
            raise RuntimeError(f'Error: Cameras file "{cameras_path}" does not exist.')

        with open(cameras_path) as f:
            cameras_json = json.load(f)
            camera0_json = cameras_json[0]
            if 'aabb' in camera0_json:
                self.aabb = camera0_json['aabb']
            else:
                self.aabb = d.AABB3(d.vec3(-0.5, -0.5, -0.5), d.vec3(0.5, 0.5, 0.5))
            self.img_width = int(camera0_json['width'])
            self.img_height = int(camera0_json['height'])
            self.fovy = float(camera0_json['fovy'])
            self.scale_pos = 1.0

        self.cameras_json = cameras_json
        self.images_dir = images_dir

    def get_view_matrix_array(self, idx):
        camera_json = self.cameras_json[idx]
        position = camera_json['position']
        rotation = camera_json['rotation']
        ivm = np.empty((4, 4))
        for k in range(4):
            ivm[k, 0] = rotation[0][k] if k < 3 else 0.0
            ivm[k, 1] = rotation[1][k] if k < 3 else 0.0
            ivm[k, 2] = rotation[2][k] if k < 3 else 0.0
            ivm[k, 3] = position[k] * self.scale_pos if k < 3 else 1.0
        vm = np.linalg.inv(ivm)
        view_matrix_array = np.empty(16)
        for k in range(4):
            for j in range(4):
                view_matrix_array[k * 4 + j] = vm[j, k]
        return view_matrix_array

    def __len__(self):
        return len(self.cameras_json)

    def __getitem__(self, idx):
        camera_json = self.cameras_json[idx]
        if 'fg_name' in camera_json:
            image_path = os.path.join(self.images_dir, camera_json['fg_name'])
        elif 'img_name' in camera_json:
            image_path = os.path.join(self.images_dir, camera_json['img_name'])
        else:
            raise RuntimeError('Error in ImagesDataset.__getitem__: No image entry found in .json file.')
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
