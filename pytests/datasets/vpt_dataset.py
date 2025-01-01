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

import math
import torch
import torch.utils.data
import difftetvr as d
try:
    import vpt
    can_use_vpt = True
except ImportError:
    vpt = None
    can_use_vpt = False
from .dataset import Dataset3D
from .sample_view import sample_view_matrix_circle, sample_view_matrix_box
from .camera_sample_method import CameraSampleMethod


is_vpt_initialized = False


class VptDataset(torch.utils.data.Dataset, Dataset3D):
    def __init__(
            self, volume_path: str, vpt_params: dict, num_iterations: int, img_width: int, img_height: int,
            cam_sample_method: CameraSampleMethod):
        super().__init__()
        self.num_iterations = num_iterations
        self.img_width = img_width
        self.img_height = img_height
        self.cam_sample_method = cam_sample_method
        vpt.load_volume_file(volume_path)
        aabb_list = vpt.get_render_bounding_box()
        self.aabb = d.AABB3(
            d.vec3(aabb_list[0], aabb_list[2], aabb_list[4]),
            d.vec3(aabb_list[1], aabb_list[3], aabb_list[5]))
        rx = 0.5 * (self.aabb.max.x - self.aabb.min.x)
        ry = 0.5 * (self.aabb.max.y - self.aabb.min.y)
        rz = 0.5 * (self.aabb.max.z - self.aabb.min.z)
        radii_sorted = sorted([rx, ry, rz])
        self.is_spherical = radii_sorted[2] / radii_sorted[0] < 1.9
        vpt.set_camera_fovy(self.get_fovy())
        self.image_tensor = torch.zeros((4, img_height, img_width), dtype=torch.float32, device=torch.device('cuda'))
        # TODO: Use vpt_params
        self.num_samples = 256

    def __len__(self):
        return self.num_iterations

    def __getitem__(self, idx):
        if self.cam_sample_method == CameraSampleMethod.DEFAULT:
            if self.is_spherical:
                view_matrix_array, vm, ivm = sample_view_matrix_circle(self.aabb)
            else:
                view_matrix_array, vm, ivm = sample_view_matrix_box(self.aabb)
        elif self.cam_sample_method == CameraSampleMethod.CIRCLE:
            view_matrix_array, vm, ivm = sample_view_matrix_circle(self.aabb)
        elif self.cam_sample_method == CameraSampleMethod.BOX:
            view_matrix_array, vm, ivm = sample_view_matrix_box(self.aabb)
        elif self.cam_sample_method == CameraSampleMethod.REPLICATE_CPP:
            view_matrix_array, vm, ivm = sample_view_matrix_circle(self.aabb, uniform_r=True)
        else:
            raise RuntimeError('Unknown CameraSampleMethod element.')
        vpt.overwrite_camera_view_matrix(view_matrix_array)
        image = vpt.render_frame(self.image_tensor, self.num_samples)
        return image, view_matrix_array

    def get_fovy(self) -> float:
        return math.atan(1.0 / 2.0) * 2.0

    def get_aabb(self) -> d.AABB3:
        return self.aabb

    def get_img_width(self) -> int:
        return self.img_width

    def get_img_height(self) -> int:
        return self.img_height
