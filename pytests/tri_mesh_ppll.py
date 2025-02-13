# BSD 2-Clause License
#
# Copyright (c) 2025, Christoph Neuhauser
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
import numpy as np
import torch
import difftetvr as d
from datasets.sample_view import make_view_matrix

# This script shows how a per-pixel linked list (PPLL) can be used for rasterizing a triangle mesh.
# Currently, this is only supported in an experimental way for testing purposes.
# Renderer.render() currently does NOT return a valid image, as the resolve stage of the renderer is not yet supported.
# Only the fragment buffer and start offset buffer of the PPLL implementation can be queried and used.


def main():
    triangle_indices = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    vertex_positions = np.array([
        [-0.25, -0.25, 0.0], [0.1, -0.25, 0.0], [-0.1, 0.25, 0.0], [0.25, 0.25, 0.0]
    ], dtype=np.float32)
    vertex_colors = np.array([
        [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]
    ], dtype=np.float32)
    triangle_indices = torch.from_numpy(triangle_indices)
    vertex_positions = torch.from_numpy(vertex_positions)
    vertex_colors = torch.from_numpy(vertex_colors)

    img_width = 512
    img_height = 512

    tet_mesh = d.TetMesh()
    tet_mesh.set_triangle_mesh_data(triangle_indices, vertex_positions, vertex_colors)

    renderer = d.Renderer(renderer_type=d.RendererType.PPLL)
    renderer.set_export_linked_list_data()
    renderer.set_tet_mesh(tet_mesh)
    renderer.set_camera_fovy(0.5 * math.pi)
    view_matrix_array = make_view_matrix(
        camera_position=[0.0, 0.0, 0.6],
        camera_right=[1.0, 0.0, 0.0],
        camera_up=[0.0, 1.0, 0.0],
        camera_forward=[0.0, 0.0, 1.0],
    )
    renderer.set_view_matrix(view_matrix_array)
    renderer.set_viewport_size(img_width, img_height)

    _ = renderer.render()
    fragment_buffer = renderer.get_fragment_buffer()
    start_offset_buffer = renderer.get_start_offset_buffer()

    start_offset_buffer_cpu = start_offset_buffer.cpu()
    fragment_buffer_cpu = fragment_buffer.cpu()
    fragment_buffer_int32_cpu = fragment_buffer_cpu.view(dtype=torch.int32)
    offset = start_offset_buffer_cpu[img_height // 2, img_width // 2]
    fragment_depths = []
    while offset != -1:
        fragment_depths.append(fragment_buffer_cpu[offset, 1])
        offset = fragment_buffer_int32_cpu[offset, 2]
    print(f'Depths: {fragment_depths}')


if __name__ == '__main__':
    main()
