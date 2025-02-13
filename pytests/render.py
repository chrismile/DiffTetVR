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

import argparse
import numpy as np
import torch
import difftetvr as d

from datasets.actions import RendererTypeAction
from datasets.sample_view import make_view_matrix
from datasets.imgutils import save_array_png, blend_image_premul
from datasets.images_dataset import ImagesDataset


def main():
    parser = argparse.ArgumentParser(
        prog='difftetvr/render.py', description='Renders a tetrahedral mesh using direct volume rendering.')

    # Tet mesh information.
    parser.add_argument('--tet_mesh_file', type=str)
    parser.add_argument('--image_output_file', type=str)

    # Rendering settings.
    parser.add_argument('--renderer_type', action=RendererTypeAction, default=d.RendererType.PPLL)
    parser.add_argument('--attenuation', type=float, default=100.0)
    parser.add_argument('--img_width', type=int, default=512)
    parser.add_argument('--img_height', type=int, default=512)

    # TODO: Add camera settings (fovy, position, rotation).
    parser.add_argument('--gt_images_path', type=str, default=None)
    parser.add_argument('--image_index', type=int, default=0)

    args = parser.parse_args()

    tet_mesh = d.TetMesh()
    if args.tet_mesh_file is not None:
        tet_mesh.load_from_file(args.tet_mesh_file)
    else:
        raise RuntimeError('No tet mesh file was specified using the argument \'--tet_mesh_file\'.')

    if args.gt_images_path is not None:
        dataset = ImagesDataset(args.gt_images_path)
    else:
        raise RuntimeError(
            '\'--gt_images_path\' needs to be passed to the script to specify the used ground truth data.')

    renderer = d.Renderer(renderer_type=args.renderer_type)
    renderer.set_tet_mesh(tet_mesh)
    renderer.set_attenuation(args.attenuation)
    renderer.set_clear_color(d.vec4(0.0, 0.0, 0.0, 0.0))
    renderer.set_camera_fovy(dataset.get_fovy())
    renderer.set_view_matrix(dataset.get_view_matrix_array(args.image_index))
    renderer.set_viewport_size(args.img_width, args.img_height)

    view_matrix_array = make_view_matrix(
        camera_position=[0.6, 0.0, 0.0],
        camera_right=[0.0, 0.0, -1.0],
        camera_up=[0.0, 1.0, 0.0],
        camera_forward=[1.0, 0.0, 0.0],
    )
    renderer.set_view_matrix(view_matrix_array)

    rendered_image = renderer.render()
    rendered_image_npy = rendered_image.detach().cpu().numpy()
    rendered_image_npy = rendered_image_npy[110:400, :, :]
    blend_image_premul(rendered_image_npy, [0.0, 0.0, 0.0, 1.0])
    save_array_png(args.image_output_file, np.transpose(rendered_image_npy, (2, 0, 1)))


if __name__ == '__main__':
    main()
