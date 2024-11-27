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
import sys
import random
import time
import argparse
import torch
import difftetvr as d
from imgutils import save_array_png


def main():
    parser = argparse.ArgumentParser(
        prog='difftetvr/render.py', description='Renders a tetrahedral mesh using direct volume rendering.')

    # Tet mesh information.
    parser.add_argument('--tet_mesh_file', type=str)
    parser.add_argument('--image_output_file', type=str)

    # Rendering settings.
    parser.add_argument('--attenuation', type=float, default=100.0)
    parser.add_argument('--img_width', type=int, default=512)
    parser.add_argument('--img_height', type=int, default=512)

    # TODO: Add camera settings (fovy, position, rotation).

    args = parser.parse_args()

    opt_tet_mesh = d.TetMesh()
    if args.opt_tet_mesh is not None:
        opt_tet_mesh.load_from_file(args.tet_mesh_file)
    else:
        raise RuntimeError('Error: No tet mesh file was specified using the argument \'--tet_mesh_file\'.')

    renderer = d.create_renderer()
    renderer.set_rendering_resolution(args.img_width, args.img_height)

    rendered_image = renderer.render()
    save_array_png(args.image_output_file, rendered_image.detach().cpu().numpy())


if __name__ == '__main__':
    main()
