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

# The synthetic datasets from the original NeRF paper (https://www.matthewtancik.com/nerf)
# do not contain SfM data by COLMAP. While we usually use the sparse point cloud data to estimate the scene bounds,
# this is thus not possible for this data. I see three possibilities:
# - Run COLMAP to only generate sparse 3D points.
# - Use voxel carving (disadvantage: not exact, and we also need bounds for the voxel grid...).
# - Use the depth maps and the formula from: https://github.com/bmild/nerf/issues/77#issuecomment-1859165869
# This file is for testing these options.

import os
import argparse
import pathlib
from aabb_colmap import compute_aabb_colmap
from aabb_depth import compute_aabb_from_depth_images
from aabb_voxel_carving import compute_aabb_voxel_carving


def main():
    parser = argparse.ArgumentParser(
        prog='difftetvr/utils/synthetic_compute_aabb.py',
        description='Computes the AABB (axis-aligned bounding box) for the synthetic NeRF datasets.')
    parser.add_argument('--datasets_path', type=str)
    parser.add_argument('--dataset_list', nargs='+', type=str)
    parser.add_argument('--colmap_path', type=str)
    args = parser.parse_args()
    default_colmap_search_path = os.path.join(pathlib.Path.home(), 'Software/COLMAP/bin/colmap')
    if args.colmap_path is None and os.path.isfile(default_colmap_search_path):
        args.colmap_path = default_colmap_search_path
    colmap_command = args.colmap_path if args.colmap_path is not None else "colmap"

    datasets_path = args.datasets_path
    datasets_folder_default = '/media/christoph/Elements16C/Datasets/NeRF/nerf_synthetic'
    if datasets_path is None and os.path.isdir(datasets_folder_default):
        datasets_path = datasets_folder_default
    if datasets_path is None:
        raise RuntimeError('Dataset path needs to be specified using \'--datasets_path\'.')

    if args.dataset_list is None:
        dataset_list = os.listdir(datasets_path)
    for dataset_name in args.dataset_list:
        dataset_path = os.path.join(datasets_path, dataset_name)
        if os.path.isdir(dataset_path):
            aabb_colmap = compute_aabb_colmap(colmap_command, dataset_path, use_commandline=False)
            print('AABB using COLMAP (pycolmap):')
            print(f'{aabb_colmap.min.x}, {aabb_colmap.min.y}, {aabb_colmap.min.z}')
            print(f'{aabb_colmap.max.x}, {aabb_colmap.max.y}, {aabb_colmap.max.z}')

            aabb_colmap = compute_aabb_colmap(colmap_command, dataset_path, use_commandline=True)
            print('AABB using COLMAP (commandline):')
            print(f'{aabb_colmap.min.x}, {aabb_colmap.min.y}, {aabb_colmap.min.z}')
            print(f'{aabb_colmap.max.x}, {aabb_colmap.max.y}, {aabb_colmap.max.z}')

            aabb_depth = compute_aabb_from_depth_images(dataset_path)
            print('AABB from depth:')
            print(f'{aabb_depth.min.x}, {aabb_depth.min.y}, {aabb_depth.min.z}')
            print(f'{aabb_depth.max.x}, {aabb_depth.max.y}, {aabb_depth.max.z}')

            aabb_voxel_carving = compute_aabb_voxel_carving(dataset_path)
            print('AABB using voxel carving:')
            print(f'{aabb_voxel_carving.min.x}, {aabb_voxel_carving.min.y}, {aabb_voxel_carving.min.z}')
            print(f'{aabb_voxel_carving.max.x}, {aabb_voxel_carving.max.y}, {aabb_voxel_carving.max.z}')

            # Currently, only aabb_depth seems to work more or less reliably...
            aabb = aabb_depth
            with open(os.path.join(dataset_path, 'aabb.json'), 'w') as aabb_file:
                aabb_file.write(f'[{aabb.min.x}, {aabb.min.y}, {aabb.min.z}, {aabb.max.x}, {aabb.max.y}, {aabb.max.z}]')


if __name__ == '__main__':
    main()
