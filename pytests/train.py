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
import os
import random
import time
import json
import pathlib
import argparse
from enum import Enum, auto
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import difftetvr as d
from pyutils import DifferentiableRenderer
from datasets.actions import enum_action, SplitGradientTypeAction, RendererTypeAction, TestCaseAction
from datasets.camera_sample_method import CameraSampleMethod
from datasets.tet_mesh_dataset import TetMeshDataset
from datasets.regular_grid_dataset import RegularGridDataset
from datasets.vpt_dataset import VptDataset, is_vpt_initialized
from datasets.images_dataset import ImagesDataset
from datasets.colmap_dataset import ColmapDataset
from datasets.nerf_synthetic_dataset import NeRFSyntheticDataset
from datasets.imgutils import save_array_png, blend_image_premul


class InitGridType(Enum):
    HEX = auto()
    TETGEN = auto()
    FTETWILD = auto()


def replace_tensor_in_optimizer(optimizer, tensor, name):
    for group in optimizer.param_groups:
        if group['name'] == name:
            stored_state = optimizer.state.get(group['params'][0], None)
            # Zero out exponential moving average (first and second moment) if used by the optimizer.
            if 'exp_avg' in stored_state:
                stored_state['exp_avg'] = torch.zeros_like(tensor)
            if 'exp_avg_sq' in stored_state:
                stored_state['exp_avg_sq'] = torch.zeros_like(tensor)
            del optimizer.state[group['params'][0]]
            group['params'][0] = tensor
            optimizer.state[group['params'][0]] = stored_state


def copy_and_freeze_optimizer_var_state(optimizer, name):
    for group in optimizer.param_groups:
        if group['name'] == name:
            lr_copy = group['lr']
            group['lr'] = 0.0
            stored_state = optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                state = stored_state.copy()
                if 'exp_avg' in stored_state:
                    state['exp_avg'] = stored_state['exp_avg'].detach().clone()
                if 'exp_avg_sq' in stored_state:
                    state['exp_avg_sq'] = stored_state['exp_avg_sq'].detach().clone()
            else:
                state = None
            return lr_copy, state
    return None, None


def restore_optimizer_var_state(optimizer, name, lr, state):
    for group in optimizer.param_groups:
        if group['name'] == name:
            group['lr'] = lr
            stored_state = optimizer.state.get(group['params'][0], None)
            if state is not None:
                if 'exp_avg' in stored_state:
                    stored_state['exp_avg'] = state['exp_avg']
                if 'exp_avg_sq' in stored_state:
                    stored_state['exp_avg_sq'] = state['exp_avg_sq']


def main():
    parser = argparse.ArgumentParser(
        prog='difftetvr/train.py',
        description='Optimizes a tetrahedral mesh using differentiable direct volume rendering.')

    # Metadata.
    parser.add_argument('--name', type=str)  # Name of the test case (used for name of output tet file).
    parser.add_argument('-o', '--out_dir', type=str)

    # Rendering settings.
    parser.add_argument('--renderer_type', action=RendererTypeAction, default=d.RendererType.PPLL)
    parser.add_argument('--attenuation', type=float, default=100.0)
    parser.add_argument('--device_name', type=str, default=None)

    # Main optimization parameters.
    parser.add_argument('--num_epochs', '--epochs', type=int, default=1)  # Only used if not coarse-to-fine.
    parser.add_argument('--num_iterations', '--iterations', type=int, default=200)
    parser.add_argument('--loss', type=str, default='l2')
    parser.add_argument('--lr_col', type=float, default=0.04)
    parser.add_argument('--lr_pos', type=float, default=0.001)
    parser.add_argument('--fix_boundary', action='store_true', default=False)
    parser.add_argument('--exp_decay', type=float, default=0.999)
    parser.add_argument('--random_seed', type=int, default=None)

    # Tetrahedral element regularizer.
    parser.add_argument('--tet_regularizer', action='store_true', default=False)
    parser.add_argument('--tet_reg_lambda', type=float, default=0.1)
    parser.add_argument('--tet_reg_softplus_beta', type=float, default=100.0)

    # Initialization grid; from file (=> init_grid_path) or created from a constant opacity with medium gray color.
    parser.add_argument('--init_grid_path', type=str, default=None)
    parser.add_argument('--init_grid_largest', type=int, default=None)
    parser.add_argument('--init_grid_x', type=int, default=16)
    parser.add_argument('--init_grid_y', type=int, default=16)
    parser.add_argument('--init_grid_z', type=int, default=16)
    parser.add_argument('--init_grid_opacity', default=0.1)
    parser.add_argument(
        '--init_grid_type', default=InitGridType.HEX, action=enum_action(InitGridType))

    # Coarse-to-fine strategy.
    parser.add_argument('--coarse_to_fine', action='store_true', default=False)
    parser.add_argument('--max_num_tets', type=int, default=60_000)
    parser.add_argument(
        '--split_grad_type', action=SplitGradientTypeAction, default=d.SplitGradientType.ABS_COLOR)
    parser.add_argument('--splits_ratio', type=float, default=0.05)
    parser.add_argument('--coarse_to_fine_save_intermediate', action='store_true', default=False)
    parser.add_argument('--coarse_to_fine_log_gradients', action='store_true', default=False)

    # Ground truth data sources.
    # Test case (A): Render tet dataset as ground truth.
    parser.add_argument('--gt_grid_path', type=str, default=None)
    parser.add_argument('--gt_grid_test_case', action=TestCaseAction, default=None)
    parser.add_argument('--img_width', type=int, default=512)
    parser.add_argument('--img_height', type=int, default=512)
    parser.add_argument(
        '--cam_sample_method', default=CameraSampleMethod.DEFAULT, action=enum_action(CameraSampleMethod))
    parser.add_argument('--gt_tf', type=str, default=None)  # Optional if regular grid is used as GT
    # Test case (B): Use images from disk as ground truth.
    parser.add_argument('--gt_images_path', type=str, default=None)
    parser.add_argument('--gt_colmap_data_path', type=str, default=None)
    parser.add_argument('--colmap_sparse_dirname', type=str, default='sparse/0')
    parser.add_argument('--gt_nerf_synthetic_data_path', type=str, default=None)
    parser.add_argument('--image_folder_name', type=str, default=None)

    # Debugging options.
    parser.add_argument('--record_video', action='store_true', default=False)
    parser.add_argument('--save_statistics', action='store_true', default=False)

    args = parser.parse_args()
    if args.random_seed is not None:
        random.seed(args.random_seed)
    if args.out_dir is None:
        raise RuntimeError('Missing output directory. Please specify it using \'--out_dir\'.')
    if not os.path.isdir(args.out_dir):
        pathlib.Path(args.out_dir).mkdir(parents=False, exist_ok=True)

    if args.device_name is not None:
        d.set_device_type(torch.device(args.device_name).type)
    # d.set_device_type(torch.device('cpu').type)

    renderer = d.Renderer(renderer_type=args.renderer_type)
    renderer.set_attenuation(args.attenuation)
    renderer.set_clear_color(d.vec4(0.0, 0.0, 0.0, 0.0))

    # Create the initialization grid (1/2).
    tet_mesh_opt = d.TetMesh()
    tet_mesh_opt.set_use_gradients(True)
    if args.coarse_to_fine:
        tet_mesh_opt.set_force_use_ovm_representation()
    if args.init_grid_path is not None:
        print(f'Loading initialization tet mesh from file {args.init_grid_path}...')
        tet_mesh_opt.load_from_file(args.init_grid_path)
        num_tets_init = tet_mesh_opt.get_num_cells()
    else:
        if args.init_grid_largest is not None:
            # We estimate num_tets_init, just to get an estimate for the shared fragment buffers for the PPLL renderer.
            num_tets_init = int(args.init_grid_largest ** 3)
        else:
            num_tets_init = args.init_grid_x * args.init_grid_y * args.init_grid_z

    # Create the ground truth data set.
    tet_mesh_gt = None
    if args.gt_grid_path is not None:
        _, file_extension = os.path.splitext(args.gt_grid_path)
        tet_mesh_extensions = ['.bintet', '.txt', '.ovm', '.ovmb', '.vtk']
        if file_extension in tet_mesh_extensions:
            tet_mesh_gt = d.TetMesh()
            tet_mesh_gt.load_from_file(args.gt_grid_path)
            dataset = TetMeshDataset(
                tet_mesh_gt, args.num_iterations, renderer, args.coarse_to_fine, args.max_num_tets, num_tets_init,
                args.img_width, args.img_height, args.cam_sample_method)
        else:
            regular_grid = d.RegularGrid()
            regular_grid.load_from_file(args.gt_grid_path)
            dataset = RegularGridDataset(
                regular_grid, args.num_iterations, args.attenuation, args.coarse_to_fine, args.max_num_tets,
                args.img_width, args.img_height, args.gt_tf, args.cam_sample_method)
    elif args.gt_grid_test_case is not None:
        tet_mesh_gt = d.TetMesh()
        tet_mesh_gt.load_test_data(args.gt_grid_test_case)
        dataset = TetMeshDataset(
            tet_mesh_gt, args.num_iterations, renderer, args.coarse_to_fine, args.max_num_tets, num_tets_init,
            args.img_width, args.img_height, args.cam_sample_method)
    elif args.gt_images_path is not None:
        dataset = ImagesDataset(args.gt_images_path)
    elif args.gt_colmap_data_path is not None:
        if args.image_folder_name is None:
            args.image_folder_name = 'images'
        dataset = ColmapDataset(
            args.gt_colmap_data_path, images_dir_name=args.image_folder_name,
            sparse_dirname=args.colmap_sparse_dirname)
    elif args.gt_nerf_synthetic_data_path is not None:
        if args.image_folder_name is None:
            args.image_folder_name = 'train'
        dataset = NeRFSyntheticDataset(args.gt_nerf_synthetic_data_path, images_dir_name=args.image_folder_name)
    else:
        raise RuntimeError(
            'Either \'--gt_grid_path\', \'--gt_images_path\', \'--gt_colmap_data_path\' or '
            '\'--gt_nerf_synthetic_data_path\' needs to be passed to the script to specify the used ground truth data.')
    img_width = dataset.get_img_width()
    img_height = dataset.get_img_height()
    data_loader = DataLoader(dataset, batch_size=None)

    # Create the initialization grid (2/2).
    if args.init_grid_path is None:
        aabb = dataset.get_aabb()

        # Use grid with approximately even cell sizes.
        if args.init_grid_largest is not None:
            aabb_dimensions = aabb.get_dimensions()
            max_dim = np.max(np.array([aabb_dimensions.x, aabb_dimensions.y, aabb_dimensions.z]))
            args.init_grid_x = math.ceil(aabb_dimensions.x / max_dim * args.init_grid_largest)
            args.init_grid_y = math.ceil(aabb_dimensions.y / max_dim * args.init_grid_largest)
            args.init_grid_z = math.ceil(aabb_dimensions.z / max_dim * args.init_grid_largest)

        print(f'Creating initialization grid of size {args.init_grid_x}x{args.init_grid_y}x{args.init_grid_z}...')
        const_color = d.vec4(0.5, 0.5, 0.5, args.init_grid_opacity)
        if args.init_grid_type == InitGridType.HEX:
            tet_mesh_opt.set_hex_mesh_const(aabb, args.init_grid_x, args.init_grid_y, args.init_grid_z, const_color)
        elif args.init_grid_type == InitGridType.FTETWILD:
            params = d.FTetWildParams()
            tet_mesh_opt.set_tetrahedralized_grid_ftetwild(
                aabb, args.init_grid_x, args.init_grid_y, args.init_grid_z, const_color, params)
        elif args.init_grid_type == InitGridType.TETGEN:
            params = d.TetGenParams()
            tet_mesh_opt.set_tetrahedralized_grid_tetgen(
                aabb, args.init_grid_x, args.init_grid_y, args.init_grid_z, const_color, params)
    print(f'#Cells (init): {tet_mesh_opt.get_num_cells()}')
    print(f'#Vertices (init): {tet_mesh_opt.get_num_vertices()}')

    renderer.set_tet_mesh(tet_mesh_opt)
    diff_renderer = DifferentiableRenderer(
        renderer, args.tet_regularizer, args.tet_reg_lambda, args.tet_reg_softplus_beta)
    renderer.set_camera_fovy(dataset.get_fovy())
    if tet_mesh_gt is None:
        renderer.set_coarse_to_fine_target_num_tets(args.max_num_tets)
        renderer.set_viewport_size(img_width, img_height)
    else:
        renderer.set_viewport_size(img_width, img_height, False)
        renderer.reuse_intermediate_buffers_from(dataset.renderer)

    variables = []
    vertex_colors = tet_mesh_opt.get_vertex_colors()
    vertex_positions = tet_mesh_opt.get_vertex_positions()
    vertex_boundary_bit_tensor = tet_mesh_opt.get_vertex_boundary_bit_tensor()
    if args.lr_col > 0.0:
        variables.append({'params': [vertex_colors], 'lr': args.lr_col, 'name': 'vertex_colors'})
    if args.lr_pos > 0.0:
        variables.append(
            {'params': [vertex_positions], 'lr': args.lr_pos, 'name': 'vertex_positions'})
    optimizer = torch.optim.Adam(variables)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    loss_name = args.loss.lower()
    if loss_name == 'l1':
        loss_fn = torch.nn.L1Loss()
    elif loss_name == 'l2':
        loss_fn = torch.nn.MSELoss()
    else:
        raise RuntimeError(f'Unknown loss name \'{loss_name}\'.')

    if args.record_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # fourcc = cv2.VideoWriter_fourcc(*'avc1')
        video_out_path = os.path.join(args.out_dir, f'{args.name}.mp4')
        video = cv2.VideoWriter(video_out_path, fourcc, 5.0, (img_width, img_height))

    if args.coarse_to_fine_log_gradients:
        vtk_file_path = os.path.join(args.out_dir, f'{args.name}.vtk')
        vtk_writer = d.TetMeshVtkWriter(vtk_file_path)

    def training_step(view_matrix_array, optimizer_step=True):
        nonlocal vertex_colors
        if optimizer_step:
            optimizer.zero_grad(set_to_none=False)
            tet_mesh_opt.on_zero_grad()  # Necessary for CPU device to propagate zeros to Vulkan buffers.
        renderer.set_view_matrix(view_matrix_array.numpy())
        image_opt = diff_renderer()
        loss = loss_fn(image_opt, image_gt)
        loss.backward()
        if optimizer_step:
            if args.fix_boundary:
                vertex_positions.grad = torch.where(vertex_boundary_bit_tensor > 0, 0.0, vertex_positions.grad)
            optimizer.step()
            with torch.no_grad():
                vertex_colors -= torch.min(vertex_colors, torch.zeros_like(vertex_colors))
            tet_mesh_opt.set_vertices_changed()
        if args.record_video and optimizer_step:
            with torch.no_grad():
                img_numpy = np.clip(image_opt.detach().cpu().numpy(), 0.0, 1.0) * 255.0
                img_numpy = img_numpy.astype(np.uint8)
                video.write(cv2.cvtColor(img_numpy[:, :, 0:3], cv2.COLOR_BGR2RGB))
                del img_numpy

    print('Starting optimization...')
    time_start = time.time()
    num_splits = 0
    if args.coarse_to_fine:
        use_accum_abs_grads = \
            args.split_grad_type == d.SplitGradientType.ABS_POSITION or \
            args.split_grad_type == d.SplitGradientType.ABS_COLOR
        while True:
            # Train colors (freeze position state and set learning rate to zero).
            print('Optimizing colors...')
            position_lr, position_state = copy_and_freeze_optimizer_var_state(optimizer, 'vertex_positions')
            for image_gt, view_matrix_array in data_loader:
                training_step(view_matrix_array)

            # Train positions + colors (set old position state).
            print('Optimizing positions + colors...')
            restore_optimizer_var_state(optimizer, 'vertex_positions', position_lr, position_state)
            for image_gt, view_matrix_array in data_loader:
                training_step(view_matrix_array)

            if tet_mesh_opt.get_num_cells() >= args.max_num_tets:
                break

            if args.coarse_to_fine_save_intermediate:
                mesh_out_path = os.path.join(args.out_dir, f'{args.name}_{tet_mesh_opt.get_num_cells()}.bintet')
                print(f'Saving intermediate mesh to "{mesh_out_path}"...')
                tet_mesh_opt.save_to_file(mesh_out_path)

            # Accumulate gradients.
            print('Accumulating gradients...')
            if use_accum_abs_grads:
                diff_renderer.set_use_abs_grad(True)
            optimizer.zero_grad(set_to_none=False)
            tet_mesh_opt.on_zero_grad()  # Necessary for CPU device to propagate zeros to Vulkan buffers.
            for image_gt, view_matrix_array in data_loader:
                training_step(view_matrix_array, False)
            if use_accum_abs_grads:
                diff_renderer.set_use_abs_grad(False)
            if args.coarse_to_fine_log_gradients:
                vtk_writer.write_next_time_step(tet_mesh_opt)

            # Split the tets incident to the vertices with the largest gradients.
            tet_mesh_opt.split_by_largest_gradient_magnitudes(renderer, args.split_grad_type, args.splits_ratio)
            vertex_colors = tet_mesh_opt.get_vertex_colors()
            vertex_positions = tet_mesh_opt.get_vertex_positions()
            vertex_boundary_bit_tensor = tet_mesh_opt.get_vertex_boundary_bit_tensor()
            replace_tensor_in_optimizer(optimizer, vertex_colors, 'vertex_colors')
            replace_tensor_in_optimizer(optimizer, vertex_positions, 'vertex_positions')
            num_splits += 1

            scheduler.step()
            # if epoch_idx % args.num_epochs == args.num_epochs - 1:
            #    scheduler.step()
            #    for param_group in optimizer.param_groups:
            #        lr = scheduler(epoch_idx)
            #        param_group['lr'] = lr
    else:
        for epoch_idx in range(args.num_epochs):
            for image_gt, view_matrix_array in data_loader:
                training_step(view_matrix_array)
            scheduler.step()
    time_end = time.time()
    print(f'Optimization finished ({time_end - time_start}s).')

    if args.coarse_to_fine_save_intermediate:
        mesh_out_path = os.path.join(args.out_dir, f'{args.name}_{tet_mesh_opt.get_num_cells()}.bintet')
    else:
        mesh_out_path = os.path.join(args.out_dir, f'{args.name}.bintet')
    print(f'Saving mesh to "{mesh_out_path}"...')
    tet_mesh_opt.save_to_file(mesh_out_path)
    if args.record_video:
        video.release()
    if args.save_statistics:
        stats = dict()
        stats['opt_time'] = time_end - time_start
        stats['num_tets'] = tet_mesh_opt.get_num_cells()
        stats['num_vertices'] = tet_mesh_opt.get_num_vertices()
        stats['num_splits'] = num_splits
        statistics_file_path = os.path.join(args.out_dir, f'{args.name}.json')
        with open(statistics_file_path, 'w') as f:
            json.dump(stats, f, ensure_ascii=False, indent=4)
    print('All done.')


if __name__ == '__main__':
    main()
