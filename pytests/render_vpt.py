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
import datetime
import queue
import json
import pathlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rotation
import torch
import array
import vpt
import difftetvr as d
import time
import argparse
from datasets.sample_view import *
from datasets.imgutils import *

# Bayesian optimization
# conda install -c conda-forge bayesian-optimization
try:
    from bayes_opt import BayesianOptimization, UtilityFunction
    can_use_bayes_opt = True
except ImportError:
    BayesianOptimization = None
    UtilityFunction = None
    can_use_bayes_opt = False
try:
    import pylimbo
    can_use_pylimbo = True
except ImportError:
    pylimbo = None
    can_use_pylimbo = False


#def save_nc(file_path, data):
#    zs = data.shape[0]
#    ys = data.shape[1]
#    xs = data.shape[2]
#    ncfile = Dataset(file_path, mode='w', format='NETCDF4_CLASSIC')
#    zdim = ncfile.createDimension('z', zs)
#    ydim = ncfile.createDimension('y', ys)
#    xdim = ncfile.createDimension('x', xs)
#    outfield_den = ncfile.createVariable('density', np.float32, ('z', 'y', 'x'))
#    outfield_den[:, :, :] = data[:, :, :]
#    ncfile.close()


class Plane:
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def get_normal(self):
        return np.array([self.a, self.b, self.c])

    def get_distance(self, pt):
        return self.a * pt[0] + self.b * pt[1] + self.c * pt[2] + self.d

    def is_outside(self, aabb):
        aabb_center = 0.5 * np.array([aabb[1] + aabb[0], aabb[3] + aabb[2], aabb[5] + aabb[4]])
        extent = 0.5 * np.array([aabb[1] - aabb[0], aabb[3] - aabb[2], aabb[5] - aabb[4]])
        center_dist = self.get_distance(aabb_center)
        max_abs_dist = abs(self.a * extent[0] + self.b * extent[1] + self.c * extent[2])
        return -center_dist > max_abs_dist


def check_aabb_visible_in_view_frustum(vp_matrix, aabb):
    # The underlying idea of the following code comes from
    # http://www.lighthouse3d.com/tutorials/view-frustum-culling/clip-space-approach-implementation-details/
    frustum_planes = []

    # Near plane
    frustum_planes.append(Plane(
            vp_matrix[3, 0] + vp_matrix[2, 0],
            vp_matrix[3, 1] + vp_matrix[2, 1],
            vp_matrix[3, 2] + vp_matrix[2, 2],
            vp_matrix[3, 3] + vp_matrix[2, 3]))

    # Far plane
    frustum_planes.append(Plane(
            vp_matrix[3, 0] - vp_matrix[2, 0],
            vp_matrix[3, 1] - vp_matrix[2, 1],
            vp_matrix[3, 2] - vp_matrix[2, 2],
            vp_matrix[3, 3] - vp_matrix[2, 3]))

    # Left plane
    frustum_planes.append(Plane(
            vp_matrix[3, 0] + vp_matrix[0, 0],
            vp_matrix[3, 1] + vp_matrix[0, 1],
            vp_matrix[3, 2] + vp_matrix[0, 2],
            vp_matrix[3, 3] + vp_matrix[0, 3]))

    # Right plane
    frustum_planes.append(Plane(
            vp_matrix[3, 0] - vp_matrix[0, 0],
            vp_matrix[3, 1] - vp_matrix[0, 1],
            vp_matrix[3, 2] - vp_matrix[0, 2],
            vp_matrix[3, 3] - vp_matrix[0, 3]))

    # Bottom plane
    frustum_planes.append(Plane(
            vp_matrix[3, 0] + vp_matrix[1, 0],
            vp_matrix[3, 1] + vp_matrix[1, 1],
            vp_matrix[3, 2] + vp_matrix[1, 2],
            vp_matrix[3, 3] + vp_matrix[1, 3]))

    # Top plane
    frustum_planes.append(Plane(
            vp_matrix[3, 0] - vp_matrix[1, 0],
            vp_matrix[3, 1] - vp_matrix[1, 1],
            vp_matrix[3, 2] - vp_matrix[1, 2],
            vp_matrix[3, 3] - vp_matrix[1, 3]))

    # Normalize parameters
    for i in range(6):
        normal_length = np.linalg.norm(frustum_planes[i].get_normal())
        frustum_planes[i].a /= normal_length
        frustum_planes[i].b /= normal_length
        frustum_planes[i].c /= normal_length
        frustum_planes[i].d /= normal_length

    # Not visible if all points on negative side of one plane
    for i in range(6):
        if frustum_planes[i].is_outside(aabb):
            return False
    return True


def check_camera_is_valid(occupation_volume, aabb, view_matrix, inverse_view_matrix, fovy, aspect):
    min_pos = np.array([aabb[0], aabb[2], aabb[4]])
    max_pos = np.array([aabb[1], aabb[3], aabb[5]])

    # Test if the camera does not lie in an occupied voxel.
    occupation_volume_shape = np.array(
        [occupation_volume.shape[2], occupation_volume.shape[1], occupation_volume.shape[0]], dtype=np.int32)
    camera_position = inverse_view_matrix[0:3, 3]
    camera_position = (camera_position - min_pos) / (max_pos - min_pos)
    camera_position = camera_position * occupation_volume_shape
    voxel_position = np.empty(3, dtype=np.int32)
    is_outside_volume = False
    outside_dist = 100000
    for i in range(3):
        voxel_position[i] = int(camera_position[i])
        if voxel_position[i] < 0 or voxel_position[i] >= occupation_volume_shape[i]:
            is_outside_volume = True
            if voxel_position[i] < 0:
                outside_dist = min(-voxel_position[i], outside_dist)
            elif voxel_position[i] >= occupation_volume_shape[i]:
                outside_dist = min(voxel_position[i] - occupation_volume_shape[i], outside_dist)
    if not is_outside_volume and occupation_volume[voxel_position[2], voxel_position[1], voxel_position[0]] != 0:
        return False

    max_outside_dist = 3
    if is_outside_volume and outside_dist < max_outside_dist:
        max_dist = outside_dist + 2
        visited_points = set()
        voxel_queue = queue.Queue()
        voxel_queue.put((0, (voxel_position[0], voxel_position[1], voxel_position[2])))
        found_neigh = False
        best_neigh_depth = 10000
        # Dist to occupied voxel.
        while not voxel_queue.empty():
            depth, curr_pos = voxel_queue.get()
            for oz in range(-1, 2):
                for oy in range(-1, 2):
                    for ox in range(-1, 2):
                        neigh_pos = (curr_pos[0] + ox, curr_pos[1] + oy, curr_pos[2] + oz)
                        is_neighbor_outside_volume = False
                        for i in range(3):
                            if neigh_pos[i] < 0 or neigh_pos[i] >= occupation_volume_shape[i]:
                                is_neighbor_outside_volume = True
                        if neigh_pos in visited_points:
                            continue
                        if not is_neighbor_outside_volume:
                            if occupation_volume[neigh_pos[2], neigh_pos[1], neigh_pos[0]] != 0:
                                found_neigh = True
                                best_neigh_depth = min(best_neigh_depth, depth + 1)
                        if depth < max_dist:
                            voxel_queue.put((depth + 1, neigh_pos))
                        visited_points.add(neigh_pos)
        if found_neigh and best_neigh_depth <= max_dist:
            return False

    # Test if the AABB is visible in the camera view frustum.
    projection_matrix = build_projection_matrix(fovy, aspect)
    vp_matrix = projection_matrix.dot(view_matrix)
    if not check_aabb_visible_in_view_frustum(vp_matrix, aabb):
        return False

    return True


def animate_envmap_0(t):
    axis = [0.0, 1.0, 0.0]
    angle = t * 2.0 * np.pi
    vpt.set_env_map_rot_angle_axis(axis, angle)


def animate_envmap_1(t):
    t_vert_10 = abs(2.0 * t - 1)
    t_vert_01 = 1 - t_vert_10
    a = t * 2.0 * np.pi
    b = t_vert_01 * np.pi * 0.25
    vpt.set_env_map_rot_yaw_pitch_roll([a, 0.0, b])
    vpt.set_environment_map_intensity_rgb(1.0, 0.5 + 0.5 * t_vert_10, 0.25 + 0.75 * t_vert_10)


def animate_envmap_2(t):
    t_inv = 1.0 - t
    #yaw_start = 1.5
    #yaw_end = 0.912
    #pitch_start = 0.9
    #pitch_end = 1.6
    a = t * np.pi
    b = t * np.pi * 0.25
    vpt.set_env_map_rot_euler_angles([a, 0.0, b])
    vpt.set_environment_map_intensity_rgb([1.0, 0.5 + 0.5 * t_inv, 0.25 + 0.75 * t_inv])


def animate_envmap_3(t):
    vpt.set_env_map_rot_yaw_pitch_roll([-0.709, -1.570, 0.0])
    vpt.set_environment_map_intensity(2.0)


if __name__ == '__main__':
    default_envmap = \
        str(pathlib.Path.home()) \
        + '/Programming/C++/CloudRendering/Data/CloudDataSets/env_maps/small_empty_room_1_4k_blurred_large.exr'

    parser = argparse.ArgumentParser(
        prog='vpt_denoise',
        description='Generates volumetric path tracing images.')
    parser.add_argument('-t', '--test_case', default='Wholebody')
    parser.add_argument('-f', '--file')
    parser.add_argument('-r', '--img_res', type=int, default=1024)
    parser.add_argument('-n', '--num_frames', type=int)  # , default=128
    parser.add_argument('-s', '--num_samples', type=int, default=256)
    parser.add_argument('-c', '--camposes')
    parser.add_argument('-o', '--out_dir')
    parser.add_argument('--use_const_seed', action='store_true', default=False)
    parser.add_argument('--envmap', default=default_envmap)
    parser.add_argument('--envmap_intensity', type=float)
    parser.add_argument('--envmap_rot_camera', action='store_true', default=False)  # Whether to rotate envmap with camera.
    parser.add_argument('--animate_envmap', type=int)
    parser.add_argument('--time', type=float)  # Between 0 and 1; uses constant time for animate_envmap.
    parser.add_argument('--global_bbox', type=float, nargs='+')  # Between 0 and 1; uses constant time for animate_envmap.
    parser.add_argument('--use_headlight', action='store_true', default=False)
    parser.add_argument('--use_lights', action='store_true', default=False)
    parser.add_argument('--use_lights_from')
    parser.add_argument('--use_black_bg', action='store_true', default=False)
    parser.add_argument('--denoiser')
    parser.add_argument('--exr', action='store_true', default=False)  # Save .exr images.
    parser.add_argument('--pytorch_denoiser_model_file')  # Only if denoiser name starts with 'PyTorch'
    parser.add_argument('--denoiser_settings', metavar="KEY=VALUE", nargs='+')
    parser.add_argument('--write_performance_info', action='store_true', default=False)
    parser.add_argument('--device_idx', type=int, default=0)
    parser.add_argument('--debug', action='store_true', default=False)
    # Custom settings.
    parser.add_argument('--render_mode')
    parser.add_argument('--transfer_function')
    parser.add_argument('--transfer_function_range_min', type=float, default=0.0)
    parser.add_argument('--transfer_function_range_max', type=float)
    parser.add_argument('--transfer_function_grad')
    parser.add_argument('--transfer_function_grad_range_min', type=float, default=0.0)
    parser.add_argument('--transfer_function_grad_range_max', type=float)
    parser.add_argument('--brdf')
    parser.add_argument('--brdf_parameters', metavar="KEY=VALUE", nargs='+')
    parser.add_argument('--iso_value', type=float)
    parser.add_argument('--scattering_albedo', type=float, default=0.99)
    parser.add_argument('--extinction_scale', type=float, default=400.0)
    parser.add_argument('--scale_pos', type=float, default=0.5)
    args = parser.parse_args()

    test_case = args.test_case
    #test_case = 'Wholebody'
    #test_case = 'Angiography'
    #test_case = 'HeadDVR'
    #test_case = 'HollowSphere'
    #test_case = 'Cloud'
    #test_case = 'Cloud Fog'
    #test_case = 'Brain'
    if args.file is not None:
        test_case = 'Custom'
        #if args.num_files is None:
        #    custom_files.append(args.file)
        #else:
        #    for i in range(args.num_files):
        #        custom_files.append(args.file % i)

    if args.use_const_seed:
        random.seed(31)
        pylimbo.seed_random(37)
    cuda_device_idx = args.device_idx
    vulkan_device_idx = args.device_idx
    cpu_device = torch.device('cpu')
    cuda_device = torch.device(f'cuda:{cuda_device_idx}' if torch.cuda.is_available() else 'cpu')
    vulkan_device = torch.device(f'vulkan:{vulkan_device_idx}' if torch.is_vulkan_available() else 'cpu')

    #print(torch.__version__)
    #print(torch.is_vulkan_available())
    #print(torch.cuda.is_available())
    #print(cpu_device)
    #print(cuda_device)
    #print(vulkan_device)

    test_mode = False
    image_width = args.img_res
    image_height = args.img_res
    aspect = image_width / image_height

    test_tensor_cpu = torch.ones((4, image_height, image_width), dtype=torch.float32, device=cpu_device)
    test_tensor_cuda = torch.ones((4, image_height, image_width), dtype=torch.float32, device=cuda_device)
    #test_tensor_vulkan = torch.ones(1, dtype=torch.float32, device=vulkan_device)
    #test_tensor_vulkan = test_tensor_cpu.to(vulkan_device)
    print(test_tensor_cpu)
    print(test_tensor_cuda)
    #print(test_tensor_vulkan)
    vpt.initialize()

    if args.global_bbox is not None:
        vpt.set_global_world_bounding_box(args.global_bbox)

    gaussian_splatting_data = True
    use_png_format = gaussian_splatting_data
    if args.exr:
        use_png_format = False
    if args.out_dir is None:
        if os.name == 'nt':
            # Windows does not support ':' in filenames.
            out_dir = f'out_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}'
        else:
            out_dir = f'out_{datetime.datetime.now():%Y-%m-%d_%H:%M:%S}'
    else:
        out_dir = args.out_dir
        if out_dir[-1] == '/' or out_dir[-1] == '\\':
            out_dir = out_dir[:-1]
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    if gaussian_splatting_data:
        pathlib.Path(f'{out_dir}/images').mkdir(exist_ok=True)
    #with open(f'{out_dir}/extrinsics.txt', 'w') as f:
    #    f.write(f'{vpt.get_camera_fovy()}')
    camera_infos = []

    use_python_bos_optimizer = False

    data_dir = '/mnt/data/Flow/Scalar/'
    if not os.path.isdir(data_dir):
        data_dir = '/home/christoph/datasets/Scalar/'
    if not os.path.isdir(data_dir):
        data_dir = '/media/christoph/Elements/Datasets/Scalar/'
    if not os.path.isdir(data_dir):
        data_dir = '/home/christoph/datasets/Flow/Scalar/'
    if test_case == 'Custom':
        vpt.load_volume_file(args.file)
    elif test_case == 'Wholebody':
        vpt.load_volume_file(
            data_dir + 'Wholebody [512 512 3172] (CT)/wholebody.dat')
    elif test_case == 'Angiography':
        vpt.load_volume_file(
            data_dir + 'Head [416 512 112] (MRT Angiography)/mrt8_angio.dat')
    elif test_case == 'HeadDVR':
        vpt.load_volume_file(
            data_dir + 'Head [256 256 256] (MR)/Head_256x256x256.dat')
    elif test_case == 'HollowSphere':
        vpt.load_volume_file(
            str(pathlib.Path.home()) + '/datasets/Toy/vpt/hollowsphere.dat')
    elif test_case == 'Cloud' or test_case == 'Cloud Fog':
        vpt.load_volume_file(
            str(pathlib.Path.home()) + '/Programming/C++/CloudRendering/Data/CloudDataSets/wdas_cloud/wdas_cloud.vdb')
    elif test_case == 'Brain':
        vpt.load_volume_file(
            str(pathlib.Path.home()) + '/datasets/Siemens/brain_cleaned/23.42um_4_cleaned.dat')
    elif test_case == 'ToothIso':
        vpt.load_volume_file(
            data_dir + 'Tooth [256 256 161](CT)/tooth_cropped.dat')
    vpt.load_environment_map(args.envmap)
    if args.envmap_intensity is not None:
        vpt.set_environment_map_intensity(args.envmap_intensity)
    vpt.set_use_transfer_function(True)
    vpt.set_use_lights(args.use_lights)
    if args.use_lights_from is not None:
        vpt.set_use_lights(True)
        vpt.load_lights_from_file(args.use_lights_from)

    #mode = 'Delta Tracking'
    mode = 'Next Event Tracking'
    #mode = 'Isosurfaces'
    if test_case == 'Wholebody':
        vpt.load_transfer_function_file(
            str(pathlib.Path.home()) + '/Programming/C++/CloudRendering/Data/TransferFunctions/TF_Wholebody3.xml')
        vpt.load_transfer_function_file_gradient(
            str(pathlib.Path.home()) + '/Programming/C++/CloudRendering/Data/TransferFunctions/TF_WholebodyGrad1.xml')
    elif test_case == 'Angiography':
        vpt.load_transfer_function_file(
            str(pathlib.Path.home()) + '/Programming/C++/CloudRendering/Data/TransferFunctions/HeadAngioDens.xml')
        vpt.load_transfer_function_file_gradient(
            str(pathlib.Path.home()) + '/Programming/C++/CloudRendering/Data/TransferFunctions/HeadAngioGrad.xml')
    elif test_case == 'HeadDVR':
        vpt.load_transfer_function_file(
            str(pathlib.Path.home()) + '/Programming/C++/CloudRendering/Data/TransferFunctions/HeadDVR.xml')
        mode = 'Ray Marching (Emission/Absorption)'
    elif test_case == 'HollowSphere':
        vpt.load_transfer_function_file(
            str(pathlib.Path.home()) + '/Programming/C++/CloudRendering/Data/TransferFunctions/HollowSphere.xml')
    elif test_case == 'Brain':
        vpt.load_transfer_function_file(
            str(pathlib.Path.home()) + '/Programming/C++/CloudRendering/Data/TransferFunctions/BrainDens.xml')
        vpt.load_transfer_function_file_gradient(
            str(pathlib.Path.home()) + '/Programming/C++/CloudRendering/Data/TransferFunctions/BrainGrad.xml')
    elif test_case == 'ToothIso':
        vpt.set_transfer_function_empty()

    if args.render_mode is not None:
        mode = args.render_mode

    denoiser_name = 'None'
    if mode != 'Ray Marching (Emission/Absorption)':
        denoiser_name = 'OptiX Denoiser'
    if args.denoiser is not None:
        if args.denoiser == 'Default':
            denoiser_name = 'OptiX Denoiser'
        else:
            denoiser_name = args.denoiser
    if denoiser_name != 'None' and test_case != 'Cloud' and test_case != 'Cloud Fog':
        vpt.set_denoiser(denoiser_name)
    if denoiser_name.startswith('PyTorch') and args.pytorch_denoiser_model_file is not None:
        vpt.set_pytorch_denoiser_model_file(args.pytorch_denoiser_model_file)

    if args.denoiser_settings is not None:
        for item in args.denoiser_settings:
            items = item.split('=')
            key = items[0].strip()
            if len(items) > 1:
                value = ','.join(items[1:])
                vpt.set_denoiser_property(key, value)

    spp = 256
    if mode == 'Delta Tracking':
        spp = 16384
    elif mode == 'Next Event Tracking':
        spp = 256
    elif mode == 'Isosurfaces':
        spp = 256
    elif mode == 'Ray Marching (Emission/Absorption)':
        spp = 16
    if args.num_samples is not None:
        spp = args.num_samples
    vpt.set_vpt_mode_from_name(mode)

    r_min = None
    r_max = None

    iso_value = 0.0
    if test_case == 'Custom':
        vpt.set_use_transfer_function(args.transfer_function is not None)
        vpt.set_use_isosurfaces(args.iso_value is not None)
        vpt.set_scattering_albedo([args.scattering_albedo, args.scattering_albedo, args.scattering_albedo])
        vpt.set_extinction_scale(args.extinction_scale)
    elif test_case == 'Wholebody':
        vpt.set_use_isosurfaces(True)
        use_gradient_mode = False
        if use_gradient_mode:
            vpt.set_isosurface_type('Gradient')
            iso_value = 0.002
        else:
            vpt.set_isosurface_type('Density')
            iso_value = 0.3
        vpt.set_iso_value(iso_value)
    elif test_case == 'Angiography':
        vpt.set_use_isosurfaces(True)
        vpt.set_isosurface_type('Gradient')
        iso_value = 0.05
        vpt.set_iso_value(iso_value)
    elif test_case == 'HeadDVR':
        vpt.set_use_isosurfaces(False)
        vpt.set_extinction_scale(20.0)
    elif test_case == 'HollowSphere':
        vpt.set_use_isosurfaces(False)
    elif test_case == 'Cloud':
        vpt.set_use_transfer_function(False)
        vpt.set_use_isosurfaces(False)
        vpt.set_scattering_albedo([0.99, 0.99, 0.99])
        vpt.set_extinction_scale(400.0)
        #spp = 2048
    elif test_case == 'Cloud Fog':
        vpt.set_use_transfer_function(False)
        vpt.set_use_isosurfaces(False)
        vpt.set_scattering_albedo([0.99, 0.99, 0.99])
        vpt.set_extinction_scale(8.0)
    if test_case == 'Brain':
        vpt.set_use_isosurfaces(True)
        vpt.set_use_isosurface_tf(True)
        vpt.set_isosurface_type('Density')
        iso_value = 0.05
        vpt.set_iso_value(iso_value)
        if args.use_headlight:
            vpt.set_use_headlight(True)
            vpt.set_use_builtin_environment_map('Black')
            vpt.set_use_headlight_distance(False)
            vpt.set_headlight_intensity(6.0)
    elif test_case == 'ToothIso':
        vpt.set_use_isosurfaces(True)
        iso_value = 0.5
        vpt.set_iso_value(iso_value)
        vpt.set_use_legacy_normals(True)
        r_min = 1.75
        r_max = 1.85

    if args.transfer_function is not None:
        vpt.set_use_transfer_function(True)
        vpt.load_transfer_function_file(args.transfer_function)
        if args.transfer_function_range_min is not None and args.transfer_function_range_max is not None:
            vpt.set_transfer_function_range(args.transfer_function_range_min, args.transfer_function_range_max)
    if args.transfer_function_grad is not None:
        vpt.load_transfer_function_file_gradient(args.transfer_function_grad)
        if args.transfer_function_grad_range_min is not None and args.transfer_function_grad_range_max is not None:
            vpt.set_transfer_function_range_gradient(args.transfer_function_grad_range_min, args.transfer_function_grad_range_max)

    if args.iso_value is not None:
        vpt.set_use_isosurfaces(True)
        if args.iso_value is not None:
            iso_value = args.iso_value
        # TODO
        #if args.use_headlight:
        #    vpt.set_use_headlight(True)
        #    vpt.set_use_builtin_environment_map('Black')
        #    vpt.set_use_headlight_distance(False)
        #    vpt.set_headlight_intensity(6.0)

    vpt.set_iso_surface_color([0.4, 0.4, 0.4])
    vpt.set_surface_brdf('Lambertian')
    if args.brdf is not None:
        vpt.set_surface_brdf(args.brdf)
    if args.brdf_parameters is not None:
        for item in args.brdf_parameters:
            items = item.split('=')
            key = items[0].strip()
            if len(items) > 1:
                value = ','.join(items[1:])
                vpt.set_brdf_parameter(key, value)

    used_feature_maps = ['Cloud Only', 'Background', 'Depth Blended']
    vpt.set_use_feature_maps(used_feature_maps)
    vpt.set_output_foreground_map(True)

    vpt.set_camera_position([0.0, 0.0, 0.3])
    vpt.set_camera_target([0.0, 0.0, 0.0])

    aabb = vpt.get_render_bounding_box()
    rx = 0.5 * (aabb[1] - aabb[0])
    ry = 0.5 * (aabb[3] - aabb[2])
    rz = 0.5 * (aabb[5] - aabb[4])
    aabb_obj = d.AABB3(d.vec3(aabb[0], aabb[2], aabb[4]), d.vec3(aabb[1], aabb[3], aabb[5]))
    radii_sorted = sorted([rx, ry, rz])
    #is_spherical = radii_sorted[2] - radii_sorted[0] < 0.01
    is_spherical = radii_sorted[2] / radii_sorted[0] < 1.9

    start = time.time()

    shall_sample_completely_random_views = True
    use_mixed_mode = True
    use_visibility_aware_sampling = True
    if test_case == 'HeadDVR':
        shall_sample_completely_random_views = False
        use_mixed_mode = False
        use_visibility_aware_sampling = False
    elif test_case == 'Cloud' or test_case == 'Cloud Fog':
        shall_sample_completely_random_views = False
        use_mixed_mode = False
        use_visibility_aware_sampling = False
    elif test_case == 'Wholebody' or test_case == 'Brain':
        shall_sample_completely_random_views = False
        use_mixed_mode = False
        use_visibility_aware_sampling = False
    elif test_case == 'ToothIso':
        shall_sample_completely_random_views = False
        use_mixed_mode = False
        use_visibility_aware_sampling = False
        is_spherical = True
    ds = 2
    vpt.set_secondary_volume_downscaling_factor(ds)
    # use_bos = False  # Bayesian optimal sampling
    use_bos = use_visibility_aware_sampling  # Bayesian optimal sampling
    num_sampled_test_views = 128
    volume_voxel_size = vpt.get_volume_voxel_size()
    vis_volume_voxel_size = [iceil(x, ds) for x in volume_voxel_size]
    vis = None
    occupation_volume = None
    gains = None
    if use_visibility_aware_sampling:
        vis = torch.zeros(
            size=(vis_volume_voxel_size[0], vis_volume_voxel_size[1], vis_volume_voxel_size[2]),
            dtype=torch.float32, device=cuda_device)
        gains = torch.zeros(num_sampled_test_views, device=cpu_device)
    # We need to render one frame before being able to call 'compute_occupation_volume'
    vpt.set_use_feature_maps(['Transmittance Volume'])
    vpt_test_tensor_cuda = vpt.render_frame(test_tensor_cuda, 1)
    vpt_test_tensor_cuda = None
    occupation_volume = vpt.compute_occupation_volume(test_tensor_cuda, ds, 3).cpu().numpy()
    occupation_volume_narrow = vpt.compute_occupation_volume(test_tensor_cuda, ds, 0)
    vpt.set_use_feature_maps(used_feature_maps)
    #occupation_volume_array = occupation_volume.cpu().numpy().astype(np.float32)
    #save_nc('/home/christoph/datasets/Test/occupation.nc', occupation_volume_array)
    fovy = vpt.get_camera_fovy()

    if args.camposes is not None:
        with open(args.camposes, 'r') as f:
            camera_infos = json.load(f)

    if args.num_frames is not None:
        num_frames = args.num_frames
        if args.camposes is not None:
            num_frames = min(num_frames, len(camera_infos))
    else:
        if args.camposes is not None:
            num_frames = len(camera_infos)
        else:
            num_frames = 128

    render_time = 0.0
    for i in range(num_frames):
        if use_mixed_mode:
            use_visibility_aware_sampling = i >= num_frames // 2
            shall_sample_completely_random_views = use_visibility_aware_sampling

        if args.camposes is None and use_visibility_aware_sampling:
            vpt.set_use_feature_maps(['Transmittance Volume'])

            if not use_bos:
                tested_matrices = []
                for view_idx in range(num_sampled_test_views):
                    is_valid = False
                    while not is_valid:
                        if shall_sample_completely_random_views:
                            view_matrix_array, vm, ivm, _ = sample_random_view(aabb_obj)
                        elif is_spherical:
                            view_matrix_array, vm, ivm = sample_view_matrix_circle(aabb_obj, r_min=r_min, r_max=r_max)
                        else:
                            view_matrix_array, vm, ivm = sample_view_matrix_box(aabb_obj)
                        is_valid = check_camera_is_valid(occupation_volume, aabb, vm, ivm, fovy, aspect)
                    vpt.overwrite_camera_view_matrix(view_matrix_array)
                    vpt_test_tensor_cuda = vpt.render_frame(test_tensor_cuda, 1)
                    transmittance_volume_tensor = vpt.get_transmittance_volume(test_tensor_cuda)
                    #transmittance_array = transmittance_volume_tensor.cpu().numpy()
                    #save_nc('/home/christoph/datasets/Test/vis.nc', transmittance_array)
                    gains[view_idx] = (((vis + transmittance_volume_tensor).clamp(0, 1) - vis) * occupation_volume_narrow).sum().cpu()
                    tested_matrices.append((view_matrix_array, vm, ivm))
                # Get the best view
                idx = gains.argmax().item()
                view_matrix_array, vm, ivm = tested_matrices[idx]
            else:
                def sample_camera_pose_gain_function_params(params):
                    view_matrix_array, vm, ivm = sample_random_view_parametrized(params)
                    vpt.overwrite_camera_view_matrix(view_matrix_array)
                    is_valid = check_camera_is_valid(occupation_volume, aabb, vm, ivm, fovy, aspect)
                    if not is_valid:
                        return 0.0
                    vpt_test_tensor_cuda = vpt.render_frame(test_tensor_cuda, spp)
                    transmittance_volume_tensor = vpt.get_transmittance_volume(test_tensor_cuda)
                    gain = (((vis + transmittance_volume_tensor).clamp(0, 1) - vis) * occupation_volume_narrow).sum().cpu()
                    return gain
                def sample_camera_pose_gain_function(cx, cy, cz, u, v, w):
                    params = {
                        'cx': cx, 'cy': cy, 'cz': cz, 'u': u, 'v': v, 'w': w
                    }
                    return sample_camera_pose_gain_function_params(params)
                cam_bounds = get_position_random_range(aabb_obj)
                pbounds = {
                    'cx': (cam_bounds[0][0], cam_bounds[0][1]),
                    'cy': (cam_bounds[1][0], cam_bounds[1][1]),
                    'cz': (cam_bounds[2][0], cam_bounds[2][1]),
                    'u': (0.0, 1.0),
                    'v': (0.0, 1.0),
                    'w': (0.0, 1.0),
                }

                if use_python_bos_optimizer:
                    bayesian_optimizer = BayesianOptimization(
                        f=sample_camera_pose_gain_function,
                        pbounds=pbounds,
                        # random_state=random_state,  # random_state = np.random.RandomState(17)
                    )
                    for view_idx in range(num_sampled_test_views):
                        is_valid = False
                        while not is_valid:
                            view_matrix_array, vm, ivm, params = sample_random_view(aabb_obj)
                            is_valid = check_camera_is_valid(occupation_volume, aabb, vm, ivm, fovy, aspect)
                        bayesian_optimizer.probe(
                           params=params, lazy=True,
                        )
                    acquisition_function = UtilityFunction(kind="ucb", kappa=10)
                    bayesian_optimizer.maximize(
                        init_points=0, n_iter=num_sampled_test_views,
                        acquisition_function=acquisition_function
                    )
                    optimal_gain = bayesian_optimizer.max['target']
                    best_cam_pose = bayesian_optimizer.max['params']
                else:
                    init_points = []
                    for view_idx in range(num_sampled_test_views):
                        is_valid = False
                        while not is_valid:
                            view_matrix_array, vm, ivm, params = sample_random_view(aabb_obj)
                            is_valid = check_camera_is_valid(occupation_volume, aabb, vm, ivm, fovy, aspect)
                        init_points.append(params)
                    settings = pylimbo.BayOptSettings()
                    settings.pbounds = pbounds
                    settings.num_iterations = num_sampled_test_views
                    settings.ucb_kappa = 10
                    optimal_gain, best_cam_pose = pylimbo.maximize(
                        settings, init_points, sample_camera_pose_gain_function_params)

                view_matrix_array, vm, ivm = sample_random_view_parametrized(best_cam_pose)

            vpt.overwrite_camera_view_matrix(view_matrix_array)
            vpt_test_tensor_cuda = vpt.render_frame(test_tensor_cuda, 1)
            transmittance_volume_tensor = vpt.get_transmittance_volume(test_tensor_cuda)
            vis = (vis + transmittance_volume_tensor).clamp(0, 1)
            vpt.set_use_feature_maps(used_feature_maps)
        elif args.camposes is None:
            if shall_sample_completely_random_views:
                view_matrix_array, vm, ivm, _ = sample_random_view(aabb_obj)
            elif is_spherical:
                view_matrix_array, vm, ivm = sample_view_matrix_circle(aabb_obj, r_min=r_min, r_max=r_max)
            else:
                view_matrix_array, vm, ivm = sample_view_matrix_box(aabb_obj)

            vpt.overwrite_camera_view_matrix(view_matrix_array)
            if use_mixed_mode:
                vpt.set_use_feature_maps(['Transmittance Volume'])
                vpt_test_tensor_cuda = vpt.render_frame(test_tensor_cuda, 1)
                transmittance_volume_tensor = vpt.get_transmittance_volume(test_tensor_cuda)
                vis = (vis + transmittance_volume_tensor).clamp(0, 1)
                vpt.set_use_feature_maps(used_feature_maps)

        if args.camposes is None:
            camera_info = dict()
        else:
            camera_info = camera_infos[i]
            #camera_info['position'] = [ivm[i, 3] for i in range(0, 3)]
            #camera_info['rotation'] = [
            #    [ivm[i, 0] for i in range(0, 3)], [ivm[i, 1] for i in range(0, 3)], [ivm[i, 2] for i in range(0, 3)]
            #]
            ivm = np.empty((4, 4))
            for k in range(4):
                ivm[k, 0] = camera_info['rotation'][0][k] if k < 3 else 0.0
                ivm[k, 1] = camera_info['rotation'][1][k] if k < 3 else 0.0
                ivm[k, 2] = camera_info['rotation'][2][k] if k < 3 else 0.0
                ivm[k, 3] = camera_info['position'][k] * args.scale_pos if k < 3 else 1.0
            vm = np.linalg.inv(ivm)
            view_matrix_array = np.empty(16)
            for k in range(4):
                for j in range(4):
                    view_matrix_array[k * 4 + j] = vm[j, k]
            vpt.overwrite_camera_view_matrix(view_matrix_array)
            fovy = camera_info['fovy']
            if abs(fovy - vpt.get_camera_fovy()) > 1e-4:
                vpt.set_camera_fovy(camera_info['fovy'])

        if args.envmap_rot_camera is not None and args.envmap_rot_camera:
            vpt.set_env_map_rot_camera()
        elif args.animate_envmap is not None:
            if num_frames > 1:
                t = i / (num_frames - 1)
            else:
                t = 0.0
            if args.time is not None:
                t = args.time
            if args.animate_envmap == 0:
                animate_envmap_0(t)
            elif args.animate_envmap == 1:
                animate_envmap_1(t)
            elif args.animate_envmap == 2:
                animate_envmap_2(t)
            elif args.animate_envmap == 3:
                animate_envmap_3(t)
            else:
                raise Exception('Error: animate_envmap is out of range')

        #torch.cuda.synchronize()

        #img_name = f'img_{i}.exr'
        #vpt_test_tensor_cuda = vpt.render_frame(test_tensor_cuda, spp)
        #save_tensor_openexr(f'{out_dir}/{img_name}', vpt_test_tensor_cuda.cpu().numpy())

        #fg_name = f'fg_{i}.exr'
        #image_cloud_only = vpt.get_feature_map_from_string(test_tensor_cuda, 'Cloud Only')
        #save_tensor_openexr(f'{out_dir}/{fg_name}', image_cloud_only.cpu().numpy(), use_alpha=True)

        begin_render = time.time()
        if 'img_name' not in camera_info:
            vpt_test_tensor_cuda = vpt.render_frame(test_tensor_cuda, spp)
            render_time += time.time() - begin_render
            if use_png_format:
                fg_name = f'fg_{i}.png'
                bg_name = f'bg_{i}.png'
            else:
                fg_name = f'fg_{i}.exr'
                bg_name = f'bg_{i}.exr'

            if gaussian_splatting_data:
                if use_png_format:
                    save_tensor_png(f'{out_dir}/images/{fg_name}', vpt_test_tensor_cuda.cpu().numpy())
                else:
                    save_tensor_openexr(f'{out_dir}/images/{fg_name}', vpt_test_tensor_cuda.cpu().numpy(), use_alpha=True)
            else:
                save_tensor_openexr(f'{out_dir}/{fg_name}', vpt_test_tensor_cuda.cpu().numpy(), use_alpha=True)

            image_background = vpt.get_feature_map_from_string(test_tensor_cuda, 'Background')
            if gaussian_splatting_data:
                if use_png_format:
                    save_tensor_png(f'{out_dir}/images/{bg_name}', image_background.cpu().numpy())
                else:
                    save_tensor_openexr(f'{out_dir}/images/{bg_name}', image_background.cpu().numpy())
            else:
                save_tensor_openexr(f'{out_dir}/{bg_name}', image_background.cpu().numpy())
            #save_camera_config(f'{out_dir}/intrinsics_{i}.txt', vpt.get_camera_view_matrix())

            depth_name = f'depth_{i}.exr'
            image_depth = vpt.get_feature_map_from_string(test_tensor_cuda, 'Depth Blended')
            #mask = image_depth[1, :, :] > 1e-5
            #image_depth[0, mask] /= image_depth[1, mask]
            if gaussian_splatting_data:
                save_tensor_openexr(f'{out_dir}/images/{depth_name}', image_depth.cpu().numpy())
            else:
                save_tensor_openexr(f'{out_dir}/{depth_name}', image_depth.cpu().numpy())
        else:
            vpt_test_tensor_cuda = vpt.render_frame(test_tensor_cuda, spp)
            render_time += time.time() - begin_render
            image_numpy = vpt_test_tensor_cuda.cpu().numpy()
            if args.use_black_bg:
                image_numpy[3, :, :] = 1.0
            img_name = f'img_{i}.png'
            print(f'{out_dir}/images/{img_name}')
            save_tensor_png(f'{out_dir}/images/{img_name}', image_numpy)

        if args.camposes is None:
            #vm = vpt.get_camera_view_matrix()
            camera_info['id'] = i
            #camera_info['img_name'] = img_name
            camera_info['fg_name'] = fg_name
            camera_info['bg_name'] = bg_name
            camera_info['width'] = image_width
            camera_info['height'] = image_height
            camera_info['position'] = [ivm[i, 3] for i in range(0, 3)]
            camera_info['rotation'] = [
                [ivm[i, 0] for i in range(0, 3)], [ivm[i, 1] for i in range(0, 3)], [ivm[i, 2] for i in range(0, 3)]
            ]
            #camera_info['view_matrix'] = [
            #    [vm[i] for i in range(0, 4)], [vm[i] for i in range(4, 8)],
            #    [vm[i] for i in range(8, 12)], [vm[i] for i in range(12, 16)]
            #]
            camera_info['fovy'] = vpt.get_camera_fovy()
            camera_info['aabb'] = aabb
            if test_case != 'HeadDVR':
                camera_info['iso'] = iso_value
            camera_infos.append(camera_info)
        print(f'{i}/{num_frames}')

        if test_mode:
            #vpt_test_tensor_cpu = vpt.render_frame(test_tensor_cpu, spp)
            #vpt_test_tensor_vulkan = vpt.render_frame(test_tensor_vulkan, spp)
            # print(vpt_test_tensor_cpu)
            print(vpt_test_tensor_cuda)
            # print(vpt_test_tensor_vulkan)
            # plt.imshow(vpt_test_tensor_cpu.permute(1, 2, 0))
            plt.imshow(vpt_test_tensor_cuda.cpu().permute(1, 2, 0))
            plt.show()
            break

    end = time.time()
    print(f'Elapsed time: {end - start}s')

    with open(f'{out_dir}/cameras.json', 'w') as f:
        json.dump(camera_infos, f, ensure_ascii=False, indent=4)

    if args.write_performance_info:
        with open(f'{out_dir}/performance.txt', 'w') as f:
            f.write(f'Total time: {end - start}s for {num_frames} - {(end - start) / num_frames}s/frame\n')
            f.write(f'Render time: {render_time}s for {num_frames} - {render_time / num_frames}s/frame\n')

    vpt.cleanup()
