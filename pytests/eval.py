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
import argparse
import pathlib
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import skimage
import skimage.io
import skimage.metrics
import torch
import torchvision
import difftetvr as d
from datasets.actions import RendererTypeAction
from datasets.paths import get_preshaded_path, get_regular_grids_path
from datasets.imgutils import blend_image_premul
from datasets.sample_view import make_view_matrix


def save_tensor_png(file_path, data, convert_to_srgb=False):
    # Convert linear RGB to sRGB.
    if convert_to_srgb:
        for i in range(3):
            data[i, :, :] = np.power(data[i, :, :], 1.0 / 2.2)
    data = np.clip(data, 0.0, 1.0)
    #data = data.transpose(1, 2, 0)
    data = (data * 255).astype('uint8')
    image_out = Image.fromarray(data)
    image_out.save(file_path)


def skimage_to_torch(img):
    t = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    tensor = t(skimage.img_as_float(img)).float()
    tensor = tensor[None, 0:3, :, :] * 2 - 1
    return tensor


def compare_images(tensor_gt, tensor_approx):
    tensor_gt = torch.clip(tensor_gt, 0.0, 1.0)
    tensor_approx = torch.clip(tensor_approx, 0.0, 1.0)

    img_gt = tensor_gt.cpu().numpy().transpose((1, 2, 0))
    img_approx = tensor_approx.cpu().numpy().transpose((1, 2, 0))
    mse = skimage.metrics.mean_squared_error(img_gt, img_approx)
    psnr = skimage.metrics.peak_signal_noise_ratio(img_gt, img_approx)
    data_range = img_gt.max() - img_approx.min()
    ssim = skimage.metrics.structural_similarity(
        img_gt, img_approx, data_range=data_range, channel_axis=-1, multichannel=True)

    return {
        'MSE': mse,
        'RMSE': math.sqrt(mse),
        'PSNR': psnr,
        'SSIM': ssim,
    }


def plot_test_case(test_name, stats_key=None):
    bintet_ext = '.bintet'
    params = []
    for dataset_path in dataset_path_list:
        if dataset_path.startswith(test_name) and dataset_path.endswith(bintet_ext):
            param = dataset_path[len(test_name)+1:len(dataset_path)-len(bintet_ext)]
            if '.' in param or 'e' in param:
                params.append(float(param))
            else:
                params.append(int(param))
    params = sorted(params)
    if stats_key is None:
        params_plot = params
    else:
        params_plot = []
    results = []
    x_params = []
    x_results = []
    for param in params:
        tet_mesh = d.TetMesh()
        tet_mesh.load_from_file(os.path.join(dataset_dir, f'{test_name}_{param}.bintet'))
        renderer.set_tet_mesh(tet_mesh)

        rendered_image = renderer.render()
        rendered_image = rendered_image.detach().cpu().numpy()
        rendered_image = rendered_image[110:400, :, :]
        blend_image_premul(rendered_image, [0.0, 0.0, 0.0, 1.0])
        save_tensor_png(os.path.join(dataset_dir, f'{test_name}_{param}.png'), rendered_image)
        rendered_image = torch.tensor(np.transpose(rendered_image, (2, 0, 1)))
        image_metrics = compare_images(rendered_image_gt, rendered_image)
        results.append(image_metrics[metric_name])

        if stats_key is not None:
            statistics_file_path = os.path.join(dataset_dir, f'{test_name}_{param}.json')
            with open(statistics_file_path) as f:
                stats = json.load(f)
                params_plot.append(stats[stats_key])

        if tet_mesh.check_is_any_tet_degenerate():
            #raise RuntimeError(f'Detected degenerate tetrahedral element in {test_name}_{param}.bintet.')
            print(f'Detected degenerate tetrahedral element in {test_name}_{param}.bintet.')
            x_results.append(results[-1])
            if stats_key is not None:
                x_params.append(params_plot[-1])
            else:
                x_params.append(param)

    plt.cla()
    plt.clf()
    plt.figure(1)
    plt.plot(params_plot, results, label='Random')
    if len(x_results) > 0:
        plt.plot(x_params, x_results, 'x', color='red')


def test_case_color():
    test_name = 'tooth_color'
    plot_test_case(test_name)
    plt.xlabel('Color learning rate')
    plt.ylabel(metric_name)
    plt.tight_layout()
    plt.savefig(os.path.join(dataset_dir, f'{test_name}.pdf'), bbox_inches='tight', pad_inches=0.01)
    plt.show()


def test_case_reg_beta():
    test_name = 'tooth_ctf_reg_beta'
    plot_test_case(test_name)
    plt.xlabel('Regularization beta')
    plt.ylabel(metric_name)
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(dataset_dir, f'{test_name}.pdf'), bbox_inches='tight', pad_inches=0.01)
    plt.show()


def test_case_reg_lambda():
    test_name = 'tooth_ctf_reg_lambda'
    plot_test_case(test_name)
    plt.xlabel('Regularization lambda')
    plt.ylabel(metric_name)
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(dataset_dir, f'{test_name}.pdf'), bbox_inches='tight', pad_inches=0.01)
    plt.show()


def test_case_pos():
    test_name = 'tooth_ctf_pos'
    plot_test_case(test_name)
    plt.xlabel('Position learning rate')
    plt.ylabel(metric_name)
    plt.tight_layout()
    plt.savefig(os.path.join(dataset_dir, f'{test_name}.pdf'), bbox_inches='tight', pad_inches=0.01)
    plt.show()


def test_case_num_tets():
    test_name = 'tooth_ctf_num_tets'
    plot_test_case(test_name, stats_key='num_tets')
    plt.xlabel('#Tets CTF')
    plt.ylabel(metric_name)
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(dataset_dir, f'{test_name}.pdf'), bbox_inches='tight', pad_inches=0.01)
    plt.show()


if __name__ == '__main__':
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams.update({'font.family': 'Linux Biolinum O'})
    matplotlib.rcParams.update({'font.size': 17.5})

    parser = argparse.ArgumentParser(
        prog='difftetvr/render.py', description='Renders a tetrahedral mesh using direct volume rendering.')

    # Rendering settings.
    parser.add_argument('--renderer_type', action=RendererTypeAction, default=d.RendererType.PPLL)
    parser.add_argument('--attenuation', type=float, default=100.0)
    parser.add_argument('--img_width', type=int, default=512)
    parser.add_argument('--img_height', type=int, default=512)

    args = parser.parse_args()

    preshaded_path = get_preshaded_path()
    regular_grids_path = get_regular_grids_path()

    dataset_dir = os.path.join(pathlib.Path.home(), 'datasets/Tet/Test')
    dataset_path_list = os.listdir(dataset_dir)
    metric_name = 'PSNR'

    view_matrix_array = make_view_matrix(
        camera_position=[0.6, 0.0, 0.0],
        camera_right=[0.0, 0.0, -1.0],
        camera_up=[0.0, 1.0, 0.0],
        camera_forward=[1.0, 0.0, 0.0],
    )

    renderer = d.Renderer(renderer_type=args.renderer_type)
    renderer.set_attenuation(args.attenuation)
    renderer.set_clear_color(d.vec4(0.0, 0.0, 0.0, 0.0))
    renderer.set_camera_fovy(math.atan(1.0 / 2.0) * 2.0)
    renderer.set_viewport_size(args.img_width, args.img_height)
    renderer.set_view_matrix(view_matrix_array)

    # renderer_gt = renderer
    renderer_gt = d.RegularGridRenderer()
    renderer_gt.set_attenuation(100.0)
    renderer_gt.set_clear_color(d.vec4(0.0, 0.0, 0.0, 0.0))
    renderer_gt.set_camera_fovy(math.atan(1.0 / 2.0) * 2.0)
    renderer_gt.set_viewport_size(args.img_width, args.img_height)
    renderer_gt.set_view_matrix(view_matrix_array)
    renderer_gt.load_transfer_function_from_file('Tooth3Gauss.xml')

    # tet_mesh_gt = d.TetMesh()
    # tet_mesh_gt.load_from_file(os.path.join(preshaded_dir, 'tooth.bintet'))
    # renderer.set_tet_mesh(tet_mesh_gt)
    regular_grid_gt = d.RegularGrid()
    regular_grid_gt.load_from_file(os.path.join(regular_grids_path, 'Tooth [256 256 161](CT)', 'tooth_cropped.dat'))
    renderer_gt.set_regular_grid(regular_grid_gt)
    rendered_image_gt = renderer_gt.render()
    rendered_image_gt = rendered_image_gt.detach().cpu().numpy()
    rendered_image_gt = rendered_image_gt[110:400, :, :]
    blend_image_premul(rendered_image_gt, [0.0, 0.0, 0.0, 1.0])
    save_tensor_png(os.path.join(dataset_dir, f'gt.png'), rendered_image_gt)
    rendered_image_gt = torch.tensor(np.transpose(rendered_image_gt, (2, 0, 1)))

    test_case_color()
    test_case_reg_beta()
    test_case_reg_lambda()
    test_case_pos()
    test_case_num_tets()
