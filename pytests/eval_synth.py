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
from torch.utils.data import DataLoader
import torchvision
import difftetvr as d
from datasets.actions import RendererTypeAction
from datasets.paths import get_nerf_datasets_path
from datasets.imgutils import blend_image_premul
from datasets.sample_view import make_view_matrix
from datasets.nerf_synthetic_dataset import NeRFSyntheticDataset


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


if __name__ == '__main__':
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams.update({'font.family': 'Linux Biolinum O'})
    matplotlib.rcParams.update({'font.size': 17.5})

    parser = argparse.ArgumentParser(
        prog='difftetvr/render.py', description='Renders a tetrahedral mesh using direct volume rendering.')

    # Rendering settings.
    parser.add_argument('--renderer_type', action=RendererTypeAction, default=d.RendererType.PPLL)
    parser.add_argument('--attenuation', type=float, default=25.0)
    parser.add_argument('--device_name', type=str, default=None)

    args = parser.parse_args()

    nerf_datasets_path = get_nerf_datasets_path()

    renderer = d.Renderer(renderer_type=args.renderer_type)
    renderer.set_attenuation(args.attenuation)
    renderer.set_clear_color(d.vec4(0.0, 0.0, 0.0, 0.0))

    nerf_scene_names = ['Lego', 'Ficus', 'Hotdog']
    for nerf_scene_name in nerf_scene_names:
        nerf_scene_name_lower = nerf_scene_name.lower()
        gt_nerf_synthetic_data_path = os.path.join(nerf_datasets_path, f'nerf_synthetic/{nerf_scene_name_lower}')

        dataset_dir = os.path.join(pathlib.Path.home(), f'datasets/Tet/{nerf_scene_name}')
        pathlib.Path(os.path.join(dataset_dir, 'test')).mkdir(parents=False, exist_ok=True)
        tet_mesh = d.TetMesh()
        tet_mesh.load_from_file(os.path.join(dataset_dir, f'{nerf_scene_name_lower}.bintet'))
        renderer.set_tet_mesh(tet_mesh)

        dataset = NeRFSyntheticDataset(
            gt_nerf_synthetic_data_path, images_dir_name='test', used_device=torch.device('cpu'))
        img_width = dataset.get_img_width()
        img_height = dataset.get_img_height()
        renderer.set_viewport_size(img_width, img_height)
        renderer.set_camera_fovy(dataset.get_fovy())
        data_loader = DataLoader(dataset, shuffle=False, batch_size=None)
        img_idx = 0
        for image_gt, view_matrix_array in data_loader:
            renderer.set_view_matrix(view_matrix_array.numpy())
            rendered_image = renderer.render()
            rendered_image = rendered_image.detach().cpu().numpy()
            blend_image_premul(rendered_image, [1.0, 1.0, 1.0, 1.0])
            save_tensor_png(os.path.join(dataset_dir, 'test', f'r_{img_idx}.png'), rendered_image)
            rendered_image = torch.tensor(np.transpose(rendered_image, (2, 0, 1)))
            image_gt = image_gt.permute(2, 0, 1)
            image_metrics = compare_images(image_gt, rendered_image)
            img_idx += 1
