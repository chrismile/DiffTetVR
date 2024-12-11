import math
import os
import pathlib
import getpass
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import skimage
import skimage.io
import skimage.metrics
import torch
import torchvision
import difftetvr as d
from datasets.imgutils import blend_image_premul
from datasets.sample_view import make_view_matrix


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

    preshaded_path = os.path.join(pathlib.Path.home(), 'Programming/C++/Correrender/Data/VolumeDataSets/preshaded')
    regular_grids_path = '/mnt/data/Flow/Scalar'
    if not os.path.isdir(regular_grids_path):
        regular_grids_path = os.path.join(pathlib.Path.home(), 'datasets/Scalar')
    if not os.path.isdir(regular_grids_path):
        regular_grids_path = os.path.join(pathlib.Path.home(), 'datasets/Flow/Scalar')
    if not os.path.isdir(regular_grids_path):
        regular_grids_path = f'/media/{getpass.getuser()}/Elements/Datasets/Scalar'

    dataset_dir = os.path.join(pathlib.Path.home(), 'datasets/Tet/Test')
    dataset_path_list = os.listdir(dataset_dir)
    metric_name = 'PSNR'

    view_matrix_array = make_view_matrix(
        camera_position=[0.6, 0.0, 0.0],
        camera_right=[0.0, 0.0, -1.0],
        camera_up=[0.0, 1.0, 0.0],
        camera_forward=[1.0, 0.0, 0.0],
    )

    renderer = d.Renderer()
    renderer.set_attenuation(100.0)
    renderer.set_clear_color(d.vec4(0.0, 0.0, 0.0, 0.0))
    renderer.set_camera_fovy(math.atan(1.0 / 2.0) * 2.0)
    renderer.set_viewport_size(512, 512)
    renderer.set_view_matrix(view_matrix_array)

    # renderer_gt = renderer
    renderer_gt = d.RegularGridRenderer()
    renderer_gt.set_attenuation(100.0)
    renderer_gt.set_clear_color(d.vec4(0.0, 0.0, 0.0, 0.0))
    renderer_gt.set_camera_fovy(math.atan(1.0 / 2.0) * 2.0)
    renderer_gt.set_viewport_size(512, 512)
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
    rendered_image_gt = torch.tensor(np.transpose(rendered_image_gt, (2, 0, 1)))

    test_case_color()
    test_case_reg_beta()
    test_case_reg_lambda()
    test_case_pos()
    test_case_num_tets()
