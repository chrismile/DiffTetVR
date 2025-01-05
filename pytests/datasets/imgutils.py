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
import array
import numpy as np
from PIL import Image
import OpenEXR
import Imath
import numba


def save_array_png(file_path, data, conv_linear_to_srgb=False):
    # Convert linear RGB to sRGB.
    if conv_linear_to_srgb:
        for i in range(3):
            data[i, :, :] = np.power(data[i, :, :], 1.0 / 2.2)
    data = np.clip(data, 0.0, 1.0)
    data = data.transpose(1, 2, 0)
    data = (data * 255).astype('uint8')
    image_out = Image.fromarray(data)
    image_out.save(file_path)


def load_image_array(file_path):
    file_path_lower = file_path.lower()
    if file_path_lower.endswith('.exr'):
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        ref_exr = OpenEXR.InputFile(file_path)
        img = np.array([array.array('f', ref_exr.channel(ch, pt)).tolist() for ch in ("R", "G", "B", "A")], dtype=np.float32)
        dw = ref_exr.header()["dataWindow"]
        img = img.reshape((4, dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1))
        img = img.transpose(1, 2, 0)
        return img
    elif file_path_lower.endswith('.png') or file_path_lower.endswith('.jpg'):
        img = Image.open(file_path)
        img = np.array(img).astype(np.float32) / 255.0
        if img.shape[2] == 3:
            img = np.dstack((img, np.ones(shape=(img.shape[0], img.shape[1], 1), dtype=img.dtype)))
        return img
    else:
        raise RuntimeError(
            f'Error in load_image_array: Unsupported file extension \'{os.path.splitext(file_path)[1]}\'.')


@numba.jit
def convert_premul_to_postmul_alpha(img):
    h, w, c = img.shape
    for y in range(h):
        for x in range(w):
            alpha = img[y, x, 3]
            if alpha > 1e-3:
                img[y, x, 0] /= alpha
                img[y, x, 1] /= alpha
                img[y, x, 2] /= alpha


@numba.jit
def blend_image_premul(img, bg_color):
    bg_alpha = bg_color[3]
    bg_color = np.array([bg_color[0], bg_color[1], bg_color[2]])
    h, w, c = img.shape
    for y in range(h):
        for x in range(w):
            img_alpha = img[y, x, 3]
            img_color = img[y, x, 0:3]
            img[y, x, 3] = img_alpha + (1.0 - img_alpha) * bg_alpha
            img[y, x, 0:3] = img_color + (1.0 - img_alpha) * bg_color


def save_tensor_openexr(file_path, data, dtype=np.float16, use_alpha=False):
    if dtype == np.float32:
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
    elif dtype == np.float16:
        pt = Imath.PixelType(Imath.PixelType.HALF)
    else:
        raise Exception('Error in save_tensor_openexr: Invalid format.')
    if data.dtype != dtype:
        data = data.astype(dtype)
    header = OpenEXR.Header(data.shape[2], data.shape[1])
    if use_alpha:
        header['channels'] = {
            'R': Imath.Channel(pt), 'G': Imath.Channel(pt), 'B': Imath.Channel(pt), 'A': Imath.Channel(pt)
        }
    else:
        header['channels'] = {'R': Imath.Channel(pt), 'G': Imath.Channel(pt), 'B': Imath.Channel(pt)}
    out = OpenEXR.OutputFile(file_path, header)
    reds = data[0, :, :].tobytes()
    greens = data[1, :, :].tobytes()
    blues = data[2, :, :].tobytes()
    if use_alpha:
        alphas = data[3, :, :].tobytes()
        out.writePixels({'R': reds, 'G': greens, 'B': blues, 'A': alphas})
    else:
        out.writePixels({'R': reds, 'G': greens, 'B': blues})


def save_tensor_png(file_path, data):
    # Convert linear RGB to sRGB.
    for i in range(3):
        data[i, :, :] = np.power(data[i, :, :], 1.0 / 2.2)
    data = np.clip(data, 0.0, 1.0)
    data = data.transpose(1, 2, 0)
    data = (data * 255).astype('uint8')
    image_out = Image.fromarray(data)
    image_out.save(file_path)


def convert_image_black_background(input_filename):
    input_image = Image.open(input_filename)
    input_image.putalpha(255)
    input_image.save(input_filename)
